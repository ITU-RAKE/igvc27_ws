#!/usr/bin/env python3
"""
Skid steer arac kontrolcusu: cmd_vel -> tekerlek hizlari, enkoder -> odometri.

Girdi:
  /cmd_vel        (geometry_msgs/Twist) linear.x [m/s], angular.z [rad/s]
  /joint_states   (sensor_msgs/JointState) tekerlek konumlari (enkoder) [rad]
Cikti:
  /wheel_velocity_controller/commands (std_msgs/Float64MultiArray) tekerlek hizlari [rad/s],
                  sira: left_wheel_joints + right_wheel_joints (ros2_controllers.yaml ile ayni)
  /odom           (nav_msgs/Odometry) odom -> base_footprint
  TF              odom -> base_footprint (publish_tf: true)
  Odom ve TF'i bu dugum disinda kimse yayinlamaz (Gazebo diff_drive eklentisi kaldirildi).

Kinematik (skid steer; kayma etkin iz genisligiyle modellenir):
  B = wheel_separation * track_scale
  Ters:  w_sol = (v - w*B/2) / r,   w_sag = (v + w*B/2) / r
  Ileri: ds = r*(dth_sol + dth_sag)/2,   dyaw = r*(dth_sag - dth_sol)/B
  track_scale: 2 ceker + 2 caster (simulasyon) icin 1.0. Gercek 4/6 tekerlekli skid steer'de
  yanal kayma yuzunden donus icin daha genis etkin iz gerekir (tipik 1.2 - 2.0): araci yerinde
  360 deg dondur, enkoder yaw'i ile gercek yaw'i karsilastirarak kalibre et.
  Taraf basina birden fazla tekerlek: hepsine ayni komut, odometride ortalama.

Guvenlik: hiz ve ivme sinirlari, tekerlek hizi doygunlugunda egrilik korunur (ikisi birlikte
olceklenir), cmd_timeout boyunca komut gelmezse durur.

Gercek aracta yalnizca 'tekerlek hizi gonder / enkoder oku' katmani (wheel_command_topic,
joint_states) motor surucusuyle degistirilir; kinematik ve odometri ayni kalir.
"""
import math

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster


def yaw_to_quaternion(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def approach(current, target, max_step):
    """current'i target'a en fazla max_step kadar yaklastir (ivme siniri)."""
    if target > current:
        return min(target, current + max_step)
    return max(target, current - max_step)


class SkidSteerController(Node):
    def __init__(self):
        super().__init__('igvc_controller')
        p = self.declare_parameter
        # Geometri (launch igvc_macros.xacro'dan verir)
        self.r = float(p('wheel_radius', 0.10).value)
        self.sep = float(p('wheel_separation', 0.73).value)
        self.track_scale = float(p('track_scale', 1.0).value)
        self.left_joints = list(p('left_wheel_joints', ['left_wheel_joint']).value)
        self.right_joints = list(p('right_wheel_joints', ['right_wheel_joint']).value)
        # Sinirlar
        self.max_v = float(p('max_linear_velocity', 1.5).value)       # m/s
        self.max_w = float(p('max_angular_velocity', 2.0).value)      # rad/s
        self.max_dv = float(p('max_linear_acceleration', 1.0).value)  # m/s^2
        self.max_dw = float(p('max_angular_acceleration', 3.0).value)  # rad/s^2
        self.max_wheel = float(p('max_wheel_velocity', 20.0).value)    # rad/s (URDF max_velocity)
        self.cmd_timeout = float(p('cmd_timeout', 0.5).value)          # s
        rate = float(p('control_rate', 50.0).value)                    # Hz
        # Topic / frame
        cmd_topic = p('cmd_vel_topic', '/cmd_vel').value
        wheel_topic = p('wheel_command_topic', '/wheel_velocity_controller/commands').value
        js_topic = p('joint_states_topic', '/joint_states').value
        self.odom_frame = p('odom_frame', 'odom').value
        self.base_frame = p('base_frame', 'base_footprint').value
        self.publish_tf = bool(p('publish_tf', True).value)
        self.pose_cov = list(p('pose_covariance_diagonal', [0.001, 0.001, 1e6, 1e6, 1e6, 0.01]).value)
        self.twist_cov = list(p('twist_covariance_diagonal', [0.001, 0.001, 1e6, 1e6, 1e6, 0.01]).value)

        self.dt = 1.0 / rate
        self.target_v = self.target_w = 0.0     # istenen (cmd_vel)
        self.v = self.w = 0.0                   # ivme sinirli, uygulanan
        self.last_cmd_time = None
        self.last_tick = None                   # gercek dongu periyodu icin
        # Odometri durumu
        self.x = self.y = self.yaw = 0.0
        self.prev_pos = None                    # {joint: konum}
        self.prev_stamp = None

        self.create_subscription(Twist, cmd_topic, self.on_cmd, 10)
        self.create_subscription(JointState, js_topic, self.on_joint_states, qos_profile_sensor_data)
        self.wheel_pub = self.create_publisher(Float64MultiArray, wheel_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_pub = TransformBroadcaster(self) if self.publish_tf else None
        self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(
            f'Skid steer: r={self.r:.3f} m, iz={self.sep:.3f} m x {self.track_scale:.2f} | '
            f'{cmd_topic} -> {wheel_topic} {self.left_joints + self.right_joints} | '
            f'/odom ({self.odom_frame} -> {self.base_frame}, TF={self.publish_tf}) | '
            f'max {self.max_v} m/s, {self.max_w} rad/s')

    # ------------------------------------------------------------------ komut
    def on_cmd(self, msg):
        self.target_v = max(-self.max_v, min(self.max_v, msg.linear.x))
        self.target_w = max(-self.max_w, min(self.max_w, msg.angular.z))
        self.last_cmd_time = self.get_clock().now()

    def control_loop(self):
        now = self.get_clock().now()
        # Ivme siniri gercek gecen sureyle: zamanlayici yuk altinda gecikirse de dogru ivme
        dt = self.dt if self.last_tick is None else (now - self.last_tick).nanoseconds * 1e-9
        dt = min(max(dt, 0.0), 0.2)
        self.last_tick = now
        if (self.last_cmd_time is None
                or (now - self.last_cmd_time).nanoseconds * 1e-9 > self.cmd_timeout):
            self.target_v = self.target_w = 0.0   # komut kesildi -> dur
        self.v = approach(self.v, self.target_v, self.max_dv * dt)
        self.w = approach(self.w, self.target_w, self.max_dw * dt)

        B = self.sep * self.track_scale
        wl = (self.v - self.w * B / 2.0) / self.r
        wr = (self.v + self.w * B / 2.0) / self.r
        # Doygunluk: ikisini birlikte olcekle (egrilik korunur)
        peak = max(abs(wl), abs(wr))
        if peak > self.max_wheel:
            k = self.max_wheel / peak
            wl, wr = wl * k, wr * k
            self.v, self.w = self.v * k, self.w * k
        msg = Float64MultiArray()
        msg.data = [wl] * len(self.left_joints) + [wr] * len(self.right_joints)
        self.wheel_pub.publish(msg)

    # ------------------------------------------------------------------ odometri
    def on_joint_states(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if any(j not in pos for j in self.left_joints + self.right_joints):
            return
        stamp = Time.from_msg(msg.header.stamp) if (
            msg.header.stamp.sec or msg.header.stamp.nanosec) else self.get_clock().now()
        if self.prev_pos is None:
            self.prev_pos, self.prev_stamp = pos, stamp
            return
        dt = (stamp - self.prev_stamp).nanoseconds * 1e-9
        if dt <= 0.0:
            return
        dl = sum(pos[j] - self.prev_pos[j] for j in self.left_joints) / len(self.left_joints)
        dr = sum(pos[j] - self.prev_pos[j] for j in self.right_joints) / len(self.right_joints)
        self.prev_pos, self.prev_stamp = pos, stamp

        B = self.sep * self.track_scale
        ds = self.r * (dl + dr) / 2.0
        dyaw = self.r * (dr - dl) / B
        # Orta nokta entegrasyonu (2. derece Runge-Kutta)
        mid = self.yaw + dyaw / 2.0
        self.x += ds * math.cos(mid)
        self.y += ds * math.sin(mid)
        self.yaw = math.atan2(math.sin(self.yaw + dyaw), math.cos(self.yaw + dyaw))
        self.publish_odom(stamp.to_msg(), ds / dt, dyaw / dt)

    def publish_odom(self, stamp, v, w):
        qx, qy, qz, qw = yaw_to_quaternion(self.yaw)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y = qx, qy
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = qz, qw
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        for i in range(6):
            odom.pose.covariance[7 * i] = self.pose_cov[i]
            odom.twist.covariance[7 * i] = self.twist_cov[i]
        self.odom_pub.publish(odom)

        if self.tf_pub is not None:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.rotation.x, t.transform.rotation.y = qx, qy
            t.transform.rotation.z, t.transform.rotation.w = qz, qw
            self.tf_pub.sendTransform(t)

    def stop(self):
        """Kapanirken tekerlekleri durdur."""
        msg = Float64MultiArray()
        msg.data = [0.0] * (len(self.left_joints) + len(self.right_joints))
        self.wheel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SkidSteerController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C / launch kapanisi
    except Exception:  # noqa: BLE001
        if rclpy.ok():
            raise
    finally:
        if rclpy.ok():
            try:
                node.stop()
            except Exception:  # noqa: BLE001
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
