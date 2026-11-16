#!/usr/bin/env python3
"""
Yol takipcisi (gecen seneki aracin path_tracker'i, igvc27_ws'e uyarlandi): pure pursuit.

Girdi:
  /igvc_nav/planned_path (nav_msgs/Path) path_planner'in yolu. Yerel cercevede (base_footprint)
                         gelirse, gelis anindaki odom pozuyla odom cercevesine cevrilir ve saklanir;
                         boylece arac yol uzerinde ilerlerken yol yerinde kalir.
  /odometry/local        (nav_msgs/Odometry) yerel EKF (igvc_localization); odom_topic ile degisir
Cikti:
  /cmd_vel               (geometry_msgs/Twist) igvc_controller'a

Kontrol (degismedi, standart pure pursuit):
  - Yol uzerinde araca en yakin nokta; ondan lookahead_dist kadar ilerideki nokta hedef.
  - angular = kp_angular * yon hatasi (max_angular_speed ile sinirli)
  - linear  = min..max arasi, yon hatasi buyudukce azalir (0.8 rad'da min); hedefe 0.5 m'den
    yakinsa yavaslar, 0.1 m'den yakinsa durur.
  - Hedef aracin arkasindaysa (yerel x < -0.1) min hizla geri gider (planner'in RECOVERY yolu).

Kaldirilanlar: rake_core (Node/config/sistem durumu), durum makinesi modlari (hizalama, su,
cakil, yan egim) ve /mission_control/is_aligned.
Eklenen guvenlik: path_timeout boyunca yeni yol gelmezse ya da odom yoksa durur.

Not: klavye surusu (keyboard_controller) da /cmd_vel'e yayinlar; ikisini ayni anda calistirma.
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class PathTracker(Node):
    def __init__(self):
        super().__init__("path_tracker")
        p = self.declare_parameter
        self.lookahead = float(p("lookahead_dist", 0.8).value)          # m
        self.kp = float(p("kp_angular", 1.0).value)       # 1.0: simulasyonda en iyi (2.5 salinim yapiyordu)
        self.max_v = float(p("max_linear_speed", 1.0).value)             # m/s
        self.min_v = float(p("min_linear_speed", 0.1).value)             # m/s
        self.max_w = float(p("max_angular_speed", 2.0).value)            # rad/s
        self.slow_yaw_error = float(p("slowdown_yaw_error", 0.8).value)  # rad: bu hatada min hiz
        self.path_timeout = float(p("path_timeout", 1.0).value)          # s
        rate = float(p("control_rate", 20.0).value)                      # Hz
        # Yol bu cercevelerden birinde gelirse yerel sayilir (araca bagli)
        self.local_frames = list(p("local_frames", ["base_footprint", "base_link"]).value)
        path_topic = p("path_topic", "/igvc_nav/planned_path").value
        odom_topic = p("odom_topic", "/odometry/local").value   # EKF (igvc_localization)
        cmd_topic = p("cmd_vel_topic", "/cmd_vel").value

        self.robot_x = self.robot_y = self.robot_yaw = 0.0
        self.odom_received = False
        self.global_path = []          # odom cercevesinde [(x, y), ...]
        self.last_path_time = None

        self.create_subscription(Path, path_topic, self.path_callback, 10)
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_timer(1.0 / rate, self.control_loop)
        self.get_logger().info(
            f"Path tracker (pure pursuit): {path_topic} + {odom_topic} -> {cmd_topic} | "
            f"lookahead {self.lookahead} m, kp {self.kp}, v {self.min_v}..{self.max_v} m/s")

    # ------------------------------------------------------------------ girdiler
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.odom_received = True

    def path_callback(self, msg):
        if not self.odom_received:
            return
        self.last_path_time = self.get_clock().now()
        if msg.header.frame_id not in self.local_frames:
            # Zaten odom/map cercevesinde
            self.global_path = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
            return
        # Yerel yol -> odom (gelis anindaki poz)
        c, s = math.cos(self.robot_yaw), math.sin(self.robot_yaw)
        self.global_path = [
            (self.robot_x + ps.pose.position.x * c - ps.pose.position.y * s,
             self.robot_y + ps.pose.position.x * s + ps.pose.position.y * c)
            for ps in msg.poses
        ]

    # ------------------------------------------------------------------ kontrol
    def control_loop(self):
        if not self.odom_received or not self.global_path:
            self.stop_robot()
            return
        age = (self.get_clock().now() - self.last_path_time).nanoseconds * 1e-9
        if age > self.path_timeout:
            self.get_logger().warn(f"Yol {age:.1f} s'dir gelmiyor, duruluyor",
                                   throttle_duration_sec=2.0)
            self.global_path = []
            self.stop_robot()
            return

        path = self.global_path
        # En yakin nokta
        closest_index = min(range(len(path)),
                            key=lambda i: (path[i][0] - self.robot_x) ** 2 + (path[i][1] - self.robot_y) ** 2)

        # Lookahead noktasi (yol boyunca)
        target_pt = path[-1]
        path_dist = 0.0
        for i in range(closest_index + 1, len(path)):
            path_dist += math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            if path_dist > self.lookahead:
                target_pt = path[i]
                break

        dx = target_pt[0] - self.robot_x
        dy = target_pt[1] - self.robot_y
        local_target_x = dx * math.cos(-self.robot_yaw) - dy * math.sin(-self.robot_yaw)
        yaw_error = normalize_angle(math.atan2(dy, dx) - self.robot_yaw)
        dist_to_target = math.hypot(dx, dy)

        twist = Twist()
        if local_target_x < -0.1:
            # Hedef arkada: geri git (RECOVERY yolu)
            twist.angular.z = self.clamp_w(self.kp * normalize_angle(yaw_error - math.pi))
            twist.linear.x = -self.min_v
        else:
            twist.angular.z = self.clamp_w(self.kp * yaw_error)
            speed_factor = max(0.0, 1.0 - abs(yaw_error) / self.slow_yaw_error)
            target_speed = self.min_v + (self.max_v - self.min_v) * speed_factor
            if dist_to_target < 0.5:
                target_speed *= max(0.0, dist_to_target / 0.5)
            twist.linear.x = max(target_speed, self.min_v)

        if dist_to_target < 0.1:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def clamp_w(self, w):
        return max(-self.max_w, min(self.max_w, w))

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = PathTracker()
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
                node.stop_robot()
            except Exception:  # noqa: BLE001
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
