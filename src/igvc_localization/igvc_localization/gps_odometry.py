#!/usr/bin/env python3
"""
GPS odometrisi: NavSatFix -> map cercevesinde (yerel ENU) anten konumu.

Girdi:  /gps/fix (sensor_msgs/NavSatFix)
Cikti:  /gps/odometry (nav_msgs/Odometry) frame map, child gps_link: anten faz merkezinin
            ENU konumu (x dogu, y kuzey). Kovaryans NavSatFix'ten (zaten ENU), alt sinirli.
            Arac merkezine anten ofseti duzeltmesi global EKF'de (TF base -> gps_link).
        /gps/datum (sensor_msgs/NavSatFix, kalici/latched): map orijininin enlem/boylami.
            GPS waypoint'lerini map'e cevirmek icin: geodesy.EnuFrame(datum).to_enu(lat, lon).

Datum: 'datum' parametresi [enlem, boylam, yukseklik] verilirse o; bos ise ilk 'settle_fixes'
gecerli fix atlandiktan sonraki ilk fix. Simulasyonda dunya dosyasinin spherical_coordinates
noktasi verilir: map = Gazebo dunya cercevesi (gercek konumla dogrudan karsilastirilabilir).

Gecen seneki rake_gps_odom'dan farklar: UTM yerine yerel ENU (meridyen yakinsamasi yok, IMU
yaw'iyla ayni eksen); kovaryans robot yonuyle dondurulmuyor (NavSatFix kovaryansi zaten ENU);
elle yaw_offset yok (Gazebo hatasi dunya dosyasinda heading_deg=180 ile telafi); her fix'te
log yok; use_sim_time koda gomulu degil.
"""
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix, NavSatStatus

from igvc_localization.geodesy import EnuFrame


class GpsOdometry(Node):
    def __init__(self):
        super().__init__('gps_odometry')
        p = self.declare_parameter
        datum = list(p('datum', [0.0]).value)            # [lat, lon, alt] ya da bos/[0]
        self.settle = int(p('settle_fixes', 5).value)
        self.min_std = float(p('min_std', 0.01).value)  # m, kovaryans alt siniri
        self.default_std = float(p('default_std', 2.0).value)  # m, kovaryans bilinmiyorsa
        self.zero_altitude = bool(p('zero_altitude', True).value)
        self.map_frame = p('map_frame', 'map').value
        self.gps_frame = p('gps_frame', 'gps_link').value

        self.enu = None
        self.valid = 0
        datum_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.datum_pub = self.create_publisher(NavSatFix, '/gps/datum', datum_qos)
        self.pub = self.create_publisher(Odometry, '/gps/odometry', 10)
        self.create_subscription(NavSatFix, '/gps/fix', self.on_fix, qos_profile_sensor_data)
        self.dup_timer = self.create_timer(2.0, self.check_duplicate)
        if len(datum) >= 2 and not (datum[0] == 0.0 and datum[1] == 0.0):
            self.set_datum(datum[0], datum[1], datum[2] if len(datum) > 2 else 0.0, 'parametre')
        else:
            self.get_logger().info(f'Datum: ilk gecerli fix ({self.settle} fix bekleniyor)')

    def check_duplicate(self):
        """Ikinci bir gps_odometry varsa (iki kez baslatilmis) bu kopya kapanir: her fix iki kez sayilirdi."""
        self.dup_timer.cancel()
        if self.count_publishers('/gps/odometry') > 1:
            self.get_logger().fatal(
                '/gps/odometry topic\'inde baska bir yayinci var: lokalizasyon zaten calisiyor '
                '(sim.launch.py ekf:=true igvc_localization\'i baslatir). Bu kopya kapaniyor.')
            raise SystemExit(1)

    def set_datum(self, lat, lon, alt, source):
        self.enu = EnuFrame(lat, lon, 0.0 if self.zero_altitude else alt)
        m = NavSatFix()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.map_frame
        m.latitude, m.longitude, m.altitude = lat, lon, alt
        m.status.status = NavSatStatus.STATUS_FIX
        self.datum_pub.publish(m)
        self.get_logger().info(f'Datum ({source}): {lat:.8f}, {lon:.8f}, {alt:.2f} m -> {self.map_frame} orijini')

    def on_fix(self, msg):
        if (msg.status.status < NavSatStatus.STATUS_FIX or math.isnan(msg.latitude)
                or math.isnan(msg.longitude) or (msg.latitude == 0.0 and msg.longitude == 0.0)):
            self.get_logger().warn('Gecerli GPS fix bekleniyor', throttle_duration_sec=5.0)
            return
        if self.enu is None:
            self.valid += 1
            if self.valid <= self.settle:
                return
            self.set_datum(msg.latitude, msg.longitude, msg.altitude, 'ilk fix')
        e, n, u = self.enu.to_enu(msg.latitude, msg.longitude,
                                  None if self.zero_altitude else msg.altitude)

        o = Odometry()
        o.header.stamp = msg.header.stamp
        o.header.frame_id = self.map_frame
        o.child_frame_id = self.gps_frame
        o.pose.pose.position.x, o.pose.pose.position.y = e, n
        o.pose.pose.position.z = 0.0 if self.zero_altitude else u
        o.pose.pose.orientation.w = 1.0
        c = [0.0] * 36
        if msg.position_covariance_type == NavSatFix.COVARIANCE_TYPE_UNKNOWN:
            var = self.default_std ** 2
            c[0], c[7], c[14] = var, var, var
        else:
            pc = msg.position_covariance                   # ENU, satir dizilimi
            for i in range(3):
                for j in range(3):
                    c[i * 6 + j] = pc[i * 3 + j]
        floor = self.min_std ** 2
        for i in (0, 7, 14):
            c[i] = max(c[i], floor)
        c[21] = c[28] = c[35] = 1e6                        # yon bilinmiyor
        o.pose.covariance = c
        self.pub.publish(o)


def main(args=None):
    rclpy.init(args=args)
    node = GpsOdometry()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # noqa: BLE001
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
