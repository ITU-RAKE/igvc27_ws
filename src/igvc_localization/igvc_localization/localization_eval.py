#!/usr/bin/env python3
"""
Lokalizasyon dogruluk testi (simulasyon): gercek konuma karsi hata.

  ros2 run igvc_localization localization_eval --ros-args -p use_sim_time:=true

Girdi:
  /ground_truth/odom  Gazebo gercek pozu (igvc_gazebo.xacro p3d, dunya = map)
  /odometry/global    global EKF (map)       -> dogrudan karsilastirilir
  /odometry/local     yerel EKF (odom)       -> baslangicta hizalanir, o andan beri KAYMA olculur
  /odom               ham tekerlek odometrisi -> ayni sekilde (referans: EKF ne kazandiriyor)
Her 'period' saniyede bir satir, kapanirken (Ctrl+C) ozet: ortalama / RMS / en buyuk hata.

Global icin 'sigma': EKF'nin kendi konum belirsizligi (kovaryanstan). Tutarli bir filtrede hata
cogunlukla 3 sigma'nin altindadir; surekli ustundeyse filtre fazla kendinden emin demektir.
Not: /odometry/global'in map'i ile Gazebo dunyasi ancak gps_odometry datum'u dunya dosyasinin
spherical_coordinates noktasiysa ayni cercevedir (varsayilan config boyle).
"""
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


def pose2d(msg):
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    return p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return -c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2]


class LocalizationEval(Node):
    SOURCES = (('global', '/odometry/global', False), ('local', '/odometry/local', True),
               ('raw', '/odom', True))
    NAMES = {'global': 'global EKF', 'local': 'yerel EKF', 'raw': 'ham odom'}

    def __init__(self):
        super().__init__('localization_eval')
        self.period = float(self.declare_parameter('period', 2.0).value)
        self.truth = None
        self.msgs = {}
        self.align = {}            # odom cercevesi kaynaklari icin hizalama (gercek * inv(kaynak))
        self.errs = {k: [] for k, _, _ in self.SOURCES}
        self.create_subscription(Odometry, '/ground_truth/odom', self.on_truth, 20)
        for key, topic, _ in self.SOURCES:
            self.create_subscription(Odometry, topic, lambda m, k=key: self.msgs.__setitem__(k, m), 20)
        self.create_timer(self.period, self.report)
        self.get_logger().info('Gercek konum (/ground_truth/odom) bekleniyor...')

    def on_truth(self, msg):
        self.truth = msg

    def errors(self):
        t = pose2d(self.truth)
        out = {}
        for key, _, in_odom in self.SOURCES:
            m = self.msgs.get(key)
            if m is None:
                continue
            p = pose2d(m)
            if in_odom:
                if key not in self.align:
                    self.align[key] = compose(t, inverse(p))
                p = compose(self.align[key], p)
            out[key] = (math.hypot(p[0] - t[0], p[1] - t[1]), abs(math.degrees(wrap(p[2] - t[2]))))
        return out

    def report(self):
        if self.truth is None:
            return
        e = self.errors()
        parts = []
        for key, _, _ in self.SOURCES:
            if key not in e:
                parts.append(f'{self.NAMES[key]}: yok')
                continue
            d, a = e[key]
            self.errs[key].append((d, a))
            extra = ''
            if key == 'global':
                c = self.msgs['global'].pose.covariance
                extra = f' (sigma {math.sqrt(max(c[0], 0) + max(c[7], 0)) * 100:.1f} cm)'
            parts.append(f'{self.NAMES[key]} {d * 100:5.1f} cm / {a:4.1f} deg{extra}')
        t = pose2d(self.truth)
        self.get_logger().info(f'gercek ({t[0]:6.2f}, {t[1]:6.2f}) | ' + ' | '.join(parts))

    def summary(self):
        lines = ['OZET (ortalama / RMS / en buyuk; local ve raw: baslangictan beri kayma):']
        for key, _, _ in self.SOURCES:
            e = self.errs[key]
            if not e:
                continue
            d = [x[0] for x in e]
            a = [x[1] for x in e]
            rms = math.sqrt(sum(x * x for x in d) / len(d))
            lines.append(f'  {self.NAMES[key]:10s} konum {sum(d) / len(d) * 100:.1f} / {rms * 100:.1f} / '
                         f'{max(d) * 100:.1f} cm | yon {sum(a) / len(a):.2f} / {max(a):.2f} deg '
                         f'({len(e)} ornek)')
        print('\n'.join(lines), flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationEval()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # noqa: BLE001
        if rclpy.ok():
            raise
        # Kapanis yarisi (sinyal context'i kapatirken bekleme): yut
    finally:
        node.summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
