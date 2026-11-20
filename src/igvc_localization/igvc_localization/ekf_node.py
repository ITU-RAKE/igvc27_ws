#!/usr/bin/env python3
"""
EKF lokalizasyon dugumu (igvc_localization). Cekirdek: ekf_core.Ekf. Iki ornek calisir:

mode: local  ->  odom -> base_footprint
  Arac odom orijininde, yonu 0 baslar. Surekli (sicramasiz) cerceve: kisa vadeli kontrol,
  path_tracker. Girdi: /odom (tekerlek v, w), /imu/data (jiroskop + sapma, yaw ilk degere bagil).
  Cikti: /odometry/local, TF odom -> base_footprint.
  Bu dugum TF yayinlarken igvc_controller'in TF'i kapali olmali (sim.launch.py ekf:=true yapar).

mode: global ->  map -> odom
  map: GPS datum'unda yerel ENU (gps_odometry). Girdi: yereldekiler + /gps/odometry (anten
  konumu; anten ofseti TF base_footprint -> gps_link ile duzeltilir) + mutlak yon:
    heading_source: imu         BNO085 mutlak yaw (ENU). imu_yaw_offset_deg: manyetik sapma
                                (BNO085 manyetik kuzey verir; ENU = manyetik + sapma, Oakland ~ -7 deg)
    heading_source: gps_course  /gps/vel hiz vektorunun yonu (ileri ve yeterince hizli giderken)
  Ilk GPS fix'i ve mutlak yon gelene kadar baslamaz/yayinlamaz.
  Cikti: /odometry/global (map -> base_footprint), TF map -> odom:
    T_map_odom = T_map_base * inv(T_odom_base)  (yerel poz TF'ten ayni zaman damgasiyla)
  (Eski kodda oteleme dondurulmeden cikariliyordu ve frame'ler uyusmuyordu.)

Yon kilitlenmesinden kurtarma (yaw_reset_after > 0): filtre yonu IMU yonundan kapinin otesine
ayrilirsa (Gazebo'da arac elle dondurulunce, IMU sifirlanip referansi degisince, IMU verisi kesilip
donuste tekerlek patinaji birikince) IMU yaw'i hep reddedilir ve fark hic kapanmaz. Ust uste
yaw_reset_after IMU yaw'i reddedilince:
  local  (bagil yaw): filtre DEGISMEZ, yalnizca IMU'nun bagil referansi (yaw0) filtreye hizalanir.
         odom cercevesi hic ziplamaz (konum gibi yon de "gorulmeden" devam eder; dunyaya hizalama
         global'in isi).
  global (mutlak yaw, heading_source imu): GPS rotasi hakem (arac duz ve >= yaw_arbiter_min_speed
         giderken GPS konumlarinin cizdigi yon, ekf_core.CourseTracker). Rota IMU'ya belirgin
         yakinsa yon IMU'ya, konum GPS'e kilitlenir; filtreye yakinsa IMU bozuk sayilir (manyetik)
         ve reddedilmeye devam eder; belirsizse beklenir. Hakem olmadan yon degistirilmez.

Zamanlama: asenkron sirali guncelleme. Her mesaj kendi zaman damgasina kadar predict edilip
yalnizca bir kez kullanilir (eski global EKF ayni GPS'i odom hizinda tekrar kullaniyordu).
Filtre zamanindan max_delay'den eski mesaj atilir, daha yenisi predict'siz uygulanir.
Yayin her guncellemeden sonra, publish_rate ile sinirli. IMU eksenleri govdeyle hizali (URDF).
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from igvc_localization.ekf_core import BG, CourseTracker, Ekf, PX, PY, V, W, YAW, arbitrate_yaw, wrap


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def roll_pitch_of(q):
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    return roll, pitch


def tilt_lever(lever, roll, pitch):
    """3B anten ofsetini (govde) yatay duzleme izdusur: Ry(pitch) Rx(roll) l -> (x, y) yaw'li
    seviye cercevesinde. Anten ~1 m yuksekte: %15 rampada (9-10 deg) duzeltmesiz ~17 cm hata."""
    lx, ly, lz = lever
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    y1, z1 = cr * ly - sr * lz, sr * ly + cr * lz          # Rx(roll)
    return cp * lx + sp * z1, y1                            # Ry(pitch) (x), y


class EkfNode(Node):
    def __init__(self):
        super().__init__('ekf_node')
        p = self.declare_parameter
        self.mode = p('mode', 'local').value
        if self.mode not in ('local', 'global'):
            raise ValueError(f"mode 'local' ya da 'global' olmali: {self.mode}")
        self.is_global = self.mode == 'global'
        self.world_frame = p('world_frame', 'map' if self.is_global else 'odom').value
        self.odom_frame = p('odom_frame', 'odom').value             # global: map -> odom TF
        self.base_frame = p('base_frame', 'base_footprint').value
        self.publish_tf = bool(p('publish_tf', True).value)
        rate = float(p('publish_rate', 50.0).value)
        self.min_pub_ns = int(1e9 / rate) if rate > 0 else 0
        self.max_delay_ns = int(float(p('max_delay', 0.1).value) * 1e9)
        out_topic = p('output_topic', '/odometry/global' if self.is_global else '/odometry/local').value
        wheel_topic = p('wheel_odom_topic', '/odom').value
        imu_topic = p('imu_topic', '/imu/data').value
        # Olcum gurultuleri (std). Mesaj kovaryanslari genelde gercekci degil -> parametre
        self.wheel_v_std = float(p('wheel_v_std', 0.05).value)        # m/s
        # Skid steer kaymasi donus hiziyla artar: std_w = wheel_w_std + wheel_w_slip * |w|.
        # Dururken/duz giderken tekerlek w'si guvenilir (jiro sapmasini ogretir), donerken
        # kayma yuzunden zayif (jiroskop baskin). Sabit std'de EKF kaymayi sapma sanar.
        self.wheel_w_std = float(p('wheel_w_std', 0.02).value)        # rad/s, taban
        self.wheel_w_slip = float(p('wheel_w_slip', 0.5).value)       # (rad/s) / (rad/s)
        self.gyro_std = float(p('gyro_std', 0.005).value)             # rad/s
        self.use_imu_yaw = bool(p('use_imu_yaw', True).value)
        self.imu_yaw_relative = bool(p('imu_yaw_relative', not self.is_global).value)
        self.yaw_std = float(p('imu_yaw_std', 0.02).value)            # rad
        self.yaw_offset = math.radians(float(p('imu_yaw_offset_deg', 0.0).value))
        # Global: GPS ve yon kaynagi
        self.heading_source = p('heading_source', 'imu').value          # imu | gps_course
        gps_topic = p('gps_odom_topic', '/gps/odometry').value
        gps_vel_topic = p('gps_vel_topic', '/gps/vel').value
        self.course_min_speed = float(p('course_min_speed', 0.4).value)  # m/s
        self.course_max_turn = float(p('course_max_turn', 0.3).value)    # rad/s
        self.course_std = float(p('course_std', 0.05).value)             # rad
        self.gps_lever_fallback = list(p('gps_lever', [0.0, 0.0, 0.0]).value)  # TF yoksa [x, y, z]
        # Surec gurultusu spektral yogunluklari [px, py, yaw, v, w, b_g] (birim^2/s)
        q = list(p('process_noise', [1e-4, 1e-4, 1e-5, 0.5, 1.0, 1e-8]).value)
        self.p0 = list(p('initial_covariance', [1e-6, 1e-6, 1e-6, 0.1, 0.1, 1e-4]).value)
        self.gate = bool(p('outlier_gate', True).value)
        # GPS ust uste bu kadar reddedilirse filtre GPS'ten kopmus demektir (tekerlek patinaji,
        # carpisma, kacirilma): konum GPS'e yeniden kilitlenir. Tek tek sicramalar reddedilmeye
        # devam eder. Kapi tek basina kopan filtreyi bir daha GPS'e dondurmez (kilitlenme).
        self.gps_reset_after = int(p('gps_reset_after', 10).value)     # 10 Hz GPS'te 1 s
        # Yon kilitlenmesinden kurtarma (bkz. dosya basi). 0 = kapali.
        self.yaw_reset_after = int(p('yaw_reset_after', 100).value)    # IMU mesaji (100 Hz'de 1 s)
        self.arb_tol = math.radians(float(p('yaw_arbiter_tol_deg', 20.0).value))
        self.arb_margin = math.radians(float(p('yaw_arbiter_margin_deg', 10.0).value))
        self.course = CourseTracker(
            min_dist=float(p('yaw_arbiter_min_dist', 1.5).value),
            dist_sigma=float(p('yaw_arbiter_dist_sigma', 16.0).value),
            max_turn=float(p('yaw_arbiter_max_turn', 0.3).value),
            min_speed=float(p('yaw_arbiter_min_speed', 0.3).value))
        self.yaw_rejects = 0        # ust uste reddedilen IMU yaw
        self.yaw_suspect = False    # global: IMU ile filtre yonu uyusmuyor, GPS rotasi bekleniyor

        self.ekf = Ekf(q, p0=np.diag(self.p0))
        self.t_ns = None            # filtre zamani
        self.last_pub_ns = 0
        self.yaw0 = None            # bagil yaw referansi (local)
        self.initialized = not self.is_global
        self.last_abs_yaw = None    # global baslatma icin son mutlak yon
        self.lever = None           # GPS anten ofseti (govde cercevesi)
        self.gps_rejects = 0        # ust uste reddedilen GPS
        self.roll = self.pitch = 0.0   # IMU egimi (GPS anten ofseti egim duzeltmesi)
        self.stats = {k: 0 for k in ('wheel', 'gyro', 'yaw', 'gps', 'course', 'late', 'rejected', 'gps_reset',
                                     'yaw_reset')}

        self.create_subscription(Odometry, wheel_topic, self.on_wheel, 20)
        self.create_subscription(Imu, imu_topic, self.on_imu, qos_profile_sensor_data)
        if self.is_global:
            self.create_subscription(Odometry, gps_topic, self.on_gps, 10)
            if self.heading_source == 'gps_course':
                self.create_subscription(Vector3Stamped, gps_vel_topic, self.on_gps_vel,
                                         qos_profile_sensor_data)
            # spin_thread=False: dugumun kendi spin'i yeterli (ayri thread Humble'da kapanista hata basar)
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.pub = self.create_publisher(Odometry, out_topic, 10)
        self.tf_pub = TransformBroadcaster(self) if self.publish_tf else None
        self.create_timer(10.0, self.log_stats)
        self.out_topic = out_topic
        self.dup_timer = self.create_timer(2.0, self.check_duplicate)
        tf_desc = (f'{self.world_frame} -> {self.odom_frame}' if self.is_global
                   else f'{self.world_frame} -> {self.base_frame}')
        extra = (f' + {gps_topic}, yon: {self.heading_source}' if self.is_global else '')
        self.get_logger().info(
            f'EKF ({self.mode}): {wheel_topic} + {imu_topic}{extra} -> {out_topic} '
            f'(TF {tf_desc}: {self.publish_tf}) | imu yaw: '
            f'{"kapali" if not self.use_imu_yaw else ("bagil" if self.imu_yaw_relative else "mutlak")}')

    # ------------------------------------------------------------------ zaman
    def advance(self, stamp):
        """Filtreyi mesaj zamanina ilerlet. False: mesaj cok eski ya da filtre baslamadi."""
        t = Time.from_msg(stamp).nanoseconds
        if t == 0:
            t = self.get_clock().now().nanoseconds
        if not self.initialized:
            return False
        if self.t_ns is None:
            self.t_ns = t
            return True
        if t < self.t_ns:
            if self.t_ns - t > self.max_delay_ns:
                self.stats['late'] += 1
                return False
            return True                       # hafif gecikmeli: predict'siz uygula
        self.ekf.predict((t - self.t_ns) * 1e-9)
        self.t_ns = t
        return True

    def count(self, key, accepted):
        self.stats[key if accepted else 'rejected'] += 1

    # ------------------------------------------------------------------ olcumler
    def on_wheel(self, msg):
        if not self.advance(msg.header.stamp):
            return
        v, w = msg.twist.twist.linear.x, msg.twist.twist.angular.z
        ok, _ = self.ekf.update_wheel(v, w, self.wheel_v_std,
                                      self.wheel_w_std + self.wheel_w_slip * abs(w), gate=self.gate)
        self.count('wheel', ok)
        self.publish()

    def on_imu(self, msg):
        yaw = None
        if msg.orientation_covariance[0] != -1.0:
            self.roll, self.pitch = roll_pitch_of(msg.orientation)
        if self.use_imu_yaw and msg.orientation_covariance[0] != -1.0:
            yaw = wrap(yaw_of(msg.orientation) + self.yaw_offset)
            if self.imu_yaw_relative:
                if self.yaw0 is None:
                    self.yaw0 = yaw
                yaw = wrap(yaw - self.yaw0)
            elif self.heading_source == 'imu':
                self.last_abs_yaw = yaw
        if not self.advance(msg.header.stamp):
            return
        ok, _ = self.ekf.update_gyro(msg.angular_velocity.z, self.gyro_std, gate=self.gate)
        self.count('gyro', ok)
        if yaw is not None and (not self.is_global or self.heading_source == 'imu'):
            ok, _ = self.ekf.update_yaw(yaw, self.yaw_std, gate=self.gate)
            self.count('yaw', ok)
            self.watch_yaw(ok, yaw)
        self.publish()

    def watch_yaw(self, ok, yaw):
        """Yon kilitlenmesi takibi (bkz. dosya basi)."""
        if ok:
            self.yaw_rejects = 0
            if self.yaw_suspect:
                self.yaw_suspect = False
                self.get_logger().info('IMU yonu yeniden filtreyle uyumlu')
            return
        self.yaw_rejects += 1
        if self.yaw_reset_after <= 0 or self.yaw_rejects < self.yaw_reset_after:
            return
        diff = math.degrees(wrap(yaw - self.ekf.x[YAW]))
        if self.imu_yaw_relative:
            # Filtreye dokunma: IMU'nun bagil referansini kaydir (odom ziplamaz)
            self.yaw0 = wrap(self.yaw0 + math.radians(diff))
            self.yaw_rejects = 0
            self.stats['yaw_reset'] += 1
            self.get_logger().warn(
                f'IMU yaw {self.yaw_reset_after} kez ust uste reddedildi (fark {diff:+.1f} deg): IMU '
                f'bagil referansi filtreye hizalandi (odom yonu korunur, ziplama yok)')
        elif not self.yaw_suspect:
            self.yaw_suspect = True
            self.course.reset()
            self.get_logger().warn(
                f'IMU yaw {self.yaw_reset_after} kez ust uste reddedildi (fark {diff:+.1f} deg): '
                f'yon GPS rotasiyla dogrulanacak (arac duz ve >= {self.course.min_speed:.1f} m/s '
                f'gidince)')

    def gps_lever(self, gps_frame):
        """Anten ofseti, egim duzeltmeli (x, y). 3B ofset TF base -> gps_link'ten bir kez okunur;
        her olcumde IMU roll/pitch ile yataya izdusurulur."""
        if self.lever is None:
            try:
                t = self.tf_buffer.lookup_transform(self.base_frame, gps_frame, Time())
                tr = t.transform.translation
                self.lever = (tr.x, tr.y, tr.z)
                self.get_logger().info(
                    f'GPS anten ofseti ({self.base_frame} -> {gps_frame}): x {tr.x:+.3f}, '
                    f'y {tr.y:+.3f}, z {tr.z:+.3f} m (egim duzeltmesi IMU roll/pitch ile)')
            except TransformException:
                self.get_logger().warn(f'TF {self.base_frame} -> {gps_frame} yok, gps_lever '
                                       f'parametresi kullaniliyor', throttle_duration_sec=5.0)
                fb = list(self.gps_lever_fallback) + [0.0, 0.0, 0.0]
                return tilt_lever(fb[:3], self.roll, self.pitch)
        return tilt_lever(self.lever, self.roll, self.pitch)

    def on_gps(self, msg):
        px, py = msg.pose.pose.position.x, msg.pose.pose.position.y
        c = msg.pose.covariance
        cov = np.array([[c[0], c[1]], [c[6], c[7]]])
        lever = self.gps_lever(msg.child_frame_id or 'gps_link')
        if not self.initialized:
            self.try_initialize(msg, px, py, cov, lever)
            return
        if not self.advance(msg.header.stamp):
            return
        if self.arbitrate_heading(msg, px, py, cov, lever):
            self.count('gps', True)
            self.publish()
            return
        ok, d2 = self.ekf.update_position(px, py, cov, lever=lever, gate=self.gate)
        if ok:
            self.gps_rejects = 0
        else:
            self.gps_rejects += 1
            if self.gps_rejects >= self.gps_reset_after:
                x = self.ekf.x
                self.get_logger().warn(
                    f'GPS {self.gps_rejects} kez ust uste reddedildi (Mahalanobis^2 {d2:.0f}): '
                    f'filtre GPS\'ten kopmus, konum GPS\'e yeniden kilitleniyor '
                    f'(({x[PX]:.2f}, {x[PY]:.2f}) -> ({px:.2f}, {py:.2f}))')
                ok, _ = self.ekf.reset_position(px, py, cov, lever=lever)
                self.gps_rejects = 0
                self.stats['gps_reset'] = self.stats.get('gps_reset', 0) + 1
            else:
                self.get_logger().warn(f'GPS aykiri olcum reddedildi (Mahalanobis^2 {d2:.0f})',
                                       throttle_duration_sec=2.0)
        self.count('gps', ok)
        self.publish()

    def arbitrate_heading(self, msg, px, py, cov, lever):
        """Global, heading_source imu: GPS rotasini takip et; yon supheliyse hakemlik yap.
        True: yon ve konum yeniden kilitlendi (bu GPS olcumu kullanildi)."""
        if self.heading_source != 'imu' or self.yaw_reset_after <= 0:
            return False
        x = self.ekf.x
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        std = math.sqrt(max(cov[0, 0], cov[1, 1], 0.0))
        course = self.course.add(t, px, py, x[V], x[W], std)
        if not self.yaw_suspect or course is None or self.last_abs_yaw is None:
            return False
        imu, filt = self.last_abs_yaw, x[YAW]
        verdict = arbitrate_yaw(course, imu, filt, self.arb_tol, self.arb_margin)
        degs = (f'filtre {math.degrees(filt):+.1f}, IMU {math.degrees(imu):+.1f}, '
                f'GPS rotasi {math.degrees(course):+.1f} deg')
        if verdict == 'imu':
            self.ekf.reset_yaw(imu, self.yaw_std)
            self.ekf.reset_position(px, py, cov, lever=lever)
            self.yaw_suspect, self.yaw_rejects, self.gps_rejects = False, 0, 0
            self.course.reset()
            self.stats['yaw_reset'] += 1
            self.get_logger().warn(f'Yon kilitlenmesi: GPS rotasi IMU ile uyumlu ({degs}): '
                                   f'yon IMU\'ya, konum GPS\'e yeniden kilitlendi')
            return True
        if verdict == 'filter':
            self.get_logger().warn(f'IMU yonu GPS rotasiyla uyusmuyor ({degs}): IMU bozuk sayiliyor '
                                   f'(manyetik?), filtre yonu korunuyor', throttle_duration_sec=10.0)
        return False

    def on_gps_vel(self, msg):
        ve, vn = msg.vector.x, msg.vector.y
        speed = math.hypot(ve, vn)
        course = math.atan2(vn, ve)
        if not self.initialized:
            if speed > self.course_min_speed:
                self.last_abs_yaw = course     # ileri gidildigi varsayilir (baslatma)
            return
        x = self.ekf.x
        if speed < self.course_min_speed or abs(x[W]) > self.course_max_turn:
            return
        if x[V] < 0.0:
            course = wrap(course + math.pi)    # geri giderken hiz vektoru arkaya bakar
        if not self.advance(msg.header.stamp):
            return
        ok, _ = self.ekf.update_yaw(course, self.course_std, gate=self.gate)
        self.count('course', ok)

    def try_initialize(self, msg, px, py, cov, lever):
        if self.last_abs_yaw is None:
            self.get_logger().info(f'Global EKF: mutlak yon bekleniyor ({self.heading_source})',
                                   throttle_duration_sec=5.0)
            return
        yaw = self.last_abs_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        x = self.ekf.x
        x[:] = 0.0
        x[PX] = px - (c * lever[0] - s * lever[1])
        x[PY] = py - (s * lever[0] + c * lever[1])
        x[YAW] = yaw
        P = np.diag(self.p0).astype(float)
        P[PX, PX], P[PY, PY] = max(cov[0, 0], 1e-4), max(cov[1, 1], 1e-4)
        P[YAW, YAW] = (self.yaw_std if self.heading_source == 'imu' else self.course_std) ** 2
        self.ekf.P = P
        self.initialized = True
        t = Time.from_msg(msg.header.stamp).nanoseconds
        self.t_ns = t if t else self.get_clock().now().nanoseconds
        self.get_logger().info(f'Global EKF basladi: ({x[PX]:.2f}, {x[PY]:.2f}) m, '
                               f'yon {math.degrees(yaw):.1f} deg ({self.heading_source})')

    # ------------------------------------------------------------------ cikti
    def publish(self):
        if self.t_ns - self.last_pub_ns < self.min_pub_ns:
            return
        self.last_pub_ns = self.t_ns
        x, P = self.ekf.x, self.ekf.P
        stamp = Time(nanoseconds=self.t_ns).to_msg()
        qz, qw = math.sin(x[YAW] / 2.0), math.cos(x[YAW] / 2.0)

        m = Odometry()
        m.header.stamp = stamp
        m.header.frame_id = self.world_frame
        m.child_frame_id = self.base_frame
        m.pose.pose.position.x, m.pose.pose.position.y = float(x[PX]), float(x[PY])
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = qz, qw
        m.twist.twist.linear.x, m.twist.twist.angular.z = float(x[V]), float(x[W])
        pc = np.zeros((6, 6))                 # [x y z roll pitch yaw]
        idx = (PX, PY, YAW)
        for a, ia in zip((0, 1, 5), idx):
            for b, ib in zip((0, 1, 5), idx):
                pc[a, b] = P[ia, ib]
        pc[2, 2] = pc[3, 3] = pc[4, 4] = 1e6  # 2B: z/roll/pitch bilinmiyor
        m.pose.covariance = pc.ravel().tolist()
        tc = np.zeros((6, 6))
        tc[0, 0], tc[5, 5], tc[0, 5], tc[5, 0] = P[V, V], P[W, W], P[V, W], P[W, V]
        tc[1, 1] = 1e-6                        # yanal hiz 0 (holonomik olmayan)
        tc[2, 2] = tc[3, 3] = tc[4, 4] = 1e6
        m.twist.covariance = tc.ravel().tolist()
        self.pub.publish(m)

        if self.tf_pub is None:
            return
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.world_frame
        if not self.is_global:
            t.child_frame_id = self.base_frame
            t.transform.translation.x, t.transform.translation.y = float(x[PX]), float(x[PY])
            t.transform.rotation.z, t.transform.rotation.w = qz, qw
        else:
            ob = self.lookup_odom_base()
            if ob is None:
                return
            ox, oy, oyaw = ob
            # T_map_odom = T_map_base * inv(T_odom_base)
            yaw_mo = wrap(x[YAW] - oyaw)
            c, s = math.cos(yaw_mo), math.sin(yaw_mo)
            t.child_frame_id = self.odom_frame
            t.transform.translation.x = float(x[PX] - (c * ox - s * oy))
            t.transform.translation.y = float(x[PY] - (s * ox + c * oy))
            t.transform.rotation.z, t.transform.rotation.w = math.sin(yaw_mo / 2), math.cos(yaw_mo / 2)
        self.tf_pub.sendTransform(t)

    def lookup_odom_base(self):
        """Yerel poz (odom -> base) filtre zamaninda; yoksa en son mevcut."""
        for when in (Time(nanoseconds=self.t_ns), Time()):
            try:
                tr = self.tf_buffer.lookup_transform(self.odom_frame, self.base_frame, when,
                                                     timeout=Duration(seconds=0.0))
                r = tr.transform
                return r.translation.x, r.translation.y, yaw_of(r.rotation)
            except TransformException:
                continue
        self.get_logger().warn(f'TF {self.odom_frame} -> {self.base_frame} yok (yerel EKF calisiyor mu?)',
                               throttle_duration_sec=5.0)
        return None

    def check_duplicate(self):
        """Ayni cikti topic'inde baska yayinci varsa (dugum iki kez baslatilmis) bu kopya kapanir."""
        self.dup_timer.cancel()
        n = self.count_publishers(self.out_topic)
        if n > 1:
            self.get_logger().fatal(
                f'{self.out_topic} topic\'inde baska bir yayinci var ({n}): lokalizasyon zaten calisiyor. '
                f'sim.launch.py (ekf:=true, varsayilan) igvc_localization\'i kendisi baslatir; ayrica '
                f'igvc_localization.launch.py acmayin. Iki kopya farkli odom cerceveleri ve iki kat GPS '
                f'olcumu demek (TF ziplar, path_tracker sasirir). Bu kopya kapaniyor.')
            raise SystemExit(1)

    def log_stats(self):
        s = self.stats
        msg = ', '.join(f'{k} {v}' for k, v in s.items() if v or k in ('rejected',))
        self.get_logger().info(f'10 s: {msg} | b_g {self.ekf.x[BG]:+.4f} rad/s'
                               + ('' if self.initialized else ' | BASLAMADI'),
                               throttle_duration_sec=9.0)
        for k in s:
            s[k] = 0


def main(args=None):
    rclpy.init(args=args)
    node = EkfNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C / launch kapanisi
    except Exception:  # noqa: BLE001
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
