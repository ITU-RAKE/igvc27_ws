#!/usr/bin/env python3
"""
On ve arka 2D lidardan ANLIK, araca bagli (robot-centric) yerel harita.

Hafiza yok: her yayinda harita yalnizca en son taramalardan sifirdan kurulur.
Zamanlayici yok: her tarama gelir gelmez harita/goruntu yayinlanir (en dusuk gecikme).
Bos alan, sensor + isin uclarindan olusan poligonun cv2.fillPoly ile doldurulmasiyla
bulunur (isin isin ornekleme yerine, ~40x hizli).
Engel kalkinca bir sonraki taramada haritadan da kalkar. Odometri/lokalizasyon
kullanilmaz; arac hareket ettikce harita onunla birlikte hareket eder.

Girdi:
  /scan_front, /scan_rear       (sensor_msgs/LaserScan)
Cikti:
  /lidar_map                    (nav_msgs/OccupancyGrid) base_footprint cercevesinde,
                                arac merkezli. -1 bilinmiyor, 0 bos, 100 dolu.
  /scan_merged                  (sensor_msgs/LaserScan) base_footprint'te 360 deg birlesik tarama
  /igvc_vision/lidar_map/debug_image (sensor_msgs/Image, bgr8) haritanin goruntusu; arac
                                dikdortgeni ve on yonu oku ile. Yukari = aracin onu.
  /igvc_vision/lidar_map/debug_image/compressed (sensor_msgs/CompressedImage, jpeg)
                                Ag uzerinden izlemek icin; yalnizca abone varsa kodlanir.

TF: yalnizca lidar cerceveleri -> base_footprint (robot_state_publisher, statik).

Inflation (opsiyonel, varsayilan kapali): lethal_radius_m / inflation_radius_m > 0 ise engeller
igvc_vision/costmap.py ile sisirilir (100 oldurucu, 99..1 azalan maliyet). A*'in kullandigi
/fused_map lidar + kamera engellerini zaten sisirir; burada da acilirsa fused_map girdide
yalnizca 100'u engel saydigi icin cift sisirme olmaz. Calisirken degistirilebilir:
  ros2 param set /lidar_map inflation_radius_m 0.6

Arac olculeri (robot_length/width/center_x) launch dosyasinda igvc_macros.xacro'dan
hesaplanip verilir; elle girmeye gerek yok.
"""
import array
import math

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from igvc_vision.costmap import inflate

UNKNOWN, FREE, OCCUPIED = -1, 0, 100

# Debug goruntu renkleri (BGR)
COLOR_UNKNOWN = (110, 110, 110)
COLOR_FREE = (235, 235, 235)
COLOR_OCCUPIED = (20, 20, 20)
COLOR_ROBOT = (110, 30, 0)     # lacivert
COLOR_ARROW = (255, 140, 30)   # mavi
# Hucre degeri (int8 -> uint8 olarak) -> renk tablosu
COLOR_LUT = np.zeros((256, 3), dtype=np.uint8)
COLOR_LUT[np.uint8(UNKNOWN & 0xFF)] = COLOR_UNKNOWN
COLOR_LUT[FREE] = COLOR_FREE
COLOR_LUT[OCCUPIED] = COLOR_OCCUPIED
# Sisirme maliyeti 1..99: isi haritasi (JET, mavi dusuk -> kirmizi yuksek)
COLOR_LUT[1:100] = cv2.applyColorMap(np.linspace(40, 255, 99).astype(np.uint8).reshape(-1, 1),
                                     cv2.COLORMAP_JET).reshape(-1, 3)


def transform_to_matrix(t):
    """geometry_msgs/TransformStamped -> 4x4 homojen matris."""
    q = t.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    m = np.eye(4)
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    tr = t.transform.translation
    m[:3, 3] = [tr.x, tr.y, tr.z]
    return m


def apply(m, pts):
    """(N,3) noktalara 4x4 donusum uygula."""
    return pts @ m[:3, :3].T + m[:3, 3]


class LidarMap(Node):
    def __init__(self):
        super().__init__('lidar_map')

        p = self.declare_parameter
        self.scan_topics = p('scan_topics', ['/scan_front', '/scan_rear']).value
        self.base_frame = p('base_frame', 'base_footprint').value
        self.res = p('resolution', 0.05).value               # m/hucre
        self.size = p('map_size', 12.0).value                # m, kare, base_frame orijininde merkezli
        self.merged_inc = math.radians(p('merged_angle_increment_deg', 0.375).value)
        self.scan_timeout = p('scan_timeout', 0.5).value     # s, eski taramalar haritaya girmez
        # Arac izdusumu (base_frame'de). Launch, igvc_macros.xacro'dan doldurur.
        self.robot_length = p('robot_length', 0.95).value
        self.robot_width = p('robot_width', 0.79).value
        self.robot_center_x = p('robot_center_x', 0.0).value
        # Arac izdusumu + bu pay icindeki donusler aracin kendisi sayilip atilir
        self.self_margin = p('self_filter_margin', 0.05).value
        self.img_scale = int(p('debug_image_scale', 3).value)  # hucre basina piksel
        # Inflation: 0 = kapali (fused_map sisirir). Acilirsa tipik: 0.445 / 0.618 (xacro'dan)
        self.lethal_r = float(p('lethal_radius_m', 0.0).value)
        self.inflation_r = float(p('inflation_radius_m', 0.0).value)
        self.add_on_set_parameters_callback(self.on_params)
        self.jpeg_quality = int(p('debug_jpeg_quality', 80).value)

        self.n = int(round(self.size / self.res))
        self.half = self.n * self.res / 2.0                  # harita yari genisligi (m)

        self.tf_buffer = Buffer()
        # Ayri spin thread yok: statik TF bir kez okunup onbellege alinir, ana spin yeterli
        # (ayri thread Humble tf2_ros'ta kapanista traceback basiyordu)
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.static_tf = {}                                  # lidar cercevesi -> 4x4 (statik, bir kez)

        # topic -> dict(t=alis zamani, stamp, free=(n,n) bool maske, occ=duz indeksler,
        #               pts=(N,2), rmax, scan_time)
        self.latest = {}
        # Kuyruk derinligi 1: yalnizca en yeni tarama islenir, eski taramalar birikmez (gecikme olmaz)
        scan_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                              reliability=ReliabilityPolicy.BEST_EFFORT)
        for topic in self.scan_topics:
            self.create_subscription(
                LaserScan, topic, lambda msg, t=topic: self.on_scan(msg, t), scan_qos)

        self.map_pub = self.create_publisher(OccupancyGrid, 'lidar_map', 1)
        self.merged_pub = self.create_publisher(LaserScan, 'scan_merged', qos_profile_sensor_data)
        self.img_pub = self.create_publisher(Image, '/igvc_vision/lidar_map/debug_image', 1)
        self.cimg_pub = self.create_publisher(
            CompressedImage, '/igvc_vision/lidar_map/debug_image/compressed', 1)

        self.get_logger().info(
            f"Dinleniyor: {', '.join(self.scan_topics)} -> /lidar_map ({self.base_frame}, "
            f"{self.n}x{self.n} @ {self.res} m, anlik), /scan_merged, /igvc_vision/lidar_map/debug_image. "
            f"Arac: {self.robot_length:.2f}x{self.robot_width:.2f} m, merkez x={self.robot_center_x:.2f}")

    def on_params(self, params):
        """Calisirken inflation ayari."""
        for prm in params:
            if prm.name in ('lethal_radius_m', 'inflation_radius_m'):
                if prm.value < 0:
                    return SetParametersResult(successful=False, reason='negatif olamaz')
                setattr(self, 'lethal_r' if prm.name == 'lethal_radius_m' else 'inflation_r',
                        float(prm.value))
                self.get_logger().info(f'{prm.name} = {prm.value:.3f} m')
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------ TF
    def sensor_tf(self, frame):
        """Lidar -> base_frame donusumu. Sensorler govdeye sabit; bulununca onbellege alinir."""
        if frame not in self.static_tf:
            try:
                self.static_tf[frame] = transform_to_matrix(
                    self.tf_buffer.lookup_transform(self.base_frame, frame, Time()))
            except TransformException as e:
                self.get_logger().warn(f'TF yok {frame} -> {self.base_frame}: {e}',
                                       throttle_duration_sec=2.0)
                return None
        return self.static_tf[frame]

    # ---------------------------------------------------------------- Tarama
    def on_scan(self, msg, topic):
        t_base = self.sensor_tf(msg.header.frame_id)
        if t_base is None:
            return

        r = np.asarray(msg.ranges, dtype=np.float64)
        a = msg.angle_min + np.arange(r.size) * msg.angle_increment
        hit = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
        # Donus yok (inf / range_max ustu): isin boyunca bos, ucu dolu islenmez
        no_return = ~hit & ~np.isnan(r) & ((r > msg.range_max) | np.isposinf(r))
        use = hit | no_return
        if not np.any(use):
            self.latest.pop(topic, None)
            return
        dist = np.where(hit, r, msg.range_max)[use]
        a = a[use]
        is_hit = hit[use]

        pts = np.zeros((dist.size, 3))
        pts[:, 0] = dist * np.cos(a)
        pts[:, 1] = dist * np.sin(a)
        pts = apply(t_base, pts)[:, :2]
        sensor = t_base[:2, 3]

        # Aracin kendi uzerine dusen donusleri at
        hx = self.robot_length / 2 + self.self_margin
        hy = self.robot_width / 2 + self.self_margin
        inside = (np.abs(pts[:, 0] - self.robot_center_x) < hx) & (np.abs(pts[:, 1]) < hy)
        keep = ~(inside & is_hit)
        pts, is_hit = pts[keep], is_hit[keep]

        free, occ = self.trace(sensor, pts, is_hit)
        self.latest[topic] = dict(t=self.get_clock().now(), stamp=msg.header.stamp,
                                  free=free, occ=occ, pts=pts[is_hit], rmax=msg.range_max,
                                  scan_time=msg.scan_time)
        self.publish()

    # ---------------------------------------------------------------- Harita
    def to_cells(self, xy):
        ij = np.floor((xy + self.half) / self.res).astype(np.int64)    # (.,2) -> [x, y]
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.n) & (ij[:, 1] >= 0) & (ij[:, 1] < self.n)
        return ij[ok, 1] * self.n + ij[ok, 0]                          # duz indeks (satir=y)

    def trace(self, origin, ends, is_hit):
        """Bos alan maskesi ve dolu hucreler.
        Isinlar aci sirasinda geldigi icin sensor + uclar yildiz seklinde bir poligondur;
        icini doldurmak her isini tek tek izlemekle ayni sonucu verir."""
        free = np.zeros((self.n, self.n), dtype=np.uint8)       # [satir=y, sutun=x]
        if ends.shape[0] == 0:
            return free.astype(bool), np.empty(0, np.int64)
        poly = np.vstack([origin, ends])
        # Hucre koordinati (sutun=x, satir=y); hucre merkezi = tam sayi. 4 bit alt piksel.
        px = np.round(((poly + self.half) / self.res - 0.5) * 16).astype(np.int32)
        cv2.fillPoly(free, [px], 1, lineType=cv2.LINE_8, shift=4)
        return free.astype(bool), self.to_cells(ends[is_hit])

    def fresh_scans(self):
        now = self.get_clock().now()
        return [v for v in self.latest.values()
                if (now - v['t']).nanoseconds * 1e-9 < self.scan_timeout]

    def publish(self):
        fresh = self.fresh_scans()
        if not fresh:
            return
        # En yeni taramanin zamani: RViz TF'i dogru anda arar
        stamp = max((v['stamp'] for v in fresh), key=lambda s: (s.sec, s.nanosec))

        free = np.zeros((self.n, self.n), dtype=bool)
        for v in fresh:
            free |= v['free']
        grid = np.where(free, FREE, UNKNOWN).astype(np.int8).reshape(-1)
        for v in fresh:                                         # dolu her zaman bosu ezer
            grid[v['occ']] = OCCUPIED
        grid = grid.reshape(self.n, self.n)                     # [satir=y, sutun=x]
        grid = inflate(grid, self.res, self.lethal_r, self.inflation_r)

        self.publish_grid(grid, stamp)
        self.publish_merged(fresh, stamp)
        self.publish_debug_image(grid, stamp)

    def publish_grid(self, grid, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame
        msg.info.map_load_time = stamp
        msg.info.resolution = float(self.res)
        msg.info.width = self.n
        msg.info.height = self.n
        msg.info.origin.position.x = -self.half
        msg.info.origin.position.y = -self.half
        msg.info.origin.orientation.w = 1.0
        # array.array: rclpy list/bytes atamasinda her elemani tek tek kontrol eder (cok yavas)
        msg.data = array.array('b', grid.tobytes())
        self.map_pub.publish(msg)

    # ------------------------------------------------------- Birlesik tarama
    def publish_merged(self, fresh, stamp):
        n_bins = int(round(2 * math.pi / self.merged_inc))
        ranges = np.full(n_bins, np.inf, dtype=np.float32)
        for v in fresh:
            pts = v['pts']
            if pts.size == 0:
                continue
            r = np.hypot(pts[:, 0], pts[:, 1])
            b = ((np.arctan2(pts[:, 1], pts[:, 0]) + math.pi) / self.merged_inc).astype(np.int64) % n_bins
            np.minimum.at(ranges, b, r.astype(np.float32))

        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame
        msg.angle_min = -math.pi
        msg.angle_max = -math.pi + (n_bins - 1) * self.merged_inc
        msg.angle_increment = self.merged_inc
        msg.time_increment = 0.0
        msg.scan_time = float(max(v['scan_time'] for v in fresh))
        msg.range_min = 0.0
        msg.range_max = float(max(v['rmax'] for v in fresh)) + max(self.robot_length, self.robot_width)
        msg.ranges = array.array('f', ranges.tobytes())
        self.merged_pub.publish(msg)

    # --------------------------------------------------------- Debug goruntu
    def to_px(self, x, y):
        """base_frame (x ileri, y sol) -> goruntu pikseli (yukari = ileri, sol = sol)."""
        s = self.img_scale / self.res
        return int(round((self.half - y) * s)), int(round((self.half - x) * s))

    def publish_debug_image(self, grid, stamp):
        want_raw = self.img_pub.get_subscription_count() > 0
        want_jpeg = self.cimg_pub.get_subscription_count() > 0
        if not (want_raw or want_jpeg):
            return

        # grid[y, x] -> goruntu[satir, sutun]: yukari = +x, sol = +y
        img = COLOR_LUT[grid.T[::-1, ::-1].view(np.uint8)]
        if self.img_scale > 1:
            img = cv2.resize(img, None, fx=self.img_scale, fy=self.img_scale,
                             interpolation=cv2.INTER_NEAREST)

        # Arac dikdortgeni (dolu). Olculer tekerlekler dahil en buyuk izdusum
        hl, hw, cx = self.robot_length / 2, self.robot_width / 2, self.robot_center_x
        cv2.rectangle(img, self.to_px(cx + hl, hw), self.to_px(cx - hl, -hw),
                      COLOR_ROBOT, cv2.FILLED)
        # On yonu oku (dikdortgenin icinde, arkadan one)
        cv2.arrowedLine(img, self.to_px(cx - 0.6 * hl, 0), self.to_px(cx + 0.75 * hl, 0),
                        COLOR_ARROW, 2, cv2.LINE_AA, tipLength=0.3)

        if want_raw:
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = self.base_frame
            msg.height, msg.width = img.shape[:2]
            msg.encoding = 'bgr8'
            msg.step = msg.width * 3
            msg.data = array.array('B', img.tobytes())   # bytes atamak ~100x yavas
            self.img_pub.publish(msg)
        if want_jpeg:
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                msg = CompressedImage()
                msg.header.stamp = stamp
                msg.header.frame_id = self.base_frame
                msg.format = 'jpeg'
                msg.data = array.array('B', buf.tobytes())
                self.cimg_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarMap()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C / launch kapanisi
    except Exception:  # noqa: BLE001
        if rclpy.ok():
            raise
        # Kapanis yarisi: sinyal context'i kapatirken bir callback yayin yapiyordu; yut
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
