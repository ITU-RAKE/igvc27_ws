#!/usr/bin/env python3
"""
Kus bakisi goruntusunden renk filtresiyle (varsayilan: kirmizi) anlik occupancy grid.

Girdi:
  /igvc_vision/merged_image_homo (sensor_msgs/Image, bgr8)  surround_view'in kus bakisi.
      Sikistirilmamis topic kullanilir: JPEG renk alt-ornekleme yapar, kirmizi kenarlarda
      sahte pikseller olusturur.
Cikti:
  /igvc_vision/camera_map (nav_msgs/OccupancyGrid)  base_footprint, arac merkezli, anlik.
      100 = filtreye uyan renk (engel), 0 = goruluyor ve renk yok (bos),
      -1 = goruntude yok (menzil disi, kor nokta, aracin kendisi).
  /igvc_vision/merged_image_homo/pers_mask (sensor_msgs/Image, mono8)  debug maskesi:
      255 = engel, 0 = bos, 100 = bilinmiyor. Yalnizca abone varsa uretilir.

Kus bakisi pikseli <-> grid hucresi birebirdir (ayni cozunurluk): ek homografi yok.
BEV: yukari = +x (ileri), sag = -y, merkez = base_footprint.
Hafiza yok, zamanlayici yok: her goruntu gelince isle ve yayinla.

Not: Homografi zemini duz kabul eder; yuksek kirmizi nesneler kus bakisinda (ve haritada)
kameradan uzaga dogru uzar. Yukseklik bilgisi icin lidar haritasini kullan.
"""
import array

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

UNKNOWN, FREE, OCCUPIED = -1, 0, 100
MASK_UNKNOWN, MASK_FREE, MASK_OCCUPIED = 100, 0, 255


class CameraMap(Node):
    def __init__(self):
        super().__init__('camera_map')
        p = self.declare_parameter
        self.input_topic = p('input_topic', '/igvc_vision/merged_image_homo').value
        # BEV geometrisi surround_view ile ayni olmali (launch surround_view.yaml'dan okur)
        self.res = p('bev_resolution_m', 0.02).value
        self.max_range = p('max_range_m', 4.9).value
        # Arac ayak izi [x0, x1, y0, y1] base_footprint'te (launch xacro'dan verir) -> bilinmiyor
        self.footprint = list(p('footprint_m', [-0.475, 0.475, -0.395, 0.395]).value)
        # HSV (OpenCV: H 0-179). Kirmizi hue cemberin iki ucunda: iki aralik birlesir.
        self.lo1 = np.array(p('hsv_lower1', [0, 120, 70]).value, np.uint8)
        self.hi1 = np.array(p('hsv_upper1', [10, 255, 255]).value, np.uint8)
        self.lo2 = np.array(p('hsv_lower2', [170, 120, 70]).value, np.uint8)
        self.hi2 = np.array(p('hsv_upper2', [179, 255, 255]).value, np.uint8)
        k = int(p('morph_kernel', 3).value)   # gurultu temizleme (acma); 0 = kapali
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) if k > 1 else None
        # Gorunur alanin kenarindan bu kadar pikseli at (remap kenar karisimi, menzil dairesi cizgisi)
        self.edge_px = int(p('edge_erode_px', 2).value)
        self.erode_kernel = np.ones((2 * self.edge_px + 1,) * 2, np.uint8)
        self.grid_lut = np.full(256, UNKNOWN, np.int8)
        self.grid_lut[MASK_FREE], self.grid_lut[MASK_OCCUPIED] = FREE, OCCUPIED

        self.static_unknown = None  # boyut ilk goruntude belli olur
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, self.input_topic, self.on_image, qos)
        self.map_pub = self.create_publisher(OccupancyGrid, '/igvc_vision/camera_map', 1)
        self.mask_pub = self.create_publisher(Image, '/igvc_vision/merged_image_homo/pers_mask', 1)
        self.get_logger().info(
            f'Dinleniyor: {self.input_topic} -> /igvc_vision/camera_map, '
            f'/igvc_vision/merged_image_homo/pers_mask | HSV {self.lo1.tolist()}-{self.hi1.tolist()} '
            f've {self.lo2.tolist()}-{self.hi2.tolist()}, {self.res * 100:.0f} cm/hucre')

    def build_static_mask(self, n):
        """Menzil disi + arac ayak izi: goruntuden bagimsiz, bir kez hesaplanir (BEV pikselleri)."""
        c = n / 2.0
        v, u = np.mgrid[0:n, 0:n].astype(np.float32)
        X, Y = (c - v) * self.res, (c - u) * self.res
        x0, x1, y0, y1 = self.footprint
        out = np.hypot(X, Y) > self.max_range - self.edge_px * self.res
        out |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        return out

    def on_image(self, msg):
        if msg.encoding not in ('bgr8', 'rgb8'):
            self.get_logger().error(f'Desteklenmeyen encoding: {msg.encoding} (bgr8 bekleniyor)',
                                    throttle_duration_sec=5.0)
            return
        n = msg.height
        if msg.width != n:
            self.get_logger().error('Kus bakisi kare olmali', throttle_duration_sec=5.0)
            return
        img = np.frombuffer(msg.data, np.uint8).reshape(n, msg.step)[:, :n * 3].reshape(n, n, 3)
        if self.static_unknown is None or self.static_unknown.shape[0] != n:
            self.static_unknown = self.build_static_mask(n)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV if msg.encoding == 'bgr8' else cv2.COLOR_RGB2HSV)
        red = cv2.inRange(hsv, self.lo1, self.hi1) | cv2.inRange(hsv, self.lo2, self.hi2)
        if self.kernel is not None:
            red = cv2.morphologyEx(red, cv2.MORPH_OPEN, self.kernel)

        # Gorulmeyen: remap'in doldurdugu tam siyah (0,0,0) -> HSV V = max(B,G,R) = 0.
        # Kenar karisimini atmak icin gorulen alan asindirilir.
        seen = hsv[:, :, 2]
        if self.edge_px > 0:
            seen = cv2.erode(seen, self.erode_kernel)
        unknown = (seen == 0) | self.static_unknown

        mask = red                       # inRange zaten 0 / 255 (= MASK_FREE / MASK_OCCUPIED)
        mask[unknown] = MASK_UNKNOWN

        self.publish_grid(mask, msg.header, n)
        if self.mask_pub.get_subscription_count() > 0:
            out = Image()
            out.header = msg.header
            out.height = out.width = n
            out.encoding, out.step = 'mono8', n
            out.data = array.array('B', mask.tobytes())   # bytes atamak rclpy'de ~100x yavas
            self.mask_pub.publish(out)

    def publish_grid(self, mask, header, n):
        # BEV[v, u] (yukari=+x, sag=-y) -> grid[satir=y, sutun=x]: iki eksen ters + transpoz
        grid = cv2.flip(mask, -1).T                       # iki eksen ters + transpoz
        grid = self.grid_lut[grid]
        # Hucre merkezleri BEV piksel merkezleriyle cakissin: BEV v -> x = (n/2 - v) * res
        origin = (n / 2.0 - n + 0.5) * self.res
        m = OccupancyGrid()
        m.header.stamp = header.stamp
        m.header.frame_id = header.frame_id or 'base_footprint'
        m.info.map_load_time = header.stamp
        m.info.resolution = float(self.res)
        m.info.width = m.info.height = n
        m.info.origin.position.x = origin
        m.info.origin.position.y = origin
        m.info.origin.orientation.w = 1.0
        m.data = array.array('b', np.ascontiguousarray(grid).tobytes())
        self.map_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = CameraMap()
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
