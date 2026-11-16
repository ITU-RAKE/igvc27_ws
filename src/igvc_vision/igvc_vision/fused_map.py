#!/usr/bin/env python3
"""
Lidar ve kamera occupancy grid'lerini tek bir anlik, arac merkezli haritada birlestirir.

Girdi (ikisi de base_footprint, hafizasiz):
  /lidar_map               (nav_msgs/OccupancyGrid) 5 cm, lidar_map dugumu
  /igvc_vision/camera_map  (nav_msgs/OccupancyGrid) 2 cm, camera_map dugumu
      (/igvc_vision/merged_image_homo/pers_mask ile ayni icerik; grid surumu cozunurluk ve
       orijin tasidigi icin hizalama bundan yapilir)
Cikti:
  /fused_map  (nav_msgs/OccupancyGrid) A* icin maliyet haritasi:
      100 = engel ya da oldurucu bolge (engele lethal_radius'tan yakin),
      99..1 = engelden uzaklastikca azalan maliyet (inflation_radius'a kadar),
      0 = bos, -1 = bilinmiyor
  /igvc_vision/fused_map/debug_image (sensor_msgs/Image, bgr8) yukari = aracin onu.
      Engeller kaynagina gore: siyah = lidar, magenta = kamera, koyu mor = ikisi.
      Sisirme: isi haritasi (JET; mavi dusuk maliyet -> koyu kirmizi oldurucu).
      Acik gri = bos, koyu gri = bilinmiyor; arac lacivert dikdortgen, on yonu mavi ok.

Inflation (igvc_vision/costmap.py): A* araci nokta olarak planlar.
  lethal_radius_m    = ic teget yaricap (yari genislik) + 5 cm = 0.445 m: arac merkezi buraya
                       girerse govde engele kesin carpar -> 100
  inflation_radius_m = cevrel yaricap sqrt((L/2)^2 + (W/2)^2) = 0.618 m (+ launch
                       inflation_gradient_extra_m, varsayilan 0): buraya kadar azalan maliyet.
                       Pay > 0 ise ustel profil (inflation_profile: exp): siyah bolge degismez,
                       gradyan genisler. igvc_course'ta tam turda faydasi gorulmedi (bkz. launch).
Launch ikisini de igvc_macros.xacro'dan hesaplar. Calisirken degistirilebilir:
  ros2 param set /fused_map inflation_radius_m 0.8   (ornegin rampada daraltmak icin)

Lidar golgesi (lidar_shadow, varsayilan acik): lidar engelinin arkasindaki (arac merkezinden
isin boyunca) bolgede kamera verisi atilir; orasi lidar'a gore de bilinmiyorsa bilinmiyor kalir.
Homografi duba gibi yuksek cisimleri zemin sanip arkaya dogru uzatir; o piksellere "bos" demek
dubanin arkasindaki seridi kesik gosteriyordu. Bilinmeyen alan planner icin gecilemez.

Birlestirme kurali (hucre bazinda):
  herhangi bir kaynak dolu (100)    -> dolu   (lidar yuksek engel, kamera serit/kirmizi gorur)
  degilse herhangi bir kaynak bos   -> bos
  degilse                           -> bilinmiyor
Farkli cozunurluk: her kaynagin dolu/bos maskesi, hedef hucre boyutu kadar genisletilip
(dilate; kamera 2 cm -> 5 cm icin 3x3) hedef hucre merkezlerinden cv2.remap ile orneklenir.
Boylece blokta tek dolu hucre bile blogu dolu yapar (ince seritler kaybolmaz). Kaynak daha
kabaysa genisletme yok, en yakin hucre alinir. Esleme grid geometrisi degismedikce bir kez
hesaplanir; her karede is birkac OpenCV cagrisi.

Zamanlayici yok: herhangi bir girdi gelince birlestir ve yayinla (max_publish_rate ile sinirli).
scan/goruntu eskiyse (stale_timeout) o kaynak haritaya katilmaz.
"""
import array
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from igvc_vision.costmap import inflate, shadow_mask

UNKNOWN, FREE, OCCUPIED = -1, 0, 100

# Debug goruntu renkleri (BGR)
COLOR_UNKNOWN = (90, 90, 90)      # koyu gri
COLOR_FREE = (235, 235, 235)      # acik gri
COLOR_LIDAR = (20, 20, 20)        # siyah
COLOR_CAMERA = (220, 30, 220)     # magenta (JET'te yok, sisirmeyle karismaz)
COLOR_BOTH = (90, 0, 90)          # koyu mor
COLOR_ROBOT = (110, 30, 0)        # lacivert
COLOR_ARROW = (255, 140, 30)      # mavi

# Debug renk tablosu: maliyet 0..100 -> renk (0 bos, 1..100 JET), 101 bilinmiyor,
# 102 lidar, 103 kamera, 104 ikisi
_JET = cv2.applyColorMap(np.linspace(40, 255, 100).astype(np.uint8).reshape(-1, 1),
                         cv2.COLORMAP_JET).reshape(-1, 3)
DEBUG_LUT = np.zeros((256, 3), np.uint8)
DEBUG_LUT[0] = COLOR_FREE
DEBUG_LUT[1:101] = _JET
DEBUG_LUT[101], DEBUG_LUT[102], DEBUG_LUT[103], DEBUG_LUT[104] = (
    COLOR_UNKNOWN, COLOR_LIDAR, COLOR_CAMERA, COLOR_BOTH)


class Source:
    """Bir girdi grid'i: son mesaj + hedef grid'e esleme indeksleri."""

    def __init__(self, name):
        self.name = name
        self.msg = None
        self.t = 0.0           # alis zamani [s], dugum saati (sim'de simulasyon saati)
        self.key = None        # esleme hangi geometri icin hesaplandi
        self.map1 = None       # cv2.remap haritalari: hedef hucre -> kaynak hucre
        self.map2 = None
        self.kernel = None     # genisletme cekirdegi (kaynak daha inceyse)


class FusedMap(Node):
    def __init__(self):
        super().__init__('fused_map')
        p = self.declare_parameter
        self.frame = p('frame_id', 'base_footprint').value
        self.res = float(p('resolution', 0.05).value)        # m/hucre
        size = float(p('map_size', 12.0).value)              # m, kare, frame orijininde merkezli
        self.n = int(round(size / self.res))
        self.origin = -self.n * self.res / 2.0
        # Girdide yalnizca 100 engel; 0..99 bos sayilir (girdi zaten sisirilmisse cift sisirme olmaz)
        self.occ_th = int(p('occupied_threshold', 100).value)  # >= ise dolu
        self.free_th = int(p('free_threshold', 100).value)     # 0 <= v < ise bos
        # Inflation (launch xacro'dan: ic teget + pay, cevrel yaricap). 0 = kapali.
        self.lethal_r = float(p('lethal_radius_m', 0.445).value)
        self.inflation_r = float(p('inflation_radius_m', 0.618).value)
        # Gradyan profili: linear | exp (launch: exp, gradyan yarim kosegen + inflation_gradient_extra_m)
        self.profile = p('inflation_profile', 'linear').value
        self.stale = float(p('stale_timeout', 0.5).value)     # s
        # Lidar golgesi: lidar engelinin arkasinda kalan bolgede kameranin "bos" ve "engel"
        # iddialari yok sayilir (bilinmiyor). Homografi yuksek cismi (duba) zemin sanip arkaya
        # dogru uzatir: turuncu kisim "bos" goruldugu icin arkadaki serit kesik gorunuyordu,
        # beyaz bantlar da buyuk sahte engel oluyordu. Gorulmeyen yere bos denmez.
        self.lidar_shadow = bool(p('lidar_shadow', True).value)
        self.shadow_margin = int(round(float(p('shadow_margin_m', 0.10).value) / self.res))
        rate = float(p('max_publish_rate', 30.0).value)
        self.min_period = 1.0 / rate if rate > 0 else 0.0
        lidar_topic = p('lidar_topic', '/lidar_map').value
        camera_topic = p('camera_topic', '/igvc_vision/camera_map').value
        # Arac izdusumu [x0, x1, y0, y1] (launch xacro'dan verir), yalnizca debug cizimi icin
        self.footprint = list(p('footprint_m', [-0.475, 0.475, -0.395, 0.395]).value)
        self.img_scale = int(p('debug_image_scale', 1).value)   # 1: 240x240 (rqt pencereye olcekler)

        self.add_on_set_parameters_callback(self.on_params)
        self.sources = {'lidar': Source('lidar'), 'camera': Source('camera')}
        self.last_pub = 0.0

        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, lidar_topic,
                                 lambda m: self.on_grid('lidar', m), qos)
        self.create_subscription(OccupancyGrid, camera_topic,
                                 lambda m: self.on_grid('camera', m), qos)
        self.map_pub = self.create_publisher(OccupancyGrid, '/fused_map', 1)
        self.img_pub = self.create_publisher(Image, '/igvc_vision/fused_map/debug_image', 1)
        self.get_logger().info(
            f'Birlestiriliyor: {lidar_topic} + {camera_topic} -> /fused_map ({self.frame}, '
            f'{self.n}x{self.n} @ {self.res} m), /igvc_vision/fused_map/debug_image | '
            f'inflation: oldurucu {self.lethal_r:.3f} m, sifira inis {self.inflation_r:.3f} m '
            f'({self.profile})')

    def on_params(self, params):
        """Calisirken inflation ayari (orn. rampa modunda daraltmak icin)."""
        for prm in params:
            if prm.name == 'lidar_shadow':
                self.lidar_shadow = bool(prm.value)
                self.get_logger().info(f'lidar_shadow = {self.lidar_shadow}')
                continue
            if prm.name in ('lethal_radius_m', 'inflation_radius_m'):
                if prm.value < 0:
                    return SetParametersResult(successful=False, reason='negatif olamaz')
                setattr(self, 'lethal_r' if prm.name == 'lethal_radius_m' else 'inflation_r',
                        float(prm.value))
                self.get_logger().info(f'{prm.name} = {prm.value:.3f} m')
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------- esleme
    def build_mapping(self, src, info):
        """Hedef hucre merkezi -> kaynak hucre koordinati (remap) + genisletme cekirdegi."""
        q = info.origin.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        if abs(yaw) > 1e-6:
            self.get_logger().warn(f'{src.name}: donuk grid desteklenmiyor (yaw={yaw:.3f})')
        W, H, rs = info.width, info.height, info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y
        ii, jj = np.mgrid[0:self.n, 0:self.n].astype(np.float32)
        # Kaynak piksel koordinati: hucre merkezi tam sayi (OpenCV)
        map_x = (self.origin + (jj + 0.5) * self.res - ox) / rs - 0.5
        map_y = (self.origin + (ii + 0.5) * self.res - oy) / rs - 0.5
        src.map1, src.map2 = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2, nninterpolation=True)
        ratio = self.res / rs
        k = int(math.ceil(ratio)) | 1 if ratio > 1.001 else 1   # tek sayi
        src.kernel = np.ones((k, k), np.uint8) if k > 1 else None
        self.get_logger().info(
            f'{src.name}: {W}x{H} @ {rs:.3f} m, orijin ({ox:.3f}, {oy:.3f}) -> '
            f'{"genisletme " + str(k) + "x" + str(k) + " + " if k > 1 else ""}ornekleme')

    def resample(self, src):
        """Kaynak grid -> hedef boyutunda (dolu, bos) bool maskeleri."""
        i = src.msg.info
        g = np.frombuffer(src.msg.data, np.int8).reshape(i.height, i.width)
        out = []
        for m in ((g >= self.occ_th), (g >= 0) & (g < self.free_th)):
            m = m.view(np.uint8)
            if src.kernel is not None:
                m = cv2.dilate(m, src.kernel)
            out.append(cv2.remap(m, src.map1, src.map2, cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool))
        return out

    # ------------------------------------------------------------- girdi
    def on_grid(self, name, msg):
        if msg.header.frame_id and msg.header.frame_id != self.frame:
            self.get_logger().error(
                f'{name}: frame {msg.header.frame_id} != {self.frame} (TF donusumu yok)',
                throttle_duration_sec=5.0)
            return
        src = self.sources[name]
        i = msg.info
        key = (i.width, i.height, round(i.resolution, 6), round(i.origin.position.x, 6),
               round(i.origin.position.y, 6), round(i.origin.orientation.z, 6))
        if key != src.key:
            self.build_mapping(src, i)
            src.key = key
        # Eskilik dugum saatiyle (use_sim_time: simulasyon saati). Duvar saati kullanilirsa yuk
        # altinda Gazebo yavaslayinca kamera haritasi duvar saatinde stale_timeout'tan gec gelir ve
        # kamera katmani haritadan anlik duser (kirmizi engeller yanip soner).
        now_sim = self.get_clock().now().nanoseconds * 1e-9
        src.msg, src.t = msg, now_sim

        wall = time.monotonic()                     # yayin hizi siniri duvar saatiyle
        if wall - self.last_pub >= self.min_period:
            self.last_pub = wall
            self.fuse_and_publish(now_sim)

    # ------------------------------------------------------------- birlestirme
    def fuse_and_publish(self, now):
        shape = (self.n, self.n)
        occ, frees = {}, {}
        stamp = None
        for name, src in self.sources.items():
            if src.msg is None or now - src.t > self.stale:
                continue
            o, f = self.resample(src)
            frees[name] = f
            occ[name] = o
            s = src.msg.header.stamp
            if stamp is None or (s.sec, s.nanosec) > (stamp.sec, stamp.nanosec):
                stamp = s
        if stamp is None:
            return

        lidar = occ.get('lidar', np.zeros(shape, bool))
        camera = occ.get('camera', np.zeros(shape, bool))
        cam_free = frees.get('camera', np.zeros(shape, bool))
        if self.lidar_shadow and 'lidar' in occ:
            sh = shadow_mask(lidar, self.shadow_margin)
            camera = camera & ~sh
            cam_free = cam_free & ~sh
        free = frees.get('lidar', np.zeros(shape, bool)) | cam_free
        any_occ = lidar | camera
        grid = np.full(shape, UNKNOWN, np.int8)
        grid[free] = FREE
        grid[any_occ] = OCCUPIED
        grid = inflate(grid, self.res, self.lethal_r, self.inflation_r, self.profile)

        m = OccupancyGrid()
        m.header.stamp = stamp
        m.header.frame_id = self.frame
        m.info.map_load_time = stamp
        m.info.resolution = float(self.res)
        m.info.width = m.info.height = self.n
        m.info.origin.position.x = self.origin
        m.info.origin.position.y = self.origin
        m.info.origin.orientation.w = 1.0
        m.data = array.array('b', grid.tobytes())   # list/bytes atamak rclpy'de cok yavas
        self.map_pub.publish(m)

        if self.img_pub.get_subscription_count() > 0:
            # Maliyet 0..100 dogrudan renk indeksi; ustune bilinmiyor ve kaynak etiketleri
            label = grid.view(np.uint8).copy()
            label[grid == UNKNOWN] = 101
            label[any_occ] = 101 + lidar[any_occ] + 2 * camera[any_occ]
            self.publish_debug(label, stamp)

    # ------------------------------------------------------------- debug
    def to_px(self, x, y):
        """frame (x ileri, y sol) -> goruntu pikseli (yukari = ileri)."""
        s = self.img_scale / self.res
        half = self.n * self.res / 2.0
        return int(round((half - y) * s)), int(round((half - x) * s))

    def publish_debug(self, label, stamp):
        img = DEBUG_LUT[label.T[::-1, ::-1]]          # grid[y, x] -> yukari = +x, sol = +y
        if self.img_scale > 1:
            img = cv2.resize(img, None, fx=self.img_scale, fy=self.img_scale,
                             interpolation=cv2.INTER_NEAREST)
        x0, x1, y0, y1 = self.footprint
        cv2.rectangle(img, self.to_px(x1, y1), self.to_px(x0, y0), COLOR_ROBOT, cv2.FILLED)
        cx, hl = (x0 + x1) / 2, (x1 - x0) / 2
        cv2.arrowedLine(img, self.to_px(cx - 0.6 * hl, 0), self.to_px(cx + 0.75 * hl, 0),
                        COLOR_ARROW, 2, cv2.LINE_AA, tipLength=0.3)
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame
        msg.height, msg.width = img.shape[:2]
        msg.encoding = 'bgr8'
        msg.step = msg.width * 3
        msg.data = array.array('B', img.tobytes())
        self.img_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FusedMap()
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
