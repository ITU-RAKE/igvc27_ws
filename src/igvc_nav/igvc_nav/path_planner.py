#!/usr/bin/env python3
"""
Yerel yol planlayici (gecen seneki aracin path_planner'i, igvc27_ws'e uyarlandi).

Hedef secimi, goal_mode parametresi:
  'frontier' (varsayilan, igvc_nav/frontier.py): yalnizca haritanin ust yarisinda (robotun
     ilerisi) arar. Hedef robottan ulasilabilen ve sensor menzilinin sinirina (acik bilinmeyen
     alan) komsu hucrelerden secilir; sonu kapali cep hedef olamaz. Sinira ulasan yol yoksa ust
     yarida cizilebilen en uzun yolun ucu. Yol Dijkstra'dan gelir. Ust yarida hic gidilebilir
     hucre yoksa asagidaki 'bfs' moduna duser. use_frontier=false: frontier aranmaz, hedef her
     zaman en uzun yolun ucu.
  'bfs' (eski algoritma):
  1) Hedef secimi: robotun bulundugu hucreden ileri (+x) ve yana (+-y) BFS, goal_search_depth
     adim. Skor = ileri * x_coeff - yanal sapma * y_coeff + derinlik * depth_coeff
                  - maliyet * goal_selection_cost_coeff. En yuksek skorlu hucre hedef.
  2) Yol: hedefe 8 yonlu, maliyet agirlikli A* (agirlikli sezgisel). Hucre maliyeti
     1 + deger * path_find_cost_coeff / 100; goal_cost_threshold ustu ve bilinmeyen gecilemez.
  3) Yol MIN_SAFE_PATH_LENGTH'ten kisaysa RECOVERY: geri cekilme yolu yayinlanir.
  planner_mode: 'normal' | 'align_for_slider' | 'steep_incline' (eskiden sistem durumu 13/17):
     iki duvarin (serit) ortasindan gecen S-egrisi.

Uyarlamalar (rake_core / ika_utils olmadan):
  - Girdi /fused_map (base_footprint, arac merkezli, igvc_vision). Baslangic hucresi eskiden
    sabit (5, h/2) idi (harita yalnizca onu gosteriyordu); simdi base_footprint orijininin
    hucresi (+ start_offset_m ileri).
  - Arac ayak izi haritada bilinmiyor (-1); A* robotun icinden cikabilsin diye bos sayilir.
  - plan_radius_m (kamera menzili) disi bilinmiyor sayilir: serit cizgileri yalnizca kameranin
    gordugu dairede var, lidar'in "bos" dedigi uzak alan seridin disina yol acmasin.
  - goal_cost_threshold varsayilani 99: /fused_map'te 100 = oldurucu (engel + yarim kosegen),
    99..1 sisirme maliyeti A* hucre maliyetine girer. (Eski 40 esigi eski inflation icindi;
    bizim rampada engelden 1 m'ye kadar her yeri kapatirdi.)
  - Eski kamera haritasindaki "50 = bilinmiyor" ozel durumu kaldirildi (bizde 50 siradan maliyet).
  - Config -> ROS parametreleri; sistem durumu -> planner_mode parametresi.
  - GPS / waypoint / odom kaldirildi (yol cizimini etkilemiyordu; yol base_footprint'te).

Yayinlar:
  /igvc_nav/planned_path  (nav_msgs/Path, base_footprint; pose.position.z = hucre maliyeti 0..99)
  /igvc_nav/goal          (geometry_msgs/PoseStamped)
  /igvc_nav/debug_image   (sensor_msgs/Image, bgr8; + /compressed) yukari = aracin onu.
      beyaz bos, gri tonlari maliyet (koyu = pahali), siyah gecilemez, koyu gri bilinmiyor,
      yesil yol, kirmizi hedef, mavi baslangic, turuncu frontier adaylari, camgobegi "ileri" yonu,
      mor cizgi gidilen iz, acik mor koridor (hedef olamaz), koyu mor izde elenen frontier'ler.
"""
import array
import math
from collections import deque
from heapq import heappop, heappush
from math import sqrt

import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

from igvc_nav import frontier
from igvc_nav.costmap_utils import Costmap

MOTION_PRIMITIVES = [
    (1, 0),  # right
    (-1, 0),  # left
    (0, 1),  # up
    (0, -1),  # down
    (1, 1),  # diagonal right-up
    (1, -1),  # diagonal right-down
    (-1, 1),  # diagonal left-up
    (-1, -1),  # diagonal left-down
]


class PathPlanner(Node):
    def __init__(self):
        super().__init__("path_planner")
        p = self.declare_parameter
        self.map_topic = p("map_topic", "/fused_map").value
        self.goal_cost_threshold = int(p("goal_cost_threshold", 99).value)
        self.x_coeff = float(p("x_coeff", 2.0).value)
        self.y_coeff = float(p("y_coeff", 1.1).value)
        self.depth_coeff = float(p("depth_coeff", 1.2).value)
        self.goal_search_depth = int(p("goal_search_depth", 70).value)      # hucre (5 cm: 2.5 m)
        self.goal_selection_cost_coeff = float(p("goal_selection_cost_coeff", 5.0).value)
        self.path_find_cost_coeff = float(p("path_find_cost_coeff", 8.5).value)
        self.path_find_distance_coeff = float(p("path_find_distance_coeff", 2.0).value)
        self.max_iters = int(p("max_iters", 1000).value)
        self.min_safe_path_length = int(p("min_safe_path_length", 5).value)
        # Mod: normal | align_for_slider | steep_incline (eskiden sistem durumu)
        self.planner_mode = p("planner_mode", "normal").value
        # Hedef secimi: frontier (sonu kapali cepleri eler) | bfs (eski)
        self.goal_mode = p("goal_mode", "frontier").value
        # goal_mode=frontier icinde: True = frontier'e git (yoksa en uzun yol), False = her zaman
        # en uzun yolun ucu. Calisirken: ros2 param set /path_planner use_frontier false
        self.use_frontier = bool(p("use_frontier", True).value)
        self.detour_coeff = float(p("frontier_detour_coeff", 1.0).value)       # dolambac cezasi
        self.frontier_min_goal_dist_m = float(p("frontier_min_goal_dist_m", 1.0).value)
        self.frontier_min_unknown_m2 = float(p("frontier_min_unknown_m2", 0.25).value)
        # Frontier yoksa en uzun yolun ucu: mumkunse bu maliyetin altindaki hucreler
        self.fallback_max_cost = int(p("frontier_fallback_max_cost", 50).value)
        self.wall_cost_threshold = int(p("wall_cost_threshold", 100).value)  # hizalama modu
        self.lane_width_m = float(p("lane_width_m", 3.0).value)              # hizalama modu
        # Baslangic: base_footprint orijini + bu kadar ileri [m]
        self.start_offset_m = float(p("start_offset_m", 0.0).value)
        # Arac ayak izi [x0, x1, y0, y1] base_footprint'te (launch xacro'dan verir)
        self.footprint = list(p("footprint_m", [-0.475, 0.475, -0.395, 0.395]).value)
        self.clear_footprint = bool(p("clear_footprint", True).value)
        # Yalnizca bu yaricap icinde planla, disi bilinmiyor (-1). Kamera menzili: serit cizgileri
        # yalnizca kameranin gordugu dairede var; lidar daha uzagi "bos" gorur ve cizgilerin
        # otesine yol acar (seritten cikis). Launch surround_view.yaml max_range_m'den verir. 0 = kapali.
        self.plan_radius_m = float(p("plan_radius_m", 4.8).value)
        self.plan_mask = None
        # "Ileri" yon sabitleme: ust yari aracin burnuna degil, son heading_anchor_m metrede
        # GIDILEN yone gore. Arac yerinde donunce (dar gecit, viraj, rampa) ust yari da donuyor,
        # parkurun gerisi "ileri" olup arac geri donebiliyordu. Yalnizca ileri hareket kaydedilir
        # (RECOVERY geri cekilmesi yonu cevirmez). 0 = kapali (aracin burnu).
        self.heading_anchor_m = float(p("heading_anchor_m", 0.0).value)
        # Kacis: robotun hucresi oldurucu bolgedeyse RECOVERY yerine bu yaricaptaki en yakin bos
        # hucreden planla. 0 = kapali (kararli eski davranis: RECOVERY ile geri cekilir).
        self.escape_radius_m = float(p("escape_radius_m", 0.0).value)
        # Iz hafizasi ("nereden geldim"): ileri giderken yeni zemindeki konumlar (odom) kaydedilir.
        # Izin koridoruna dusen frontier adaylari hedef olamaz (cepte donunce geldigi yola gitmez).
        # Yol maliyetine KARISMAZ: yol izin ustunden gecebilir, dar alan manevrasi degismez.
        # Ileri hedef yoksa (gercek cikmaz) izde en az geri giden hedef secilir, en fazla
        # trail_max_backtrack_m; daha fazlasi gerekirse arac durur. Hepsi calisirken degistirilebilir:
        #   ros2 param set /path_planner trail_enable false
        self.trail_enable = bool(p("trail_enable", True).value)
        # Koridor yari genisligi: serit 3.05 m, sisirmeden sonra serbest bolge ~+-1.1 m; arac seridin
        # ortasindan 0.3-0.5 m kaysa da karsi kenar koridorda kalsin. Parkurun farkli bolumleri
        # (viraj kollari) birbirinden >= 8 m uzak.
        self.trail_radius_m = float(p("trail_radius_m", 2.5).value)            # 1.6 
        self.trail_length_m = float(p("trail_length_m", 20.0).value)           # tur 113 m: eski iz unutulur
        self.trail_recent_m = float(p("trail_recent_m", 4.0).value)            # son kisim koridorda degil
        self.trail_max_backtrack_m = float(p("trail_max_backtrack_m", 8.0).value)
        # Onde (ust yari) bu kadar sure hedef yoksa arka yariya bakilir (cikmazdan donus)
        self.trail_turn_after_s = float(p("trail_turn_after_s", 2.0).value)
        self.no_goal_since = None
        self.trail_step_m = 0.25
        self.trail = []                      # [(x, y, s)] odom; s = kumulatif mesafe
        self.trail_back = None               # son haritanin koridor rasteri (debug icin de)
        self.trail_pixels = None             # debug: izin hucre koordinatlari (tamami)
        self.odom_topic = p("odom_topic", "/odometry/local").value
        self.track = deque(maxlen=400)       # (x, y) odom, ileri giderken
        self.pose = None                     # (x, y, yaw) odom
        self.path_frame = p("path_frame", "base_footprint").value
        # Debug goruntusu olcegi: 1 = 240x240 (~170 KB). 3x (720x720, ~1.5 MB) ham goruntu
        # DDS'te parcalanip izleyicide kare kaybina yol aciyordu; rqt zaten pencereye olcekler.
        self.debug_scale = int(p("debug_image_scale", 1).value)
        # Planlama: her yeni haritada, harita geri cagrisinin icinde: plan + yol + debug tek seferde.
        # Abonelik derinligi 1: islem surerken gelen haritalarin yalnizca en yenisi beklenir
        # (kuyruk birikmez, gecikme olmaz). Eskiden harita geri cagrisi planliyor, ayri 10 Hz
        # zamanlayici yolu/debug'i yayinliyordu; birbirini bekleyip debug ~5 Hz'e dusuyordu.
        # Zamanlayici yalnizca harita hic yokken bos yol yayinlar (takipci dursun).
        rate = float(p("plan_rate", 2.0).value)

        self.costmap = None
        self.pending_map = None       # son gelen, henuz islenmemis harita
        self.debug_lut = None
        self.map_frame = self.path_frame
        self.best_pos = None
        self.best_score = None
        self.frontier_path = None     # goal_mode=frontier: son haritanin yolu (None = bfs'e dus)
        self.frontier_cells = None    # debug
        self.path = Path()
        self.path.header.frame_id = self.path_frame
        # State Machine Değişkenleri
        self.state = "PLANNING"  # "PLANNING" veya "RECOVERY"

        self.path_pub = self.create_publisher(Path, "/igvc_nav/planned_path", 10)
        self.goal_pub = self.create_publisher(PoseStamped, "/igvc_nav/goal", 10)
        self.debug_img_pub = self.create_publisher(Image, "/igvc_nav/debug_image", 1)
        self.debug_cimg_pub = self.create_publisher(
            CompressedImage, "/igvc_nav/debug_image/compressed", 1)
        self.occupancy_grid_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self.grid_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.create_timer(1.0 / rate, self.create_path)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.add_on_set_parameters_callback(self.on_params)
        self.get_logger().info(
            f"Path planner: {self.map_topic} -> /igvc_nav/planned_path, /igvc_nav/debug_image | "
            f"mod={self.planner_mode}, hedef={self.goal_mode}"
            f"{'' if self.goal_mode != 'frontier' else (' (frontier acik)' if self.use_frontier else ' (frontier kapali: en uzun yol)')}, "
            f"gecilemez > {self.goal_cost_threshold}")

    # ------------------------------------------------------------------ yardimcilar
    def start_pixel(self):
        """base_footprint orijini (+ start_offset_m ileri) -> hucre."""
        sx, sy = self.costmap.poseToPixel(self.start_offset_m, 0.0)
        w, h = self.costmap.pixel_width, self.costmap.pixel_height
        return max(0, min(sx, w - 1)), max(0, min(sy, h - 1))

    def footprint_pixels(self):
        x0, x1, y0, y1 = self.footprint
        c = self.costmap
        (i0, j0), (i1, j1) = c.poseToPixel(x0, y0), c.poseToPixel(x1, y1)
        return (max(0, i0), min(c.pixel_width - 1, i1), max(0, j0), min(c.pixel_height - 1, j1))

    # ------------------------------------------------------------------ debug
    def path_debug_img(self, path, goal):
        if self.costmap is None:
            return None

        g = self.costmap.data
        if self.debug_lut is None:
            # Tek tablo: int8 deger (uint8 gorunumu) -> renk. -1 = 255 indeksi.
            lut = np.zeros((256, 3), np.uint8)
            lut[255] = 90                                   # bilinmiyor: koyu gri
            lut[0] = 255                                    # bos: beyaz
            thr = max(self.goal_cost_threshold, 1)
            v = np.arange(1, thr + 1)
            lut[1:thr + 1] = (235 - v * 175.0 / thr).astype(np.uint8)[:, None]   # maliyet: gri
            lut[thr + 1:128] = 0                            # gecilemez: siyah
            self.debug_lut = lut
        debug_img = self.debug_lut[g.view(np.uint8)]

        tb = self.trail_back
        on_trail = np.isfinite(tb) if tb is not None and tb.shape == g.shape else None
        if on_trail is not None:
            # mor: iz koridoru (alttaki maliyet gorunsun diye karisim), izde elenen frontier'ler koyu mor
            debug_img[on_trail] = (0.55 * debug_img[on_trail] + 0.45 * np.array([200, 60, 200])).astype(np.uint8)
        if self.frontier_cells is not None and self.frontier_cells.shape == g.shape:
            fc = self.frontier_cells if on_trail is None else self.frontier_cells & ~on_trail
            debug_img[fc] = [0, 165, 255]                         # turuncu: frontier adaylari
            if on_trail is not None:
                debug_img[self.frontier_cells & on_trail] = [110, 0, 110]
        if self.trail_pixels is not None and len(self.trail_pixels) > 1:
            cv2.polylines(debug_img, [self.trail_pixels.reshape(-1, 1, 2)], False, (160, 0, 160), 1)
        start = self.start_pixel()
        fx, fy = getattr(self, "fwd", (1.0, 0.0))
        n = math.hypot(fx, fy) or 1.0
        cv2.line(debug_img, start, (int(start[0] + 15 * fx / n), int(start[1] + 15 * fy / n)),
                 (255, 255, 0), 1)                              # camgobegi: sabitlenmis "ileri"
        cv2.circle(debug_img, start, 2, (255, 0, 0), -1)
        if goal is not None:
            cv2.circle(debug_img, (goal[0], goal[1]), 2, (0, 0, 255), -1)
        if path is not None and len(path) > 1:
            cv2.polylines(debug_img, [np.asarray(path, np.int32).reshape(-1, 1, 2)], False,
                          (0, 200, 0), 1)
        return debug_img

    def publish_path_debug(self, debug_img):
        if debug_img is None:
            return
        want_raw = self.debug_img_pub.get_subscription_count() > 0
        want_jpeg = self.debug_cimg_pub.get_subscription_count() > 0
        if not (want_raw or want_jpeg):
            return
        # grid[y, x] -> yukari = +x (ileri), sol = +y  (rotate CCW + yatay flip = iki eksen ters transpoz)
        debug_img = np.ascontiguousarray(debug_img.transpose(1, 0, 2)[::-1, ::-1])
        if self.debug_scale > 1:
            debug_img = cv2.resize(debug_img, None, fx=self.debug_scale, fy=self.debug_scale,
                                   interpolation=cv2.INTER_NEAREST)
        stamp = self.get_clock().now().to_msg()
        if want_raw:
            m = Image()
            m.header.stamp, m.header.frame_id = stamp, self.map_frame
            m.height, m.width = debug_img.shape[:2]
            m.encoding, m.step = "bgr8", debug_img.shape[1] * 3
            m.data = array.array("B", debug_img.tobytes())   # bytes atamak rclpy'de cok yavas
            self.debug_img_pub.publish(m)
        if want_jpeg:
            ok, buf = cv2.imencode(".jpg", debug_img)
            if ok:
                m = CompressedImage()
                m.header.stamp, m.header.frame_id = stamp, self.map_frame
                m.format = "jpeg"
                m.data = array.array("B", buf.tobytes())
                self.debug_cimg_pub.publish(m)

    # ------------------------------------------------------------------ yayin
    def execute_recovery(self):
        """Sadece geri gitmek için sabit bir yol üretir (Tracker bu yolla geri vitese takar)."""
        recovery_path_poses = [
            (0.0, 0.0),  # Mevcut konum
            (-0.5, 0.0),  # 50cm geri
            (-1.0, 0.0),  # 1m geri
        ]
        self.path.poses = self.poses_to_path(recovery_path_poses)
        self.path.header.stamp = self.get_clock().now().to_msg()
        self.path.header.frame_id = self.path_frame
        self.path_pub.publish(self.path)
        self.get_logger().warn("RECOVERY MODU: Robot geri çekiliyor...", throttle_duration_sec=0.5)

    def publish_valid_path(self, path, goal):
        path_poses = [self.costmap.pixelToPose(p[0], p[1]) for p in path]
        d = self.costmap.data
        costs = [max(0, int(d[p[1], p[0]])) for p in path]      # z = hucre maliyeti (takipci hiz icin)
        self.path.poses = self.poses_to_path(path_poses, costs)
        self.path.header.stamp = self.get_clock().now().to_msg()
        self.path.header.frame_id = self.path_frame
        self.path_pub.publish(self.path)
        self.get_logger().info(
            f"PLANNING MODU: Hedefe ilerleniyor (Yol Uzunluğu: {len(path)})",
            throttle_duration_sec=1.0)

    def publish_empty_path(self):
        self.path = Path()
        self.path.header.stamp = self.get_clock().now().to_msg()
        self.path.header.frame_id = self.path_frame
        self.path_pub.publish(self.path)

    def poses_to_path(self, poses, costs=None):
        """costs verilirse pose.position.z = haritadaki hucre maliyeti (0..99). Yol 2B oldugu icin z
        bostur; path_tracker engele yakin bolumlerde yavaslamak icin okur (clearance_slowdown)."""
        out = []
        stamp = self.get_clock().now().to_msg()
        for i, (x, y) in enumerate(poses):
            ps = PoseStamped()
            ps.header.frame_id = self.path_frame
            ps.header.stamp = stamp
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            if costs is not None:
                ps.pose.position.z = float(costs[i])
            ps.pose.orientation.w = 1.0
            out.append(ps)
        return out

    # ------------------------------------------------------------------ planlama
    def create_path(self):
        if self.pending_map is None:
            if self.costmap is None:
                self.publish_empty_path()
            return                    # yeni harita yok: ayni plani tekrar yapma
        msg, self.pending_map = self.pending_map, None
        self.process_map(msg)

        w = self.costmap.pixel_width
        h = self.costmap.pixel_height
        start = self.start_pixel()
        start_x, start_y = start
        data = self.costmap.data

        # =================================================================
        # KAYAR ENGEL HIZALAMA VE DİK EĞİM: KUSURSUZ MERKEZLEME VE S-EĞRİSİ
        # (eskiden sistem durumu 13 / 17; simdi planner_mode parametresi)
        # =================================================================
        is_align_mode = self.planner_mode == "align_for_slider"
        is_steep_incline_mode = self.planner_mode == "steep_incline"

        if is_align_mode or is_steep_incline_mode:
            path = []
            res = self.costmap.resolution if self.costmap.resolution > 0 else 0.05
            half_lane_px = int(self.lane_width_m / 2 / res)
            wall = self.wall_cost_threshold

            # 1. ADIM: Robotun tam burnunun dibindeki (engelsiz bölge) gerçek merkezi bul
            init_left_y = h - 1
            init_right_y = 0
            for y in range(start_y, h):
                if data[y, start_x] >= wall:
                    init_left_y = y
                    break
            for y in range(start_y, -1, -1):
                if data[y, start_x] >= wall:
                    init_right_y = y
                    break

            if init_left_y != h - 1 and init_right_y != 0:
                last_center_y = int((init_left_y + init_right_y) / 2)
            elif init_left_y != h - 1:
                last_center_y = init_left_y - half_lane_px
            elif init_right_y != 0:
                last_center_y = init_right_y + half_lane_px
            else:
                last_center_y = start_y  # Hiçbir şey yoksa varsayılan

            # 2. ADIM: İleriye doğru Dinamik Duvar Beklentisi ile tarama
            if is_steep_incline_mode:
                scan_depth_px = int(0.7 / res)  # dik eğimde sahte duvara çarpmamak için 70 cm
            else:
                scan_depth_px = int(1.5 / res)  # kayar engel için 1.5 m
            tol_px = int(0.30 / res)            # beklenen duvar konumu toleransı (~30 cm)

            for x_px in range(start_x, min(start_x + scan_depth_px, w), 2):
                left_y = h - 1
                right_y = 0
                expected_left_y = last_center_y + half_lane_px
                expected_right_y = last_center_y - half_lane_px

                for y in range(last_center_y, h):
                    if data[y, x_px] >= wall and abs(y - expected_left_y) <= tol_px:
                        left_y = y
                        break
                for y in range(last_center_y, -1, -1):
                    if data[y, x_px] >= wall and abs(y - expected_right_y) <= tol_px:
                        right_y = y
                        break

                found_left = left_y != h - 1
                found_right = right_y != 0
                if found_left and found_right:
                    center_y = int((left_y + right_y) / 2)
                elif found_left:
                    center_y = max(0, min(left_y - half_lane_px, h - 1))
                elif found_right:
                    center_y = max(0, min(right_y + half_lane_px, h - 1))
                else:
                    center_y = last_center_y
                path.append((x_px, center_y))
                last_center_y = center_y

            if len(path) > 3:
                goal = path[-1]
                self.publish_path_debug(self.path_debug_img(path, goal))
                self.publish_valid_path(path, goal)
                return

        # =================================================================
        # NORMAL MOD (Standart A* Algoritması)
        # =================================================================
        if self.best_pos is None:
            self.publish_empty_path()
            return

        goal = self.best_pos
        path = self.frontier_path if self.frontier_path is not None else self.find_path(start, goal)
        path_length = len(path) if path is not None else 0

        self.publish_path_debug(self.path_debug_img(path, goal))

        if self.state == "PLANNING":
            if path_length < self.min_safe_path_length:
                self.get_logger().error(
                    f"ENGEL ÇOK YAKIN! (Yol uzunluğu: {path_length}). RECOVERY'ye geçiliyor.",
                    throttle_duration_sec=1.0)
                self.state = "RECOVERY"
                self.execute_recovery()
            else:
                self.publish_valid_path(path, goal)
        elif self.state == "RECOVERY":
            if path_length >= self.min_safe_path_length:
                self.state = "PLANNING"
                self.publish_valid_path(path, goal)
            else:
                self.execute_recovery()

    def grid_callback(self, msg):
        self.pending_map = msg
        self.create_path()

    def process_map(self, msg):
        new_w = msg.info.width
        new_h = msg.info.height
        if (self.costmap is None or self.costmap.pixel_width != new_w
                or self.costmap.pixel_height != new_h
                or self.costmap.resolution != msg.info.resolution
                or self.costmap.origin_x != msg.info.origin.position.x
                or self.costmap.origin_y != msg.info.origin.position.y):
            self.costmap = Costmap(new_w, new_h, msg.info.resolution,
                                   msg.info.origin.position.x, msg.info.origin.position.y)
        self.map_frame = msg.header.frame_id or self.path_frame
        data = np.frombuffer(msg.data, np.int8).reshape(new_h, new_w).copy()
        if self.clear_footprint:
            # Robotun altindaki bilinmeyen hucreler: A* robotun icinden cikabilsin
            i0, i1, j0, j1 = self.footprint_pixels()
            fp = data[j0:j1 + 1, i0:i1 + 1]
            fp[fp == -1] = 0
        if self.plan_radius_m > 0:
            if self.plan_mask is None or self.plan_mask.shape != data.shape:
                c = self.costmap
                ys, xs = np.mgrid[0:new_h, 0:new_w]
                px = c.origin_x + (xs + 0.5) * c.resolution
                py = c.origin_y + (ys + 0.5) * c.resolution
                self.plan_mask = np.hypot(px, py) > self.plan_radius_m
            data[self.plan_mask] = -1
        self.costmap.update_data(data)

        # Çevre değerlendirmesi (hedef seçimi) her zaman yapılmalı
        self.frontier_path = None
        self.frontier_cells = None
        self.fwd = self.forward_vector()
        if self.goal_mode == "frontier" and self.planner_mode == "normal" and self.select_frontier_goal():
            return
        w = self.costmap.pixel_width
        h = self.costmap.pixel_height
        start_x, start_y = self.start_pixel()

        depth = 0
        temp_best_pos = (start_x, start_y)
        temp_best_score = -100000000
        frontier = {(start_x, start_y)}
        explored = set()

        # Dik eğimde sadece ~60 cm ileriye bak (duvar yanılsamasını önler)
        if self.planner_mode == "steep_incline":
            current_search_depth = int(0.6 / max(self.costmap.resolution, 1e-3))
        else:
            current_search_depth = self.goal_search_depth

        while depth < current_search_depth and len(frontier) > 0:
            next_frontier = set()
            for pos in frontier:
                x, y = pos
                score = (
                    (x - start_x) * self.x_coeff
                    - abs(y - start_y) * self.y_coeff
                    + depth * self.depth_coeff
                    - data[y, x] * self.goal_selection_cost_coeff
                )
                if score > temp_best_score:
                    temp_best_score = score
                    temp_best_pos = (x, y)
                explored.add(pos)

                for dx, dy in ((1, 0), (0, -1), (0, 1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if (nx, ny) not in explored and (nx, ny) not in frontier:
                            val = data[ny, nx]
                            if val != -1 and val <= self.goal_cost_threshold:
                                next_frontier.add((nx, ny))
            frontier = next_frontier
            depth += 1

        self.best_pos = temp_best_pos
        self.best_score = temp_best_score

        goal_point = self.get_goal_point()
        if goal_point is not None:
            self.goal_pub.publish(goal_point)

    def on_params(self, params):
        """Calisirken degistirilebilen: use_frontier, trail_*."""
        for prm in params:
            if prm.name == "use_frontier":
                self.use_frontier = bool(prm.value)
                self.get_logger().info(
                    "Frontier " + ("ACIK" if self.use_frontier else "KAPALI: her zaman en uzun yol"))
            elif prm.name == "trail_enable":
                self.trail_enable = bool(prm.value)
                self.get_logger().info("Iz hafizasi " + ("ACIK" if self.trail_enable else "KAPALI"))
            elif prm.name in ("trail_radius_m", "trail_length_m", "trail_recent_m",
                              "trail_max_backtrack_m", "trail_turn_after_s"):
                setattr(self, prm.name, float(prm.value))
        return SetParametersResult(successful=True)

    def odom_callback(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, yaw)
        if msg.twist.twist.linear.x > 0.05 and (
                not self.track or math.hypot(p.x - self.track[-1][0], p.y - self.track[-1][1]) > 0.05):
            self.track.append((p.x, p.y))
        if msg.twist.twist.linear.x > 0.05:
            self.extend_trail(p.x, p.y)

    def extend_trail(self, x, y):
        """Ileri giderken her trail_step_m'de bir nokta; yalnizca YENI zeminde. Kendi izinin ustunden
        geri giderken (cikmazdan donus) eklenmez: iz "en son yeni zemine ciktigim yer"de biter,
        geri gidis mesafesi buna gore olculur. Uzun sicrama (geri donup baska kola sapma) s'e eklenir,
        debug/koridor cizimi o parcayi atlar."""
        tr = self.trail
        if tr:
            lx, ly, ls = tr[-1]
            d = math.hypot(x - lx, y - ly)
            if d < self.trail_step_m:
                return
            # Yeni zemin: her iz noktasina, ileri gidiste olacagindan (iz boyu mesafe + d) cok daha
            # yakin degil. Ust sinir 0.8 * koridor: eski izin koridorundaysa yeni zemin degil.
            # Duz ileri: mesafe ~ iz boyu + d, gecer. Kendi izinde geri: bir onceki noktaya ~0, gecmez.
            t = np.asarray(tr, float)
            dist = np.hypot(t[:, 0] - x, t[:, 1] - y)
            th = np.minimum(0.8 * self.trail_radius_m, 0.7 * (ls - t[:, 2] + d))
            if np.any(dist < th):
                return
            tr.append((x, y, ls + d))
            while tr and tr[-1][2] - tr[0][2] > self.trail_length_m:
                tr.pop(0)
        else:
            tr.append((x, y, 0.0))

    def facing_back(self, span_m=1.5, limit_deg=120.0):
        """Burun, izin son span_m'sindeki gidis yonune limit_deg'den fazla ters mi."""
        if self.pose is None or len(self.trail) < 2:
            return False
        ex, ey, es = self.trail[-1]
        for ox, oy, os_ in reversed(self.trail):
            if es - os_ >= span_m:
                yaw = self.pose[2]
                d = math.atan2(ey - oy, ex - ox) - yaw
                return abs(math.atan2(math.sin(d), math.cos(d))) > math.radians(limit_deg)
        return False

    def trail_raster(self, shape):
        """Iz koridoru, arac cercevesindeki harita hucrelerinde: koridorda izin sonundan geri mesafe
        (hucre), disinda inf. Izin son trail_recent_m'si cizilmez: sik dubada arac yavas ve ileri
        geri manevra yaparken etrafindaki her hucre "iz" olup en uzun yol hedefleri elenmesin."""
        self.trail_pixels = None
        if not self.trail_enable or self.pose is None or len(self.trail) < 2:
            return None
        c = self.costmap
        res = max(c.resolution, 1e-3)
        x, y, yaw = self.pose
        t = np.asarray(self.trail, float)
        dx, dy = t[:, 0] - x, t[:, 1] - y
        cs, sn = math.cos(yaw), math.sin(yaw)
        lx, ly = cs * dx + sn * dy, -sn * dx + cs * dy            # odom -> base_footprint
        px = np.round((lx - c.origin_x) / res - 0.5).astype(np.int32)
        py = np.round((ly - c.origin_y) / res - 0.5).astype(np.int32)
        self.trail_pixels = np.column_stack([px, py])
        back = (t[-1, 2] - t[:, 2]) / res                          # hucre
        big = 1e9
        img = np.full(shape, big, np.float32)
        r = max(1, int(round(self.trail_radius_m / res)))
        jump = 1.0 / res
        for i in range(len(t) - 1):                                # eskiden yeniye: yeni ustune yazar
            if back[i + 1] * res < self.trail_recent_m:
                break                                              # son kisim: yerel manevra serbest
            a, b = (int(px[i]), int(py[i])), (int(px[i + 1]), int(py[i + 1]))
            if math.hypot(b[0] - a[0], b[1] - a[1]) > jump:
                continue
            cv2.line(img, a, b, float(back[i + 1]), 2 * r + 1)
        img[img >= big] = np.inf
        return img

    def forward_vector(self):
        """Arac cercevesinde "ileri" birim vektor: son heading_anchor_m'de gidilen yon."""
        if self.heading_anchor_m <= 0 or self.pose is None or len(self.track) < 2:
            return (1.0, 0.0)
        x, y, yaw = self.pose
        lx, ly = self.track[-1]
        for ox, oy in reversed(self.track):
            if math.hypot(lx - ox, ly - oy) >= self.heading_anchor_m:
                dx, dy = lx - ox, ly - oy
                c, s = math.cos(yaw), math.sin(yaw)
                return (c * dx + s * dy, -s * dx + c * dy)
        return (1.0, 0.0)                    # henuz yeterince yol gidilmedi: aracin burnu

    def select_frontier_goal(self):
        """frontier.plan ile hedef + yol; aday yoksa False (bfs'e dusulur)."""
        res = max(self.costmap.resolution, 1e-3)
        self.trail_back = self.trail_raster(self.costmap.data.shape)

        def run(fwd):
            return frontier.plan(
                self.costmap.data, self.start_pixel(),
                cost_threshold=self.goal_cost_threshold, cost_coeff=self.path_find_cost_coeff,
                x_coeff=self.x_coeff, y_coeff=self.y_coeff,
                goal_cost_coeff=self.goal_selection_cost_coeff, detour_coeff=self.detour_coeff,
                min_goal_dist=self.frontier_min_goal_dist_m / res,
                min_unknown_area=self.frontier_min_unknown_m2 / res ** 2,
                fallback_max_cost=self.fallback_max_cost, use_frontier=self.use_frontier,
                frontier_radius=self.plan_radius_m / res if self.plan_radius_m > 0 else 0,
                forward=fwd, escape_radius=int(round(self.escape_radius_m / res)),
                trail_back=self.trail_back, max_backtrack=self.trail_max_backtrack_m / res)

        goal, path, info = run(self.fwd)
        now = self.get_clock().now().nanoseconds * 1e-9
        if goal is not None:
            self.no_goal_since = None
        elif self.no_goal_since is None:
            self.no_goal_since = now
        stuck = self.no_goal_since is not None and now - self.no_goal_since >= self.trail_turn_after_s
        # Arka yariya bakma (yalnizca iz aciksa):
        #  - ondeki her sey eski iz (geri gidis secildi ya da sinir asildi): arac geldigi yola donmus,
        #    arkada iz disi hedef varsa o secilir (cepte donup parkurun ilerisine yonelme);
        #  - onde trail_turn_after_s boyunca hic hedef yok (burun cikmaza bakiyor; RECOVERY'nin kisa
        #    geri cekilmesi cozmedi. Anlik tikanmada eski davranis aynen kalir).
        #  - burun son gidis yonune > 120 deg ters (cepte/virajda geri donmus).
        # Normal surus (virajlar dahil) hicbirini tetiklemez: davranis iz kapaliyla ayni.
        need_flip = (info["backtrack"] or info["backtrack_blocked"] or (goal is None and stuck)
                     or self.facing_back())
        if self.trail_back is not None and need_flip:
            fx, fy = self.fwd
            g2, p2, i2 = run((-fx, -fy))
            # Yalnizca arka yarida gercek frontier (halkada yeni zemin) varsa: en uzun yol / geri
            # gidis hedefi icin donulmez (sik dubada anlik tikanmada arac geri donmesin).
            better = g2 is not None and not i2["fallback"] and not i2["backtrack"]
            if better:
                goal, path, info = g2, p2, i2
                self.fwd = (-fx, -fy)
                self.get_logger().info("Onde yeni zemin yok: arka yaridan planlaniyor (iz haric)",
                                       throttle_duration_sec=2.0)
            elif goal is None and i2["backtrack_blocked"]:
                info = i2
        self.frontier_cells = info["frontier"]
        if info["backtrack_blocked"]:
            self.get_logger().error(
                f"CIKMAZ: ileri yol yok, geri gidis {self.trail_max_backtrack_m:.1f} m sinirini asiyor. "
                "Arac duruyor (trail_max_backtrack_m / trail_enable).", throttle_duration_sec=2.0)
            self.best_pos, self.best_score, self.frontier_path = None, None, None
            return True                           # bos yol: create_path best_pos None -> dur
        if info["backtrack"]:
            self.get_logger().info("Hedef eski izin uzerinde (onde yeni zemin yok, geri gidis)",
                                   throttle_duration_sec=2.0)
        if goal is None:
            self.get_logger().warn("Ust yarida gidilebilir hucre yok, eski BFS hedef secimine dusuldu",
                                   throttle_duration_sec=2.0)
            return False
        if info["fallback"] and self.use_frontier:
            self.get_logger().info("Ust yarida sinira ulasan yol yok: en uzun yolun ucuna gidiliyor",
                                   throttle_duration_sec=2.0)
        self.best_pos, self.best_score, self.frontier_path = goal, None, path
        goal_point = self.get_goal_point()
        if goal_point is not None:
            self.goal_pub.publish(goal_point)
        return True

    def find_path(self, start, goal):
        grid_w = self.costmap.pixel_width
        grid_h = self.costmap.pixel_height
        data = self.costmap.data

        def h(pt):
            return sqrt((pt[0] - goal[0]) ** 2 + (pt[1] - goal[1]) ** 2)

        def move_cost(pt):
            val = data[pt[1], pt[0]]
            cost_val = val if val != -1 else 0
            return 1.0 + (cost_val * self.path_find_cost_coeff / 100.0)

        g_score = {start: 0}
        open_heap = [(h(start), start)]
        open_set = {start}
        closed_set = set()
        came_from = {}
        iters = 0
        closest_node = start
        min_h_val = h(start)

        while open_heap and iters < self.max_iters:
            iters += 1
            f, current = heappop(open_heap)
            if current in open_set:
                open_set.remove(current)
            if current in closed_set:
                continue

            current_h = h(current)
            if current_h < min_h_val:
                min_h_val = current_h
                closest_node = current

            if abs(current[0] - goal[0]) < 2 and abs(current[1] - goal[1]) < 2:
                return self.recreate_path(came_from, current)

            closed_set.add(current)

            for dx, dy in MOTION_PRIMITIVES:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < grid_w and 0 <= neighbor[1] < grid_h:
                    if neighbor in closed_set:
                        continue
                    val = data[neighbor[1], neighbor[0]]
                    if val == -1 or val > self.goal_cost_threshold:
                        continue
                    tentative_g = g_score[current] + move_cost(neighbor)
                    if tentative_g < g_score.get(neighbor, float("inf")):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + self.path_find_distance_coeff * h(neighbor)
                        if neighbor not in open_set:
                            open_set.add(neighbor)
                            heappush(open_heap, (f_score, neighbor))

        if closest_node != start:
            return self.recreate_path(came_from, closest_node)
        return None

    def recreate_path(self, path, current):
        total_path = [current]
        while current in path:
            current = path[current]
            total_path.append(current)
        return total_path[::-1]

    def get_goal_point(self):
        if self.best_pos is None:
            return None
        goal_point = PoseStamped()
        goal_point.header.frame_id = self.path_frame
        goal_point.header.stamp = self.get_clock().now().to_msg()
        goal_point.pose.position.x, goal_point.pose.position.y = (
            self.costmap.pixelToPose(self.best_pos[0], self.best_pos[1]))
        goal_point.pose.orientation.w = 1.0
        return goal_point

    def on_reset(self):
        self.costmap = None
        self.best_pos = None
        self.best_score = None
        self.state = "PLANNING"


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C / launch kapanisi
    except Exception:  # noqa: BLE001
        if rclpy.ok():
            raise
        # Kapanis yarisi: sinyal context'i kapatirken bir callback yayin yapiyordu; yut
    finally:
        # Kapanirken bos yol yayinla (takipci dursun)
        if rclpy.ok():
            try:
                node.publish_empty_path()
            except Exception:  # noqa: BLE001
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
