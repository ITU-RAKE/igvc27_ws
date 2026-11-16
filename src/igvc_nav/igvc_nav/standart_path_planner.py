#!/usr/bin/env python3
"""
Yerel yol planlayici (gecen seneki aracin path_planner'i, igvc27_ws'e uyarlandi).

Algoritma (degismedi):
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
  - goal_cost_threshold varsayilani 99: /fused_map'te 100 = oldurucu (engel + yarim kosegen),
    99..1 sisirme maliyeti A* hucre maliyetine girer. (Eski 40 esigi eski inflation icindi;
    bizim rampada engelden 1 m'ye kadar her yeri kapatirdi.)
  - Eski kamera haritasindaki "50 = bilinmiyor" ozel durumu kaldirildi (bizde 50 siradan maliyet).
  - Config -> ROS parametreleri; sistem durumu -> planner_mode parametresi.
  - GPS / waypoint / odom kaldirildi (yol cizimini etkilemiyordu; yol base_footprint'te).

Yayinlar:
  /igvc_nav/planned_path  (nav_msgs/Path, base_footprint)
  /igvc_nav/goal          (geometry_msgs/PoseStamped)
  /igvc_nav/debug_image   (sensor_msgs/Image, bgr8; + /compressed) yukari = aracin onu.
      beyaz bos, gri tonlari maliyet (koyu = pahali), siyah gecilemez, koyu gri bilinmiyor,
      yesil yol, kirmizi hedef, mavi baslangic.
"""
import array
from heapq import heappop, heappush
from math import sqrt

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

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
        self.goal_search_depth = int(p("goal_search_depth", 85).value)      # hucre (5 cm: 2.5 m)
        self.goal_selection_cost_coeff = float(p("goal_selection_cost_coeff", 5.0).value)
        self.path_find_cost_coeff = float(p("path_find_cost_coeff", 8.5).value)
        self.path_find_distance_coeff = float(p("path_find_distance_coeff", 2.0).value)
        self.max_iters = int(p("max_iters", 1000).value)
        self.min_safe_path_length = int(p("min_safe_path_length", 5).value)
        # Mod: normal | align_for_slider | steep_incline (eskiden sistem durumu)
        self.planner_mode = p("planner_mode", "normal").value
        self.wall_cost_threshold = int(p("wall_cost_threshold", 100).value)  # hizalama modu
        self.lane_width_m = float(p("lane_width_m", 3.0).value)              # hizalama modu
        # Baslangic: base_footprint orijini + bu kadar ileri [m]
        self.start_offset_m = float(p("start_offset_m", 0.0).value)
        # Arac ayak izi [x0, x1, y0, y1] base_footprint'te (launch xacro'dan verir)
        self.footprint = list(p("footprint_m", [-0.475, 0.475, -0.395, 0.395]).value)
        self.clear_footprint = bool(p("clear_footprint", True).value)
        self.path_frame = p("path_frame", "base_footprint").value
        self.debug_scale = int(p("debug_image_scale", 3).value)
        rate = float(p("plan_rate", 10.0).value)

        self.costmap = None
        self.map_frame = self.path_frame
        self.best_pos = None
        self.best_score = None
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
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_timer(1.0 / rate, self.create_path)
        self.get_logger().info(
            f"Path planner: {self.map_topic} -> /igvc_nav/planned_path, /igvc_nav/debug_image | "
            f"mod={self.planner_mode}, gecilemez > {self.goal_cost_threshold}")

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
        debug_img = np.zeros((g.shape[0], g.shape[1], 3), dtype=np.uint8)
        debug_img[g == -1] = [90, 90, 90]
        debug_img[g == 0] = [255, 255, 255]
        debug_img[g > self.goal_cost_threshold] = [0, 0, 0]
        # Gecilebilir maliyet: acik (dusuk) -> koyu (yuksek) gri
        cost_mask = (g > 0) & (g <= self.goal_cost_threshold)
        if np.any(cost_mask):
            shade = (235 - g[cost_mask].astype(np.float32) * 175.0
                     / max(self.goal_cost_threshold, 1)).astype(np.uint8)
            debug_img[cost_mask] = np.stack([shade] * 3, axis=-1)

        start = self.start_pixel()
        cv2.circle(debug_img, start, 2, (255, 0, 0), -1)
        if goal is not None:
            cv2.circle(debug_img, (goal[0], goal[1]), 2, (0, 0, 255), -1)
        if path is not None and len(path) > 1:
            for i in range(len(path) - 1):
                cv2.line(debug_img, path[i], path[i + 1], (0, 200, 0), 1)
        return debug_img

    def publish_path_debug(self, debug_img):
        if debug_img is None:
            return
        want_raw = self.debug_img_pub.get_subscription_count() > 0
        want_jpeg = self.debug_cimg_pub.get_subscription_count() > 0
        if not (want_raw or want_jpeg):
            return
        # grid[y, x] -> yukari = +x (ileri), sol = +y
        debug_img = cv2.flip(cv2.rotate(debug_img, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
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
        self.path.poses = self.poses_to_path(path_poses)
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

    def poses_to_path(self, poses):
        out = []
        stamp = self.get_clock().now().to_msg()
        for x, y in poses:
            ps = PoseStamped()
            ps.header.frame_id = self.path_frame
            ps.header.stamp = stamp
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            out.append(ps)
        return out

    # ------------------------------------------------------------------ planlama
    def create_path(self):
        if self.costmap is None:
            self.publish_empty_path()
            return

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
        path = self.find_path(start, goal)
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
        self.costmap.update_data(data)

        # Çevre değerlendirmesi (hedef seçimi) her zaman yapılmalı
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
        if self.best_pos is None or self.best_score is None:
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
