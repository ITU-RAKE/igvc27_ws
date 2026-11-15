#!/usr/bin/env python3
"""
IGVC 4 kamera surround-view (kuş bakışı) node'u.

Homografiler elle girilmez; her kameranın intrinsic'i (fx, fy, cx, cy, distorsiyon) ve
montaj pozundan (x, y, z, yaw, pitch) açılışta hesaplanır. Bunlardan tek bir LUT üretilir
ve her karede tek bir cv2.remap yapılır.

Koordinatlar (base_footprint, zeminde):              X ileri, Y sol, Z yukarı [m]
Kamera optik çerçevesi (OpenCV):                    x sağ,  y aşağı, z ileri
BEV görüntüsü:                                     yukarı = +X, sağ = -Y, araç merkezde

Dikiş (kameraların çakıştığı yer): araç merkezinden geçen, simetrik çapraz çizgiler.
Örneğin ön-sol çakışmasında çizginin solu sol kameradan, sağı ön kameradan alınır.
Çizginin açısı seam_angle_deg (X ekseninden, varsayılan 45°). Sahibi olan kamera bir pikseli
göremiyorsa (çok yakın ya da çok uzak köşeler) komşu kamera görüyorsa ondan doldurulur.

Arac geometrisi (kamera pozlari, intrinsic'ler, govde, tekerlekler) surround_view.launch.py
tarafindan igvc_description/urdf/igvc_macros.xacro'dan hesaplanip parametre olarak verilir.
Govde ve tekerlekler, kameranin zemini gormesini engelleyen disbukey cisimler olarak modellenir;
arkalarinda kalan zemin kus bakisina yansitilmaz.

Yayınlar:
  igvc_vision/merged_image/compressed       4 ham görüntü, ortada kutu etrafında
  igvc_vision/merged_image_homo/compressed  dairesel kuş bakışı görüntüsü
  igvc_vision/merged_image_homo             (publish_raw_homo:=true ise) sıkıştırılmamış bgr8
  igvc_vision/merged_image                  (publish_raw_merged:=true ise) sıkıştırılmamış bgr8
  Sıkıştırılmamış olanlar RViz "Image" panelinde görüntülenebilir.
"""

import array
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

CAMS = ("front", "back", "left", "right")  # LUT / stack sırası

# Varsayilanlar: igvc_macros.xacro'daki degerler (launch hep xacro'dan gunceller).
DEFAULT_TOPICS = {
    "front": "/front_camera/image_raw/compressed",
    "back": "/rear_camera/image_raw/compressed",
    "left": "/left_camera/image_raw/compressed",
    "right": "/right_camera/image_raw/compressed",
}
# [x, y, z, yaw_deg, pitch_deg (aşağı +), roll_deg (opsiyonel)]
DEFAULT_POSES = {
    "front": [0.325, 0.0, 1.0, 0.0, 43.0],
    "back": [-0.325, 0.0, 1.0, 180.0, 43.0],
    "left": [0.0, 0.20, 1.0, 90.0, 43.0],
    "right": [0.0, -0.20, 1.0, -90.0, 43.0],
}
DEFAULT_IMAGE_SIZE = (1280, 854)
DEFAULT_INTRINSICS = [556.343512, 556.343512, 639.5, 426.5]  # fx, fy, cx, cy (1280x854, 98 deg)
DEFAULT_DIST = [0.0, 0.0, 0.0, 0.0, 0.0]  # k1, k2, p1, p2, k3
# Arac: kesik piramit govde + iki ceker tekerlek (base_footprint, metre)
DEFAULT_VEHICLE = {
    "chassis_bottom_length": 0.95, "chassis_bottom_width": 0.65,
    "chassis_top_length": 0.85, "chassis_top_width": 0.60,
    "chassis_height": 0.30, "ground_clearance": 0.10, "chassis_center_x": 0.0,
    "wheel_radius": 0.10, "wheel_width": 0.06, "wheel_separation": 0.73, "wheel_x": 0.0,
}


# ======================================================================================
# Geometri (ROS'tan bağımsız, test edilebilir)
# ======================================================================================
def camera_rotation(yaw_deg, pitch_deg, roll_deg=0.0):
    """R: base <- kamera optik çerçevesi. Sütunlar: sağ, aşağı, ileri.
    URDF ile ayni: kamera linki rpy = (roll, pitch, yaw), R_link = Rz(yaw) Ry(pitch) Rx(roll),
    pitch + = asagi. Optik cerceve: sag = -y_link, asagi = -z_link, ileri = x_link."""
    psi, th, ph = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    cz, sz, cy, sy, cx, sx = (math.cos(psi), math.sin(psi), math.cos(th), math.sin(th),
                              math.cos(ph), math.sin(ph))
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    L = Rz @ Ry @ Rx
    return np.column_stack([-L[:, 1], -L[:, 2], L[:, 0]])


def _pose(pose):
    """[x, y, z, yaw, pitch(, roll)] -> (x, y, z, R)."""
    x, y, z, yaw, pitch = pose[:5]
    roll = pose[5] if len(pose) > 5 else 0.0
    return x, y, z, camera_rotation(yaw, pitch, roll)


def intrinsic_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def ground_to_image_h(K, pose):
    """H_g2i: zemin (X, Y, 1) [m] -> görüntü pikseli. 3. bileşen = kamera derinliği."""
    x, y, z, R = _pose(pose)
    M = np.array([[1.0, 0.0, -x], [0.0, 1.0, -y], [0.0, 0.0, -z]])
    return K @ R.T @ M


def ground_to_bev_matrix(bev_px, res_m):
    """A: zemin (X, Y, 1) [m] -> BEV pikseli (u = sütun, v = satır)."""
    c = bev_px / 2.0
    s = 1.0 / res_m
    return np.array([[0.0, -s, c], [-s, 0.0, c], [0.0, 0.0, 1.0]])


# ---- Kamerayi kapatan cisimler: disbukey cisim = yari uzaylar {p : n . p <= h} ----
def box_convex(x0, x1, y0, y1, z0, z1):
    """Eksen hizali kutu -> (normaller (k,3), h (k,))."""
    n = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    return n, np.array([x1, -x0, y1, -y0, z1, -z0], float)


def frustum_convex(cx, bl, bw, tl, tw, z0, height):
    """Kesik piramit: taban (bl x bw) z0'da, tavan (tl x tw) z0+height'ta, x=cx merkezli.
    bl/bw/tl/tw tam olculer."""
    kx = (tl - bl) / 2 / height   # yukari cikildikca yari uzunluk degisimi
    ky = (tw - bw) / 2 / height
    n = np.array([[1, 0, -kx], [-1, 0, -kx], [0, 1, -ky], [0, -1, -ky],
                  [0, 0, 1], [0, 0, -1]], float)
    h = np.array([cx + bl / 2 - kx * z0, -cx + bl / 2 - kx * z0,
                  bw / 2 - ky * z0, bw / 2 - ky * z0, z0 + height, -z0], float)
    return n, h


def box_body(hx, hy, hz):
    """Eski tip govde: merkezli kutu (yari uzunluk, yari genislik, yukseklik)."""
    return {"convex": [box_convex(-hx, hx, -hy, hy, 0.0, hz)], "footprint": (-hx, hx, -hy, hy)}


def vehicle_body(v):
    """Kesik piramit govde + iki tekerlek (kutu yaklasimi). Ayak izi tekerlekler dahil."""
    cx = v["chassis_center_x"]
    convex = [frustum_convex(cx, v["chassis_bottom_length"], v["chassis_bottom_width"],
                             v["chassis_top_length"], v["chassis_top_width"],
                             v["ground_clearance"], v["chassis_height"])]
    r, ww, sep, wx = v["wheel_radius"], v["wheel_width"], v["wheel_separation"], v["wheel_x"]
    for s in (1, -1):
        y0, y1 = sorted((s * (sep / 2 - ww / 2), s * (sep / 2 + ww / 2)))
        convex.append(box_convex(wx - r, wx + r, y0, y1, 0.0, 2 * r))
    hl = max(v["chassis_bottom_length"], v["chassis_top_length"]) / 2
    hw = max(v["chassis_bottom_width"], v["chassis_top_width"], sep + ww) / 2
    fp = (min(cx - hl, wx - r), max(cx + hl, wx + r), -hw, hw)
    return {"convex": convex, "footprint": fp}


def _ray_hits_convex(origin, dirs, normals, h):
    """origin + t*dirs (t ∈ [0,1]) disbukey cisme giriyor mu? dirs: (3, N). Cyrus-Beck."""
    tmin = np.zeros(dirs.shape[1])
    tmax = np.ones(dirs.shape[1])
    for nk, hk in zip(normals, h):
        nd = nk @ dirs
        rhs = hk - nk @ origin          # nd * t <= rhs
        with np.errstate(divide="ignore", invalid="ignore"):
            t = rhs / nd
        tmax = np.where(nd > 0, np.minimum(tmax, t), tmax)
        tmin = np.where(nd < 0, np.maximum(tmin, t), tmin)
        if rhs < 0:                     # paralel ve disarida -> hic kesmez
            tmax = np.where(nd == 0, -1.0, tmax)
    return tmax >= tmin


def project_ground(X, Y, K, dist, pose, img_w, img_h, body=None, min_depth=0.05):
    """
    Zemin noktalarını kameraya izdüşürür.
    Dönüş: u, v (distorsiyonlu piksel) ve valid (kameranın gerçekten gördüğü noktalar).
    body: {'convex': [(normaller, h), ...]} kamerayi kapatan cisimler (veya None)
    """
    x, y, z, R = _pose(pose)
    P = np.stack([X.ravel() - x, Y.ravel() - y, np.full(X.size, -z)])
    Pc = R.T @ P
    zc = Pc[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xn, yn = Pc[0] / zc, Pc[1] / zc
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Önce ideal pinhole ile kaba FOV kontrolü (distorsiyon polinomu FOV dışında katlanabilir)
    mx, my = 0.15 * img_w, 0.15 * img_h
    ui, vi = fx * xn + cx, fy * yn + cy
    valid = (zc > min_depth) & (ui > -mx) & (ui < img_w + mx) & (vi > -my) & (vi < img_h + my)

    k1, k2, p1, p2, k3 = (list(dist) + [0.0] * 5)[:5]
    if any(abs(c) > 0 for c in (k1, k2, p1, p2, k3)):
        r2 = xn * xn + yn * yn
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    else:
        xd, yd = xn, yn
    u, v = fx * xd + cx, fy * yd + cy
    valid &= (u >= 0) & (u <= img_w - 1) & (v >= 0) & (v <= img_h - 2)

    if body is not None:  # gövde/tekerlek bu zemin noktasını kameradan saklıyor mu?
        dirs = np.stack([X.ravel() - x, Y.ravel() - y, np.full(X.size, -z)])
        origin = np.array([x, y, z])
        for normals, h in body["convex"]:
            valid &= ~_ray_hits_convex(origin, dirs, normals, h)
    shp = X.shape
    return u.reshape(shp), v.reshape(shp), valid.reshape(shp)


class SurroundLUT:
    """4 kamera -> dairesel BEV, tek cv2.remap."""

    def __init__(self, cams, img_w, img_h, bev_px, res_m, max_range_m, seam_deg,
                 body, fill_from_neighbor=True):
        """
        cams: CAMS sırasıyla [{'K': 3x3, 'dist': [5], 'pose': [5 veya 6]}, ...]
        body: vehicle_body()/box_body() sonucu ya da (yarı uzunluk, yarı genişlik, yükseklik)
        """
        if not isinstance(body, dict):
            body = box_body(*body)
        self.img_w, self.img_h, self.bev_px, self.res = img_w, img_h, bev_px, res_m
        S = bev_px
        c = S / 2.0
        vv, uu = np.mgrid[0:S, 0:S].astype(np.float64)
        X = (c - vv) * res_m
        Y = (c - uu) * res_m
        az = np.degrees(np.arctan2(Y, X))
        fx0, fx1, fy0, fy1 = body["footprint"]

        self.in_range = np.hypot(X, Y) <= max_range_m
        self.footprint = (X >= fx0) & (X <= fx1) & (Y >= fy0) & (Y <= fy1)
        usable = self.in_range & ~self.footprint

        s = seam_deg
        sectors = [
            np.abs(az) <= s,  # front
            np.abs(az) > 180.0 - s,  # back
            (az > s) & (az <= 180.0 - s),  # left
            (az < -s) & (az >= -(180.0 - s)),  # right
        ]
        neighbors = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}

        proj = []
        for k, cam in enumerate(cams):
            u, v, ok = project_ground(X, Y, cam["K"], cam["dist"], cam["pose"],
                                      img_w, img_h, body)
            proj.append((u, v + k * img_h, ok & usable))

        map_x = np.full((S, S), -100.0, np.float32)
        map_y = np.full((S, S), -100.0, np.float32)
        owner = np.full((S, S), -1, np.int8)
        for k in range(4):  # 1) sektör sahibi
            u, v, ok = proj[k]
            sel = sectors[k] & ok
            map_x[sel], map_y[sel], owner[sel] = u[sel], v[sel], k
        if fill_from_neighbor:  # 2) sahibi görmüyorsa komşu kamera
            for k in range(4):
                hole = sectors[k] & usable & (owner < 0)
                for n in neighbors[k]:
                    u, v, ok = proj[n]
                    sel = hole & ok & (owner < 0)
                    map_x[sel], map_y[sel], owner[sel] = u[sel], v[sel], n

        self.owner = owner  # -1 = görülmüyor
        self.sectors = sectors
        self.map1, self.map2 = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
        self._stack = np.zeros((4 * img_h, img_w, 3), np.uint8)

    def coverage_stats(self):
        """Kesintisiz 360° menzil [m] ve kapsama oranı (menzil dairesi içinde)."""
        S, c = self.bev_px, self.bev_px / 2.0
        seen = self.owner >= 0
        best = np.inf
        for a in np.radians(np.arange(0.0, 360.0, 0.25)):
            rr = np.arange(0.0, S / 2.0) * self.res
            u = np.clip(np.round(c - rr * np.sin(a) / self.res).astype(int), 0, S - 1)
            v = np.clip(np.round(c - rr * np.cos(a) / self.res).astype(int), 0, S - 1)
            ok = seen[v, u] | self.footprint[v, u]
            inside = self.in_range[v, u]
            # dışarıdan içeri: gövde kenarından sonraki ilk sürekli görülen bant
            first_seen = np.argmax(ok & ~self.footprint[v, u]) if ok.any() else 0
            run = ok[first_seen:] & inside[first_seen:]
            end = first_seen + (np.argmin(run) if not run.all() else len(run))
            best = min(best, rr[min(end, len(rr) - 1)])
        usable = self.in_range & ~self.footprint
        return best, float((seen & usable).sum() / usable.sum())

    def warp(self, imgs):
        h = self.img_h
        for k, im in enumerate(imgs):
            self._stack[k * h:(k + 1) * h] = im
        return cv2.remap(self._stack, self.map1, self.map2, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def build_overlay(lut, max_range_m, draw_vehicle, draw_seams, draw_circle, seam_deg):
    """Dairesel çıktı üstüne sabit çizimler: araç, dikiş, menzil dairesi. (img, mask)"""
    S, res, c = lut.bev_px, lut.res, lut.bev_px / 2.0
    img = np.zeros((S, S, 3), np.uint8)
    mask = np.zeros((S, S), np.uint8)
    if draw_seams:
        L = max_range_m / res
        for a in (seam_deg, 180 - seam_deg, -seam_deg, -(180 - seam_deg)):
            t = math.radians(a)
            p2 = (int(round(c - L * math.sin(t))), int(round(c - L * math.cos(t))))
            cv2.line(img, (int(c), int(c)), p2, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.line(mask, (int(c), int(c)), p2, 255, 1)
    if draw_circle:
        r = int(round(max_range_m / res))
        cv2.circle(img, (int(c), int(c)), r, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.circle(mask, (int(c), int(c)), r, 255, 1)
    if draw_vehicle:
        fp = lut.footprint
        img[fp] = (45, 45, 45)
        mask[fp] = 255
        ys, xs = np.nonzero(fp)
        if len(xs):
            cv2.rectangle(img, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 200, 0), 1)
            cv2.rectangle(mask, (xs.min(), ys.min()), (xs.max(), ys.max()), 255, 1)
            tip = (int(c), int(ys.min() + 3))
            cv2.arrowedLine(img, (int(c), int(c)), tip, (0, 200, 0), 2, tipLength=0.35)
            cv2.arrowedLine(mask, (int(c), int(c)), tip, 255, 2, tipLength=0.35)
    return img, mask


# ======================================================================================
# ROS2 node
# ======================================================================================
class SurroundViewNode(Node):
    def __init__(self):
        super().__init__("surround_view_node")
        p = self.declare_parameter

        self.img_w = p("image_width", DEFAULT_IMAGE_SIZE[0]).value
        self.img_h = p("image_height", DEFAULT_IMAGE_SIZE[1]).value
        bev_m = p("bev_size_m", 10.0).value
        res = p("bev_resolution_m", 0.02).value
        max_range = p("max_range_m", 4.9).value
        seam = p("seam_angle_deg", 45.0).value
        # Arac: launch xacro'dan 'vehicle.*' verir. use_vehicle_model=false ise eski tip kutu.
        use_vehicle = p("use_vehicle_model", True).value
        vehicle = {k: float(p(f"vehicle.{k}", d).value) for k, d in DEFAULT_VEHICLE.items()}
        body_box = p("body_box_m", [0.465, 0.325, 0.34]).value  # yarı X, yarı Y, yükseklik
        body = vehicle_body(vehicle) if use_vehicle else box_body(*body_box)
        fill_nb = p("fill_from_neighbor", True).value
        rate = p("rate_hz", 20.0).value
        self.stale_s = p("stale_timeout_s", 0.5).value
        self.frame_id = p("frame_id", "base_footprint").value
        self.jpeg = [int(cv2.IMWRITE_JPEG_QUALITY), p("jpeg_quality", 85).value]
        raw_topic = p("merged_topic", "igvc_vision/merged_image").value
        homo_topic = p("homo_topic", "igvc_vision/merged_image_homo").value
        self.publish_raw_homo = p("publish_raw_homo", False).value
        self.publish_raw_merged = p("publish_raw_merged", False).value
        # Ham mozaik buyuk (1280x854 icin 2988x2988, ~27 MB): RViz/rqt icin kucultulur
        self.merged_raw_scale = float(p("merged_raw_scale", 0.4).value)
        self.mosaic_small = None
        draw_vehicle = p("draw_vehicle", True).value
        draw_seams = p("draw_seams", False).value
        draw_circle = p("draw_range_circle", True).value

        cams, topics = [], {}
        for name in CAMS:
            topics[name] = p(f"{name}.topic", DEFAULT_TOPICS[name]).value
            pose = list(p(f"{name}.pose", DEFAULT_POSES[name]).value)
            fx, fy, cx, cy = p(f"{name}.intrinsics", DEFAULT_INTRINSICS).value
            dist = list(p(f"{name}.dist_coeffs", DEFAULT_DIST).value)
            K = intrinsic_matrix(fx, fy, cx, cy)
            cams.append({"K": K, "dist": dist, "pose": pose})

        bev_px = int(round(bev_m / res))
        t0 = time.perf_counter()
        self.lut = SurroundLUT(cams, self.img_w, self.img_h, bev_px, res, max_range, seam,
                               body, fill_nb)
        self.ov_img, self.ov_mask = build_overlay(
            self.lut, max_range, draw_vehicle, draw_seams, draw_circle, seam)
        rng, cov = self.lut.coverage_stats()
        self.get_logger().info(
            f"LUT hazır: {bev_px}x{bev_px} px, {res * 100:.0f} cm/px, "
            f"{(time.perf_counter() - t0) * 1e3:.0f} ms | kesintisiz menzil ≈ {rng:.2f} m, "
            f"{max_range} m dairesi içinde kapsama %{cov * 100:.1f}")

        A = ground_to_bev_matrix(bev_px, res)
        np.set_printoptions(precision=6, suppress=True)
        fp = body["footprint"]
        self.get_logger().info(
            f"Görüntü {self.img_w}x{self.img_h} | ayak izi x[{fp[0]:.3f}, {fp[1]:.3f}] "
            f"y[{fp[2]:.3f}, {fp[3]:.3f}] m | "
            + " | ".join(f"{n}: {np.round(c['pose'], 3).tolist()}" for n, c in zip(CAMS, cams)))
        for name, cam in zip(CAMS, cams):
            H_g2i = ground_to_image_h(cam["K"], cam["pose"])
            H_i2bev = A @ np.linalg.inv(H_g2i)
            self.get_logger().info(
                f"{name} H_i2bev (distorsiyonsuz):\n{H_i2bev / H_i2bev[2, 2]}")

        # Durum
        self._latest = {n: None for n in CAMS}
        self._cache = {n: (None, None) for n in CAMS}
        self.blank = np.zeros((self.img_h, self.img_w, 3), np.uint8)
        self._warned_size = set()
        side = self.img_w + 2 * self.img_h
        self.mosaic = np.zeros((side, side, 3), np.uint8)
        a, b = self.img_h, self.img_h + self.img_w
        cv2.rectangle(self.mosaic, (a, a), (b, b), (60, 60, 60), 3)
        self._times, self._stats_t = [], time.monotonic()

        for name in CAMS:
            self.create_subscription(CompressedImage, topics[name],
                                     lambda m, n=name: self._store(n, m),
                                     qos_profile_sensor_data)
        self.pub_raw = self.create_publisher(CompressedImage, raw_topic + "/compressed", 10)
        self.pub_homo = self.create_publisher(CompressedImage, homo_topic + "/compressed", 10)
        self.pub_homo_raw = (self.create_publisher(Image, homo_topic, 10)
                             if self.publish_raw_homo else None)
        self.pub_raw_img = (self.create_publisher(Image, raw_topic, 10)
                            if self.publish_raw_merged else None)
        self.create_timer(1.0 / rate, self.loop)
        self.get_logger().info("Surround-view node başlatıldı.")

    # ------------------------------------------------------------------------------
    def _store(self, name, msg):
        # Alis zamani dugum saatiyle (use_sim_time: simulasyon saati); bkz. _images
        self._latest[name] = (msg, self.get_clock().now().nanoseconds * 1e-9)  # decode döngüde

    def _images(self):
        # Eskilik dugum saatiyle: duvar saatinde yuk altinda Gazebo yavaslayinca kareler stale_s'den
        # gec gelir, kamera siyaha doner ve kirmizi engeller kus bakisindan anlik kaybolurdu.
        now, out = self.get_clock().now().nanoseconds * 1e-9, []
        for name in CAMS:
            entry = self._latest[name]
            if entry is None or now - entry[1] > self.stale_s:
                out.append(self.blank)
                continue
            msg = entry[0]
            cmsg, cimg = self._cache[name]
            if cmsg is msg:
                out.append(cimg)
                continue
            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                out.append(self.blank)
                continue
            if img.shape[:2] != (self.img_h, self.img_w):
                if name not in self._warned_size:
                    self._warned_size.add(name)
                    self.get_logger().warn(
                        f"{name}: görüntü {img.shape[1]}x{img.shape[0]}, beklenen "
                        f"{self.img_w}x{self.img_h}. Yeniden boyutlandırılıyor; intrinsic "
                        f"değerler bu çözünürlüğe göre olmalı!")
                img = cv2.resize(img, (self.img_w, self.img_h))
            self._cache[name] = (msg, img)
            out.append(img)
        return out

    def _publish_jpeg(self, pub, img, stamp):
        msg = CompressedImage()
        msg.header.stamp, msg.header.frame_id = stamp, self.frame_id
        msg.format = "jpeg"
        msg.data = cv2.imencode(".jpg", img, self.jpeg)[1].tobytes()
        pub.publish(msg)

    def _publish_bgr(self, pub, img, stamp):
        m = Image()
        m.header.stamp, m.header.frame_id = stamp, self.frame_id
        m.height, m.width = img.shape[:2]
        m.encoding, m.step = "bgr8", img.shape[1] * 3
        m.data = array.array('B', img.tobytes())  # bytes atamak rclpy'de ~100x yavas
        pub.publish(m)

    @staticmethod
    def _mosaic(imgs, c):
        """4 goruntuyu ortadaki kutunun etrafina yerlestir (on ust, arka alt, sol sol, sag sag)."""
        f, b, l, r = imgs
        h, w = f.shape[:2]
        c[0:h, h:h + w] = f
        c[h + w:2 * h + w, h:h + w] = cv2.rotate(b, cv2.ROTATE_180)
        c[h:h + w, 0:h] = cv2.rotate(l, cv2.ROTATE_90_COUNTERCLOCKWISE)
        c[h:h + w, h + w:2 * h + w] = cv2.rotate(r, cv2.ROTATE_90_CLOCKWISE)
        return c

    def loop(self):
        if all(v is None for v in self._latest.values()):
            return
        t0 = time.perf_counter()
        stamp = self.get_clock().now().to_msg()
        f, b, l, r = imgs = self._images()

        want_homo = (self.pub_homo.get_subscription_count() > 0 or
                     (self.pub_homo_raw is not None and
                      self.pub_homo_raw.get_subscription_count() > 0))
        if want_homo:
            bev = self.lut.warp(imgs)
            cv2.copyTo(self.ov_img, self.ov_mask, bev)
            if self.pub_homo.get_subscription_count() > 0:
                self._publish_jpeg(self.pub_homo, bev, stamp)
            if self.pub_homo_raw is not None and self.pub_homo_raw.get_subscription_count() > 0:
                self._publish_bgr(self.pub_homo_raw, bev, stamp)

        want_raw_img = (self.pub_raw_img is not None and
                        self.pub_raw_img.get_subscription_count() > 0)
        if self.pub_raw.get_subscription_count() > 0:
            self._publish_jpeg(self.pub_raw, self._mosaic(imgs, self.mosaic), stamp)
        if want_raw_img:
            # Once her kamerayi kucult, sonra yerlestir (tam boy mozaigi kucultmekten ~5x ucuz)
            s = self.merged_raw_scale
            small = imgs if s >= 1.0 else [
                cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for im in imgs]
            hh, ww = small[0].shape[:2]
            side = ww + 2 * hh
            if self.mosaic_small is None or self.mosaic_small.shape[0] != side:
                self.mosaic_small = np.zeros((side, side, 3), np.uint8)
                cv2.rectangle(self.mosaic_small, (hh, hh), (hh + ww, hh + ww), (60, 60, 60), 2)
            self._publish_bgr(self.pub_raw_img, self._mosaic(small, self.mosaic_small), stamp)

        self._times.append(time.perf_counter() - t0)
        now = time.monotonic()
        if now - self._stats_t >= 5.0:
            self.get_logger().info(
                f"{len(self._times) / (now - self._stats_t):.1f} Hz, "
                f"ort {np.mean(self._times) * 1e3:.1f} ms/döngü")
            self._times, self._stats_t = [], now


def main():
    rclpy.init()
    node = SurroundViewNode()
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


if __name__ == "__main__":
    main()
