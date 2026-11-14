#!/usr/bin/env python3
"""
Hızlı 4 kameralı kuş bakışı (BEV) node'u — Jetson Orin NX.

Eski sürüme göre farklar:
  * Callback'ler sadece mesajı saklar; JPEG decode yalnızca döngüde, en son kare için yapılır
    (eskiden 4 x 30 = 120 decode/sn tek thread'de yapılıyordu).
  * 4 ayrı warpPerspective + bitwise_or yerine tek bir önceden hesaplanmış LUT ve tek cv2.remap.
    LUT her BEV pikseli için hangi kameradan, hangi pikselden okunacağını tutar.
  * BEV, kalibre edilen menzille (max_range_m) sınırlı; ufka yakın satırların aşırı gerilmesi
    (renkli şeritler) ve kameraların üst üste binip OR'lanması artık yok.
  * YOLO ayrı thread'de (MultiThreadedExecutor), BEV akışını yavaşlatmıyor.
    TensorRT engine batch=4 ile export edildiyse 4 kamera tek çağrıda işlenir.
  * YOLO maskeleri de aynı LUT ile BEV'e çevrilip igvc_vision/bev_mask (mono8) olarak yayınlanır
    -> doğrudan costmap / harita katmanına verilebilir. Değer: 0 = boş, sınıf_id + 1.
"""

import array
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray

try:
    import torch
except ImportError:  # sadece LUT testi için
    torch = None

CAMS = ["front", "back", "left", "right"]  # LUT / stack sırası (cam_id 0..3)


# --------------------------------------------------------------------------------------
# BEV LUT
# --------------------------------------------------------------------------------------
class BevWarper:
    """
    4 kamerayı tek bir cv2.remap ile kuş bakışına çevirir.

    homographies: [front, back, left, right] için 3x3 matris; kamera pikseli -> BEV pikseli.
    BEV ekseni: yukarı = ileri, merkez = mast.
    """

    def __init__(self, cam_w, cam_h, homographies, bev_size, bev_meters, max_range_m):
        self.cam_w, self.cam_h = cam_w, cam_h
        S = bev_size
        ppm = S / bev_meters
        c = S / 2.0

        ys, xs = np.mgrid[0:S, 0:S].astype(np.float64)
        dx, dy = xs - c, ys - c  # dy < 0 => ileri
        in_range = np.hypot(dx, dy) / ppm <= max_range_m

        # Her pikselin "sahibi" olan kamera: köşegenlerle ayrılmış 4 sektör
        sectors = [
            (dy < 0) & (np.abs(dx) <= -dy),  # front
            (dy > 0) & (np.abs(dx) <= dy),  # back
            (dx < 0) & (np.abs(dy) < -dx),  # left
            (dx > 0) & (np.abs(dy) < dx),  # right
        ]

        pts = np.stack([xs.ravel(), ys.ravel(), np.ones(S * S)])
        cands = []
        for k, M in enumerate(homographies):
            M = np.asarray(M, np.float64)
            Minv = np.linalg.inv(M)
            uvw = Minv @ pts
            w = uvw[2]
            # Ufkun "arkasındaki" BEV pikselleri de görüntüye (ters dönmüş) düşer;
            # w işaretini görüntünün alt-orta noktasınınkiyle karşılaştırıp bunları eliyoruz.
            d = M @ np.array([cam_w / 2.0, cam_h * 0.9, 1.0])
            ref_sign = np.sign((Minv @ (d / d[2]))[2])
            with np.errstate(divide="ignore", invalid="ignore"):
                u = uvw[0] / w
                v = uvw[1] / w
            ok = (
                (np.sign(w) == ref_sign)
                & (u >= 0)
                & (u <= cam_w - 1)
                & (v >= 0)
                & (v <= cam_h - 2)  # bilinear komşu kameranın satırına taşmasın
            ).reshape(S, S) & in_range
            cands.append((ok, u.reshape(S, S), v.reshape(S, S) + k * cam_h))

        map_x = np.full((S, S), -100.0, np.float32)
        map_y = np.full((S, S), -100.0, np.float32)
        filled = np.zeros((S, S), bool)
        # 1. tur: piksel kendi sektörünün kamerasından okunur
        # 2. tur: o kamera görmüyorsa, gören başka bir kameradan
        for use_sector in (True, False):
            for k, (ok, u, v) in enumerate(cands):
                sel = ok & ~filled
                if use_sector:
                    sel &= sectors[k]
                map_x[sel] = u[sel]
                map_y[sel] = v[sel]
                filled |= sel

        self.coverage = filled  # debug: True = en az bir kamera görüyor
        self.map1, self.map2 = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
        self._stack_rgb = np.zeros((4 * cam_h, cam_w, 3), np.uint8)
        self._stack_mask = np.zeros((4 * cam_h, cam_w), np.uint8)

    def _remap(self, stack, imgs, interp):
        h = self.cam_h
        for k, im in enumerate(imgs):
            stack[k * h : (k + 1) * h] = im
        return cv2.remap(
            stack, self.map1, self.map2, interp,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    def warp(self, imgs):
        """imgs: [front, back, left, right] BGR görüntüler. Tek thread'den çağrılmalı."""
        return self._remap(self._stack_rgb, imgs, cv2.INTER_LINEAR)

    def warp_mask(self, masks):
        """masks: [front, back, left, right] uint8 sınıf maskeleri (nearest)."""
        return self._remap(self._stack_mask, masks, cv2.INTER_NEAREST)


def placeholder_homographies(cam_w, cam_h, bev_size, bev_meters):
    """
    Eski koddaki tahmini yonca homografileri (KALİBRE DEĞİL).
    Gerçek değerler için zeminde bilinen noktalarla cv2.findHomography yapıp
    homography_file parametresiyle verin.
    """
    ppm = bev_size / bev_meters
    cx = cy = bev_size / 2.0
    d_near, d_far = 0.7 * ppm, 5.0 * ppm
    w_near, w_far = 0.7 * ppm, 3.3 * ppm
    src = np.float32(
        [
            [0.05 * cam_w, 0.35 * cam_h],
            [0.95 * cam_w, 0.35 * cam_h],
            [0.00 * cam_w, 0.95 * cam_h],
            [1.00 * cam_w, 0.95 * cam_h],
        ]
    )
    dst = {
        "front": [[cx - w_far, cy - d_far], [cx + w_far, cy - d_far],
                  [cx - w_near, cy - d_near], [cx + w_near, cy - d_near]],
        "back": [[cx + w_far, cy + d_far], [cx - w_far, cy + d_far],
                 [cx + w_near, cy + d_near], [cx - w_near, cy + d_near]],
        "left": [[cx - d_far, cy + w_far], [cx - d_far, cy - w_far],
                 [cx - d_near, cy + w_near], [cx - d_near, cy - w_near]],
        "right": [[cx + d_far, cy - w_far], [cx + d_far, cy + w_far],
                  [cx + d_near, cy - w_near], [cx + d_near, cy + w_near]],
    }
    return [cv2.getPerspectiveTransform(src, np.float32(dst[n])) for n in CAMS]


# --------------------------------------------------------------------------------------
# YOLO yükleme (eski koddaki loss yaması korunuyor)
# --------------------------------------------------------------------------------------
def load_yolo(engine_path, cam_w, cam_h, logger):
    try:
        import torch.nn as nn
        import ultralytics.utils.loss
        from ultralytics import YOLO

        for name in ("BCEDiceLoss", "MultiChannelDiceLoss"):
            if not hasattr(ultralytics.utils.loss, name):

                class _Dummy(nn.Module):
                    def __init__(self, *a, **k):
                        super().__init__()

                setattr(ultralytics.utils.loss, name, _Dummy)

        if not engine_path:
            logger.warn("engine_path verilmedi -> YOLO kapali, yalnizca BEV calisiyor.")
            return None, False
        if not os.path.exists(engine_path):
            logger.error(f"TensorRT (.engine) bulunamadı: {engine_path}")
            return None, False
        model = YOLO(engine_path, task="segment")
        dummy = np.zeros((cam_h, cam_w, 3), np.uint8)
        # Statik batch=4 engine tam 4 görüntü ister, batch=1 engine tek görüntü.
        try:
            model([dummy] * 4, verbose=False)
            batch_ok = True
        except Exception as e4:  # noqa: BLE001
            logger.warn(f"4'lü warm-up başarısız, tekliye geçiliyor: {e4!r}")
            model(dummy, verbose=False)
            batch_ok = False
        logger.info(f"TensorRT modeli yüklendi: {engine_path} (batch=4: {batch_ok})")
        return model, batch_ok
    except Exception as e:  # noqa: BLE001
        logger.error(f"Model yüklenemedi: {e}")
        return None, False


# --------------------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------------------
class FastCloverleafNode(Node):
    MASK_COLORS = np.array(  # sınıf_id + 1 -> BGR
        [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], np.uint8
    )

    def __init__(self):
        super().__init__("fast_cloverleaf_node")

        p = self.declare_parameter
        topics = {
            "front": p("front_topic", "/front_camera/image_raw/compressed").value,
            "back": p("back_topic", "/rear_camera/image_raw/compressed").value,
            "left": p("left_topic", "/left_camera/image_raw/compressed").value,
            "right": p("right_topic", "/right_camera/image_raw/compressed").value,
        }
        self.cam_w = p("cam_width", 864).value
        self.cam_h = p("cam_height", 480).value
        bev_size = p("bev_size", 960).value
        bev_meters = p("bev_meters", 12.0).value
        max_range = p("max_range_m", 5.0).value
        bev_rate = p("bev_rate_hz", 20.0).value
        yolo_rate = p("yolo_rate_hz", 15.0).value
        self.stale_s = p("stale_timeout_s", 0.5).value
        self.frame_id = p("bev_frame_id", "base_link").value
        jpeg_q = p("jpeg_quality", 80).value
        # TensorRT engine yolu; bos ise YOLO yuklenmez (yalnizca BEV calisir)
        engine = p("engine_path", "").value
        h_file = p("homography_file", "").value  # npz: front/back/left/right 3x3
        # Ya da dogrudan: 36 sayi (front, back, left, right; her biri 3x3 satir-satir),
        # kamera pikseli -> BEV pikseli. four_cam.launch.py bunu igvc_macros.xacro'dan hesaplar.
        h_flat = list(p("homographies", [0.0]).value)
        # Aynı executable'ı iki ayrı süreç olarak çalıştırmak için (GIL çakışmasını önler):
        #   BEV süreci:  run_bev:=true  run_yolo:=false
        #   YOLO süreci: run_bev:=false run_yolo:=true
        self.run_bev = p("run_bev", True).value
        self.run_yolo = p("run_yolo", True).value

        # --- Homografi + LUT ---
        if len(h_flat) == 36:
            Hs = [np.array(h_flat[9 * k:9 * k + 9], np.float64).reshape(3, 3) for k in range(4)]
            self.get_logger().info("Homografiler parametreden (arac geometrisinden) yüklendi.")
        elif h_file and os.path.exists(h_file):
            data = np.load(h_file)
            Hs = [data[n] for n in CAMS]
            self.get_logger().info(f"Kalibre homografiler yüklendi: {h_file}")
        else:
            Hs = placeholder_homographies(self.cam_w, self.cam_h, bev_size, bev_meters)
            self.get_logger().warn("homography_file yok -> TAHMİNİ homografiler kullanılıyor.")
        t0 = time.perf_counter()
        self.warper = BevWarper(self.cam_w, self.cam_h, Hs, bev_size, bev_meters, max_range)
        self.get_logger().info(f"BEV LUT hazır ({(time.perf_counter() - t0) * 1e3:.0f} ms).")

        # --- Durum ---
        self.blank = np.zeros((self.cam_h, self.cam_w, 3), np.uint8)
        self.blank_mask = np.zeros((self.cam_h, self.cam_w), np.uint8)
        self._latest = {n: None for n in CAMS}  # (msg, alınma zamanı)
        self._dec_cache = {n: (None, None) for n in CAMS}  # (msg, decode edilmiş görüntü)
        self._dec_lock = threading.Lock()
        self._bev_lock = threading.Lock()
        self._last_bev = None
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q]
        self._stats = {"bev": [], "yolo": []}
        self._yolo_speed = []  # (preprocess, inference, postprocess) ms / döngü
        lut = np.repeat(self.MASK_COLORS[-1:], 256, axis=0)
        lut[: len(self.MASK_COLORS)] = self.MASK_COLORS
        self._color_luts = [np.ascontiguousarray(lut[:, k]) for k in range(3)]
        self._stats_t = time.monotonic()
        self._batch_ok = True

        raw_size = self.cam_w + 2 * self.cam_h
        self.raw_canvas = np.zeros((raw_size, raw_size, 3), np.uint8)
        cs, ce = self.cam_h, self.cam_h + self.cam_w
        cv2.rectangle(self.raw_canvas, (cs, cs), (ce, ce), (60, 60, 60), 3)

        # --- Callback grupları: aboneler paralel, iki döngü birbirinden bağımsız ---
        sub_group = ReentrantCallbackGroup()
        for name in CAMS:
            self.create_subscription(
                CompressedImage, topics[name],
                lambda msg, n=name: self._store(n, msg),
                qos_profile_sensor_data, callback_group=sub_group,
            )

        self.pub_homo = self.create_publisher(
            CompressedImage, "igvc_vision/merged_image_homo/compressed", 10)
        self.pub_homo_model = self.create_publisher(
            CompressedImage, "igvc_vision/merged_image_homo_model/compressed", 10)
        self.pub_raw = self.create_publisher(
            CompressedImage, "igvc_vision/merged_image_raw/compressed", 10)
        self.pub_mask = self.create_publisher(Image, "igvc_vision/bev_mask", 10)
        self.box_pub = self.create_publisher(Float32MultiArray, "/yolo_coordinates", 10)

        if self.run_bev:
            self.create_timer(1.0 / bev_rate, self.bev_loop,
                              callback_group=MutuallyExclusiveCallbackGroup())

        self.model = None
        if self.run_yolo:
            self.model, self._batch_ok = load_yolo(
                engine, self.cam_w, self.cam_h, self.get_logger())
        if self.model is not None:
            self.create_timer(1.0 / yolo_rate, self.yolo_loop,
                              callback_group=MutuallyExclusiveCallbackGroup())

        self.get_logger().info("Hızlı BEV node başlatıldı.")

    # ---------------------------------------------------------------- yardımcılar
    def _store(self, name, msg):
        # Burada decode YOK: sadece en son mesajı tut.
        self._latest[name] = (msg, time.monotonic())

    def _have_any(self):
        return any(v is not None for v in self._latest.values())

    def _snapshot(self):
        """Her kameranın en son karesini döndürür; yeni değilse önbellekten, eskiyse siyah."""
        now = time.monotonic()
        out = []
        for name in CAMS:
            entry = self._latest[name]
            if entry is None or now - entry[1] > self.stale_s:
                out.append(self.blank)
                continue
            msg = entry[0]
            with self._dec_lock:
                cached_msg, cached_img = self._dec_cache[name]
            if cached_msg is msg:
                out.append(cached_img)
                continue
            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                out.append(self.blank)
                continue
            if img.shape[1] != self.cam_w or img.shape[0] != self.cam_h:
                img = cv2.resize(img, (self.cam_w, self.cam_h))
            with self._dec_lock:
                self._dec_cache[name] = (msg, img)
            out.append(img)
        return out

    def _publish_jpeg(self, pub, img):
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.format = "jpeg"
        msg.data = cv2.imencode(".jpg", img, self.encode_param)[1].tobytes()
        pub.publish(msg)

    def _publish_mono(self, pub, img):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height, msg.width = img.shape
        msg.encoding = "mono8"
        msg.step = img.shape[1]
        msg.data = array.array('B', img.tobytes())  # bytes atamak rclpy'de ~100x yavas
        pub.publish(msg)

    def _tick(self, key, dt):
        self._stats[key].append(dt)
        now = time.monotonic()
        if now - self._stats_t >= 5.0:
            span = now - self._stats_t
            parts = []
            for k, v in self._stats.items():
                if v:
                    parts.append(f"{k}: {len(v) / span:.1f} Hz, ort {np.mean(v) * 1e3:.1f} ms")
            if self._yolo_speed:
                pre, inf, post = np.mean(self._yolo_speed, axis=0)
                parts.append(
                    f"yolo detay (4 kamera toplam): pre {pre:.1f} / infer {inf:.1f} / "
                    f"post {post:.1f} ms, batch={'evet' if self._batch_ok else 'HAYIR'}")
            self.get_logger().info(" | ".join(parts))
            self._stats = {k: [] for k in self._stats}
            self._yolo_speed = []
            self._stats_t = now

    # ---------------------------------------------------------------- BEV döngüsü
    def bev_loop(self):
        if not self._have_any():
            return
        t0 = time.perf_counter()
        f, b, l, r = imgs = self._snapshot()
        bev = self.warper.warp(imgs)
        with self._bev_lock:
            self._last_bev = bev

        if self.pub_homo.get_subscription_count() > 0:
            self._publish_jpeg(self.pub_homo, bev)

        if self.pub_raw.get_subscription_count() > 0:
            h, w = self.cam_h, self.cam_w
            c = self.raw_canvas
            c[0:h, h:h + w] = f
            c[h + w:2 * h + w, h:h + w] = cv2.rotate(b, cv2.ROTATE_180)
            c[h:h + w, 0:h] = cv2.rotate(l, cv2.ROTATE_90_COUNTERCLOCKWISE)
            c[h:h + w, h + w:2 * h + w] = cv2.rotate(r, cv2.ROTATE_90_CLOCKWISE)
            self._publish_jpeg(self.pub_raw, c)

        self._tick("bev", time.perf_counter() - t0)

    # ---------------------------------------------------------------- YOLO döngüsü
    def _infer(self, imgs):
        if self._batch_ok:
            try:
                return self.model(imgs, verbose=False)  # engine batch>=4 ise tek çağrı
            except Exception as e:  # noqa: BLE001
                self._batch_ok = False
                self.get_logger().warn(
                    f"Batch çıkarım olmadı ({e}); tek tek çalışılıyor. "
                    "Engine'i batch=4 ile export edin.")
        return [self.model(im, verbose=False)[0] for im in imgs]

    def _class_map(self, res):
        """
        Segment sonucunu kamera çözünürlüğünde sınıf haritasına çevirir (0 = boş, sınıf + 1).
        masks.xy yerine GPU'daki masks.data kullanılır: CPU'da kontur çıkarma yok.
        """
        h, w = self.cam_h, self.cam_w
        if res.masks is None or res.boxes is None or len(res.boxes) == 0:
            return np.zeros((h, w), np.uint8)
        md = res.masks.data  # (N, mh, mw), letterbox'lı model girişi uzayında
        cls = res.boxes.cls.to(md.device).float()
        lab = ((md > 0.5).float() * (cls + 1).view(-1, 1, 1)).amax(0)
        lab = lab.clamp(max=255).to(torch.uint8).cpu().numpy()
        mh, mw = lab.shape
        g = min(mh / h, mw / w)  # letterbox dolgusunu kırp
        nh, nw = round(h * g), round(w * g)
        top, left = (mh - nh) // 2, (mw - nw) // 2
        lab = lab[top:top + nh, left:left + nw]
        return cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)

    def yolo_loop(self):
        if not self._have_any():
            return
        t0 = time.perf_counter()
        imgs = self._snapshot()
        try:
            results = self._infer(imgs)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Çıkarım hatası: {e}")
            return

        self._yolo_speed.append([
            sum(r.speed.get(k) or 0.0 for r in results)
            for k in ("preprocess", "inference", "postprocess")
        ])

        coords, masks = [], []
        for cam_id, res in enumerate(results):
            masks.append(self._class_map(res))
            if res.boxes is not None and len(res.boxes):
                for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
                    coords.extend([float(cam_id), float(x1), float(x2), float(y1), float(y2)])

        if coords:
            msg = Float32MultiArray()
            msg.data = coords
            self.box_pub.publish(msg)

        bev_mask = self.warper.warp_mask(masks)
        if self.pub_mask.get_subscription_count() > 0:
            self._publish_mono(self.pub_mask, bev_mask)

        if self.pub_homo_model.get_subscription_count() > 0:
            if self.run_bev:
                with self._bev_lock:
                    bev = None if self._last_bev is None else self._last_bev.copy()
            else:
                bev = self.warper.warp(imgs)  # YOLO-only süreçte BEV thread'i yok, güvenli
            if bev is not None:
                # numpy indeksleme yerine cv2.LUT: ~7x daha hızlı
                color = cv2.merge([cv2.LUT(bev_mask, lut) for lut in self._color_luts])
                blended = cv2.addWeighted(bev, 0.55, color, 0.45, 0)
                cv2.copyTo(blended, bev_mask, bev)  # sadece maskeli pikseller
                self._publish_jpeg(self.pub_homo_model, bev)

        self._tick("yolo", time.perf_counter() - t0)


def main():
    rclpy.init()
    node = FastCloverleafNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        if rclpy.ok():  # launch Ctrl+C'de context zaten kapatılmış olabilir
            rclpy.shutdown()


if __name__ == "__main__":
    main()
