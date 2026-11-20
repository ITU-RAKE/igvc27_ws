"""
Diferansiyel / skid steer arac icin EKF cekirdegi (ROS'tan bagimsiz, test edilebilir).

Durum (n = 6):  x = [px, py, yaw, v, w, b_g]
  px, py  konum [m] (filtre cercevesinde: odom ya da map)
  yaw     yon [rad], [-pi, pi]
  v       ileri hiz [m/s] (govde x'i; skid steer yana kaymaz -> vy yok)
  w       donus hizi [rad/s]
  b_g     jiroskop z sapmasi [rad/s] (IMU w olcumu = w + b_g)

Hareket modeli: sabit hiz + sabit donus hizi, orta nokta entegrasyonu
  m   = yaw + w*dt/2
  px' = px + v cos(m) dt          py' = py + v sin(m) dt
  yaw'= yaw + w dt                v' = v,  w' = w,  b_g' = b_g
  Jacobian analitik ve modelle birebir tutarli (eski koddaki gibi golgelenen degisken yok).

Surec gurultusu: surekli beyaz gurultu spektral yogunluklari q (birim^2/s) -> Q = diag(q) dt.

Olcum guncellemesi: asenkron ve sirali. Her olcum kendi zamanina kadar predict edilip yalnizca
bir kez kullanilir. Aci bilesenlerinde inovasyon [-pi, pi]'ye sarilir. Joseph formu kovaryans
guncellemesi (sayisal olarak pozitif tanimli kalir). Mahalanobis (chi^2) kapisi ile aykiri olcum
reddi. inv yerine solve.
"""
import math

import numpy as np

PX, PY, YAW, V, W, BG = range(6)
N = 6

# chi^2 ters CDF, %99.9 (aykiri olcum kapisi), serbestlik derecesi 1..3
CHI2_999 = {1: 10.828, 2: 13.816, 3: 16.266}


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Ekf:
    def __init__(self, process_noise, x0=None, p0=None):
        """process_noise: N uzunlukta spektral yogunluk (birim^2/s)."""
        self.q = np.asarray(process_noise, float)
        assert self.q.shape == (N,)
        self.x = np.zeros(N) if x0 is None else np.asarray(x0, float).copy()
        self.P = np.eye(N) if p0 is None else np.asarray(p0, float).copy()
        self._F = np.eye(N)
        self._I = np.eye(N)

    # ------------------------------------------------------------------ predict
    def predict(self, dt):
        if dt <= 0.0:
            return
        x = self.x
        yaw, v, w = x[YAW], x[V], x[W]
        m = yaw + 0.5 * w * dt
        c, s = math.cos(m), math.sin(m)
        x[PX] += v * c * dt
        x[PY] += v * s * dt
        x[YAW] = wrap(yaw + w * dt)

        F = self._F
        F[PX, YAW] = -v * s * dt
        F[PX, V] = c * dt
        F[PX, W] = -0.5 * v * s * dt * dt
        F[PY, YAW] = v * c * dt
        F[PY, V] = s * dt
        F[PY, W] = 0.5 * v * c * dt * dt
        F[YAW, W] = dt
        P = F @ self.P @ F.T
        P[np.diag_indices(N)] += self.q * dt
        self.P = 0.5 * (P + P.T)

    # ------------------------------------------------------------------ update
    def update(self, z, h, H, R, angle_rows=(), gate=True):
        """
        z: olcum (m,), h: tahmin edilen olcum (m,), H: (m, N), R: (m, m)
        angle_rows: inovasyonu sarilacak satirlar. gate: chi^2 kapisi.
        Donus: (kabul edildi mi, Mahalanobis^2)
        """
        y = np.asarray(z, float) - np.asarray(h, float)
        for i in angle_rows:
            y[i] = wrap(y[i])
        PHt = self.P @ H.T
        S = H @ PHt + R
        d2 = float(y @ np.linalg.solve(S, y))
        if gate and d2 > CHI2_999.get(len(y), 16.266 + 2.0 * (len(y) - 3)):
            return False, d2
        K = np.linalg.solve(S.T, PHt.T).T          # K = P H^T S^-1
        self.x += K @ y
        self.x[YAW] = wrap(self.x[YAW])
        IKH = self._I - K @ H
        P = IKH @ self.P @ IKH.T + K @ R @ K.T     # Joseph formu
        self.P = 0.5 * (P + P.T)
        return True, d2

    # ------------------------------------------------------------------ hazir olcum modelleri
    _H_WHEEL = np.zeros((2, N)); _H_WHEEL[0, V] = 1.0; _H_WHEEL[1, W] = 1.0
    _H_GYRO = np.zeros((1, N)); _H_GYRO[0, W] = 1.0; _H_GYRO[0, BG] = 1.0
    _H_YAW = np.zeros((1, N)); _H_YAW[0, YAW] = 1.0

    def update_wheel(self, v, w, std_v, std_w, gate=True):
        """Tekerlek odometrisi: ileri hiz ve donus hizi."""
        return self.update((v, w), (self.x[V], self.x[W]), self._H_WHEEL,
                           np.diag((std_v * std_v, std_w * std_w)), gate=gate)

    def update_gyro(self, w_meas, std, gate=True):
        """IMU jiroskop z: w + b_g."""
        return self.update((w_meas,), (self.x[W] + self.x[BG],), self._H_GYRO,
                           np.array([[std * std]]), gate=gate)

    def update_yaw(self, yaw_meas, std, gate=True):
        """Mutlak (ya da bagil) yon olcumu, aci sarmali."""
        return self.update((yaw_meas,), (self.x[YAW],), self._H_YAW,
                           np.array([[std * std]]), angle_rows=(0,), gate=gate)

    def reset_yaw(self, yaw, std):
        """Yonu verilen degere yeniden kilitle (yon kilitlenmesinden kurtarma): yaw'in capraz
        kovaryansi silinir, belirsizligi std^2. Konum/hiz/sapma degismez."""
        self.x[YAW] = wrap(yaw)
        self.P[YAW, :] = 0.0
        self.P[:, YAW] = 0.0
        self.P[YAW, YAW] = std * std

    def reset_position(self, px, py, cov_xy, lever=(0.0, 0.0), var=100.0):
        """
        Konumu olcume yeniden kilitle (kapi kilitlenmesinden kurtarma): konum belirsizligini
        buyut, capraz kovaryansi sil, kapisiz guncelle. Yon/hiz/sapma korunur.
        """
        self.P[(PX, PY), :] = 0.0
        self.P[:, (PX, PY)] = 0.0
        self.P[PX, PX] = self.P[PY, PY] = var
        return self.update_position(px, py, cov_xy, lever=lever, gate=False)

    def update_position(self, px, py, cov_xy, lever=(0.0, 0.0), gate=True):
        """
        Konum olcumu (GPS). lever: anten konumu govde cercevesinde (x ileri, y sol) [m];
        olculen nokta = arac merkezi + R(yaw) lever.
        """
        yaw = self.x[YAW]
        c, s = math.cos(yaw), math.sin(yaw)
        lx, ly = lever
        h = (self.x[PX] + c * lx - s * ly, self.x[PY] + s * lx + c * ly)
        H = np.zeros((2, N))
        H[0, PX] = 1.0
        H[1, PY] = 1.0
        H[0, YAW] = -s * lx - c * ly
        H[1, YAW] = c * lx - s * ly
        return self.update((px, py), h, H, np.asarray(cov_xy, float), gate=gate)


# ============================================================================ yon kurtarma hakemi
class CourseTracker:
    """
    GPS konumlarindan rota (gidis yonu) tahmini: arac DUZ ve yeterince HIZLI giderken, iz
    min_dist (ve GPS gurultusune gore dist_sigma * std) kadar uzayinca bastan sona yon.
    Rota std ~ sqrt(2) * std / mesafe: dist_sigma 16 -> en fazla ~5 deg (RTK 2 cm: 1.5 m'de ~1 deg).
    Donus, yavaslama ya da ileri/geri degisimi izi sifirlar (rota yalnizca duz parcadan).
    Geri giderken (v < 0) rota 180 derece cevrilir: aracin burnunun yonu.
    """

    def __init__(self, min_dist=1.5, dist_sigma=16.0, max_turn=0.3, min_speed=0.3, max_age=30.0):
        self.min_dist, self.dist_sigma = min_dist, dist_sigma
        self.max_turn, self.min_speed, self.max_age = max_turn, min_speed, max_age
        self.track = []           # [(t, x, y)]
        self.sign = 0

    def reset(self):
        self.track, self.sign = [], 0

    def add(self, t, x, y, v, w, pos_std):
        """GPS konumu + filtrenin hiz/donus tahmini. Rota hazirsa (yon [rad]) yoksa None."""
        sign = 1 if v > 0 else -1
        if abs(w) > self.max_turn or abs(v) < self.min_speed or (self.sign and sign != self.sign):
            self.reset()
            return None
        self.sign = sign
        self.track.append((t, x, y))
        while self.track and t - self.track[0][0] > self.max_age:
            self.track.pop(0)
        t0, x0, y0 = self.track[0]
        d = math.hypot(x - x0, y - y0)
        if d < max(self.min_dist, self.dist_sigma * pos_std):
            return None
        course = math.atan2(y - y0, x - x0)
        return wrap(course + math.pi) if sign < 0 else course


def arbitrate_yaw(course, imu_yaw, filter_yaw, tol, margin):
    """
    IMU ile filtre yonu uyusmuyorken GPS rotasi hakem. Dondurur:
      'imu'    rota IMU'ya tol icinde ve filtreden en az margin daha yakin: filtre yanlis -> IMU'ya kilitle
      'filter' rota filtreye tol icinde ve IMU'dan en az margin daha yakin: IMU bozuk -> reddetmeye devam
      None     belirsiz (ikisi de rotadan uzak ya da fark margin'den kucuk): bekle
    Rota gurultusu margin'in altinda tutulur (CourseTracker dist_sigma).
    """
    e_imu = abs(wrap(course - imu_yaw))
    e_filt = abs(wrap(course - filter_yaw))
    if e_imu <= tol and e_filt - e_imu >= margin:
        return 'imu'
    if e_filt <= tol and e_imu - e_filt >= margin:
        return 'filter'
    return None
