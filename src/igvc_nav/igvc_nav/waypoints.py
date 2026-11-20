"""
GPS waypoint takibi: dosya okuma, map'e cevirme ve durum makinesi (ROS'tan bagimsiz, test edilebilir).

path_planner waypoint_mode=true iken kullanir. Waypoint yalnizca "NEREYE" sorusunu cevaplar; yol yine
maliyetli Dijkstra'dan (frontier.plan_to_target) gelir, engel/serit cizgisi asla ezilmez.

Durumlar:
  BEKLEME  Waypoint'lerden uzak: planner standart surusune birebir devam eder (waypoint hicbir hesaba
           girmez). Giris waypoint'i (listenin iki ucundan araca yakin olani) activation_m'den yakin,
           aracin onunde (+-activation_fov_deg) ve mesafe son approach_window_s'de azaliyorsa AKTIF.
           Hangi uctan girildiyse sira ona gore kurulur (parkur iki yonde kosulabilir).
  AKTIF    Siradaki waypoint hedef. Varis: mesafe <= reach_radius - reach_margin.
           Kacirma: miss_near'dan yakina gelinmis ama varilmamis ve en yakin noktadan miss_away
           uzaklasilmis -> retries kez geri donup tekrar denenir, sonra atlanir.
           Zaman asimi (timeout_s): atlanir. Son waypoint gecilince/atlaninca BITTI.
  BITTI    Standart surus. Ilk exit_hint_m boyunca "ileri" yonu son iki waypoint'in gidis yonu
           (cizgisiz No Man's Land'de burun yana bakarken yanlis yone frontier secilmesin).
           Kosu boyunca bir daha AKTIF olunmaz (reset() ile yeniden baslar).
"""
import math
import os
from collections import deque

import yaml

STANDBY, ACTIVE, DONE = 'BEKLEME', 'AKTIF', 'BITTI'


class WaypointFileError(ValueError):
    """waypoints.yaml hatali."""


def default_waypoints_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('igvc_nav'), 'config', 'waypoints.yaml')
    except Exception:  # noqa: BLE001  kaynak agacindan (test)
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(here, '..', 'config', 'waypoints.yaml'))


def load_waypoints(path=None):
    """-> {'datum': (lat, lon) | None, 'reach_radius_m': float, 'waypoints': [{name, lat, lon}]}"""
    path = path or default_waypoints_path()
    if not os.path.exists(path):
        raise WaypointFileError(f'Waypoint dosyasi bulunamadi: {path}')
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    wps = []
    for k, w in enumerate(doc.get('waypoints') or []):
        try:
            wps.append({'name': str(w.get('name', f'wp{k}')),
                        'lat': float(w['latitude']), 'lon': float(w['longitude'])})
        except (KeyError, TypeError, ValueError) as e:
            raise WaypointFileError(f'{path}: {k}. waypoint latitude/longitude eksik ya da hatali') from e
    if not wps:
        raise WaypointFileError(f'{path}: waypoint yok')
    d = doc.get('datum')
    datum = (float(d['latitude']), float(d['longitude'])) if d else None
    return {'datum': datum, 'reach_radius_m': float(doc.get('reach_radius_m', 1.0)), 'waypoints': wps}


def to_map(waypoints, lat0, lon0):
    """Enlem/boylam -> datum'da yerel ENU (map: x dogu, y kuzey) [(x, y), ...].
    gps_odometry ile ayni donusum (igvc_localization.geodesy.EnuFrame)."""
    from igvc_localization.geodesy import EnuFrame
    enu = EnuFrame(lat0, lon0, 0.0)
    out = []
    for w in waypoints:
        e, n, _ = enu.to_enu(w['lat'], w['lon'], 0.0)
        out.append((e, n))
    return out


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class WaypointManager:
    """Waypoint sirasi ve durum. Konumlar map cercevesinde (x, y) [m], yaw [rad], t [s]."""

    def __init__(self, points, names=None, *, reach_radius=1.0, reach_margin=0.3, activation=4.5,
                 activation_fov_deg=100.0, approach_window_s=1.0, approach_min_m=0.05,
                 timeout_s=40.0, retries=1, miss_near=2.0, miss_away=1.5, exit_hint_m=4.0,
                 jump_m=2.0):
        if not points:
            raise ValueError('waypoint listesi bos')
        self.points = [(float(x), float(y)) for x, y in points]
        self.names = list(names) if names else [f'wp{k}' for k in range(len(points))]
        self.reach = max(0.05, reach_radius - reach_margin)
        self.activation = activation
        self.fov = math.radians(activation_fov_deg)
        self.approach_window_s = approach_window_s
        self.approach_min_m = approach_min_m
        self.timeout_s = timeout_s
        self.retries = int(retries)
        self.miss_near = miss_near
        self.miss_away = miss_away
        self.exit_hint_m = exit_hint_m
        self.jump_m = jump_m
        self.reset()

    # ------------------------------------------------------------------ durum
    def reset(self):
        self.state = STANDBY
        self.order = list(range(len(self.points)))
        self.i = 0                       # order icindeki konum
        self.attempts = 0
        self.d_min = math.inf
        self.t_start = None
        self.hist = deque()              # BEKLEME: (t, giris mesafesi)
        self.last = None                 # son kabul edilen poz (x, y, t)
        self.exit_dir = None             # BITTI: map'te birim vektor
        self.exit_travel = 0.0
        self.log = []                    # [(t, ad, sonuc)] varildi / atlandi

    @property
    def n(self):
        return len(self.points)

    def target_index(self):
        return self.order[self.i] if self.state == ACTIVE and self.i < self.n else None

    def target(self):
        k = self.target_index()
        return None if k is None else self.points[k]

    def next_point(self):
        """Hedeften sonraki gercek waypoint; son waypoint'teyse None."""
        if self.state != ACTIVE or self.i + 1 >= self.n:
            return None
        return self.points[self.order[self.i + 1]]

    def heading_out(self):
        """Son iki waypoint'in (sira yonunde) gidis yonu, birim vektor; tek waypoint ise None."""
        if self.n < 2:
            return None
        a, b = self.points[self.order[-2]], self.points[self.order[-1]]
        d = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
        return ((b[0] - a[0]) / d, (b[1] - a[1]) / d)

    def exit_hint(self):
        """BITTI'nin ilk exit_hint_m'sinde 'ileri' yonu (map birim vektor), yoksa None."""
        if self.state == DONE and self.exit_dir is not None and self.exit_travel < self.exit_hint_m:
            return self.exit_dir
        return None

    def distance(self, x, y):
        p = self.target()
        return None if p is None else math.hypot(p[0] - x, p[1] - y)

    # ------------------------------------------------------------------ guncelleme
    def update(self, x, y, yaw, t):
        """Yeni global poz. Olay listesi dondurur: [(tur, mesaj)], tur: aktif|varildi|tekrar|atlandi|bitti|sicrama."""
        ev = []
        if self.last is not None and t < self.last[2]:
            # Saat geri gitti (simulasyon sifirlandi): zaman pencereleri gecersiz
            self.hist.clear()
            self.last = None
            if self.t_start is not None:
                self.t_start = t
        if self.last is not None:
            lx, ly, lt = self.last
            step = math.hypot(x - lx, y - ly)
            if step > self.jump_m and t - lt < 1.0:
                # Lokalizasyon yeniden kilitlendi (EKF GPS/yon sifirlamasi; tekil GPS sicramalarini
                # EKF kapisi zaten eler). Tek seferlik: yeni konum referans olur, bu ornekte gecis
                # degerlendirilmez, kacirma/yaklasma gecmisi sifirlanir. (Eskiden sicrama hep eski
                # referansa gore olculuyordu: arac uzaklasirsa yonetici sonsuza kadar donardi.)
                self.last = (x, y, t)
                self.hist.clear()
                self.d_min = math.inf
                ev.append(('sicrama', f'global poz {step:.1f} m sicradi (lokalizasyon yeniden '
                                      f'kilitlendi): yeni konum esas aliniyor'))
                return ev
            if self.state == DONE:
                self.exit_travel += step
        self.last = (x, y, t)

        if self.state == STANDBY:
            ev += self._standby(x, y, yaw, t)
        elif self.state == ACTIVE:
            ev += self._active(x, y, yaw, t)
        return ev

    def _standby(self, x, y, yaw, t):
        d_first = math.hypot(self.points[0][0] - x, self.points[0][1] - y)
        d_last = math.hypot(self.points[-1][0] - x, self.points[-1][1] - y)
        rev = d_last < d_first
        entry = self.points[-1] if rev else self.points[0]
        d = min(d_first, d_last)
        self.hist.append((t, d))
        # hist[0]: en az approach_window_s eski olan en yeni ornek (henuz yoksa en eski)
        while len(self.hist) > 1 and t - self.hist[1][0] >= self.approach_window_s:
            self.hist.popleft()
        t0, d0 = self.hist[0]
        if d > self.activation:
            return []
        bearing = math.atan2(entry[1] - y, entry[0] - x)
        ahead = abs(_wrap(bearing - yaw)) <= self.fov or d < self.reach
        approaching = (t - t0) >= 0.5 * self.approach_window_s and d0 - d >= self.approach_min_m
        if not (ahead and approaching):
            return []
        self.order = list(range(self.n))[::-1] if rev else list(range(self.n))
        self.state, self.i = ACTIVE, 0
        self._start_target(t)
        names = ' -> '.join(self.names[k] for k in self.order)
        return [('aktif', f'waypoint modu AKTIF ({"ters" if rev else "duz"} sira: {names}), '
                          f'giris {d:.1f} m')] + self._active(x, y, yaw, t)

    def _start_target(self, t):
        self.attempts, self.d_min, self.t_start = 0, math.inf, t

    def _advance(self, t, result):
        k = self.order[self.i]
        self.log.append((t, self.names[k], result))
        self.i += 1
        if self.i >= self.n:
            self.state = DONE
            self.exit_dir, self.exit_travel = self.heading_out(), 0.0
            return [('bitti', 'tum waypoint\'ler tamamlandi, standart surus')]
        self._start_target(t)
        return []

    def _active(self, x, y, yaw, t):
        ev = []
        k = self.order[self.i]
        d = math.hypot(self.points[k][0] - x, self.points[k][1] - y)
        if d <= self.reach:
            ev.append(('varildi', f'{self.names[k]} varildi ({d:.2f} m)'))
            return ev + self._advance(t, 'varildi')
        self.d_min = min(self.d_min, d)
        if self.d_min <= self.miss_near and d >= self.d_min + self.miss_away:
            if self.attempts < self.retries:
                self.attempts += 1
                ev.append(('tekrar', f'{self.names[k]} kacirildi (en yakin {self.d_min:.2f} m), '
                                     f'geri donup tekrar deneniyor ({self.attempts}/{self.retries})'))
                self.d_min = math.inf
                self.t_start = t
                return ev
            ev.append(('atlandi', f'{self.names[k]} kacirildi (en yakin {self.d_min:.2f} m), atlandi'))
            return ev + self._advance(t, 'atlandi')
        if self.t_start is not None and t - self.t_start > self.timeout_s:
            ev.append(('atlandi', f'{self.names[k]} zaman asimi ({self.timeout_s:.0f} s), atlandi'))
            return ev + self._advance(t, 'atlandi')
        return ev

    def status(self, x=None, y=None):
        k = self.target_index()
        s = f'durum={self.state}'
        if k is not None:
            s += f' hedef={self.names[k]} ({self.i + 1}/{self.n}) deneme={self.attempts}'
            if x is not None:
                s += f' mesafe={self.distance(x, y):.2f}'
        if self.log:
            s += ' | ' + ', '.join(f'{n}:{r}' for _, n, r in self.log)
        return s
