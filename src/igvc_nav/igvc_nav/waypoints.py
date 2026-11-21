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
           rearm=False (varsayilan): kosu boyunca bir daha AKTIF olunmaz (reset() ile yeniden baslar).
           rearm=True: BITTI'den beri en az rearm_travel_m yol gidilmis VE arac butun waypoint'lere
           rearm_dist_m'den uzaksa yeniden BEKLEME (ikinci tur). Iki sart birden: yalnizca mesafe, son
           waypoint'ten cikarken geri donuste erken tetiklenir; yalnizca yol, kisa parkurda erken tetiklenir.
           max_laps > 0 ise o kadar tamamlanan geciste durur (0 = sinirsiz). Yeni turda yon yine giris
           ucuna gore secilir; iki uc neredeyse esit uzaksa onceki turun yonu tercih edilir.
  YAKLASMA (approach=True; varsayilan kapali): BEKLEME'de giris waypoint'i approach_m'den yakin ve
           aracin onunde (+-approach_fov_deg) ise. Planner "nereye" kararina waypoint yonunu ekler (bkz.
           path_planner); activation_m'de AKTIF olunur (yon tespiti ayni). approach_m + approach_hyst_m'den
           uzaklasinca ya da approach_timeout_s boyunca approach_progress_m yaklasamazsa BEKLEME'ye donulur;
           ilerleme yokken donuldugunde approach_cooldown_s boyunca tekrar YAKLASMA'ya girilmez.
"""
import math
import os
from collections import deque

import yaml

STANDBY, ACTIVE, DONE, APPROACH = 'BEKLEME', 'AKTIF', 'BITTI', 'YAKLASMA'


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
                 jump_m=2.0, rearm=False, rearm_travel_m=30.0, rearm_dist_m=10.0, max_laps=0,
                 approach=False, approach_m=20.0, approach_fov_deg=90.0, approach_hyst_m=3.0,
                 approach_timeout_s=20.0, approach_progress_m=1.0, approach_cooldown_s=30.0):
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
        self.rearm = bool(rearm)
        self.rearm_travel_m = rearm_travel_m
        self.rearm_dist_m = rearm_dist_m
        self.max_laps = int(max_laps)
        self.approach = bool(approach)
        self.approach_m = approach_m
        self.approach_fov_deg = approach_fov_deg
        self.approach_hyst_m = approach_hyst_m
        self.approach_timeout_s = approach_timeout_s
        self.approach_progress_m = approach_progress_m
        self.approach_cooldown_s = approach_cooldown_s
        self.reset()

    # ------------------------------------------------------------------ durum
    def reset(self):
        """Tam sifirlama (waypoint_mode acilinca): tur sayaci ve gecmis de sifirlanir."""
        self.laps = 0                    # tamamlanan gecis (tur) sayisi
        self.lap_logs = []               # onceki turlarin log'lari
        self.prev_rev = None             # onceki turun yonu (True = ters)
        self._new_lap()

    def _new_lap(self):
        """Tur basi: tur sayaci / gecmis korunur, gecis durumu sifirlanir."""
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
        self.app_entry = None            # YAKLASMA: giris waypoint indeksi
        self.app_best = math.inf         # YAKLASMA: girise en yakin mesafe
        self.app_t = None                # YAKLASMA: son ilerleme zamani
        self.app_cooldown_until = None   # ilerleme yokken birakildiysa: bu zamana kadar girilmez

    @property
    def n(self):
        return len(self.points)

    def target_index(self):
        return self.order[self.i] if self.state == ACTIVE and self.i < self.n else None

    def approach_index(self):
        return self.app_entry if self.state == APPROACH else None

    def approach_target(self):
        """YAKLASMA'da giris waypoint'i (map), degilse None."""
        k = self.approach_index()
        return None if k is None else self.points[k]

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
            self.app_t, self.app_best, self.app_cooldown_until = (t if self.app_t is not None else None,
                                                                  math.inf, None)
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
                if self.state == APPROACH:
                    self.app_best, self.app_t = math.inf, t
                ev.append(('sicrama', f'global poz {step:.1f} m sicradi (lokalizasyon yeniden '
                                      f'kilitlendi): yeni konum esas aliniyor'))
                return ev
            if self.state == DONE:
                self.exit_travel += step
        self.last = (x, y, t)

        if self.state in (STANDBY, APPROACH):
            ev += self._standby(x, y, yaw, t)
            if self.state == APPROACH:
                ev += self._approach(x, y, t)
        elif self.state == ACTIVE:
            ev += self._active(x, y, yaw, t)
        elif self.state == DONE and self.rearm:
            ev += self._done(x, y, t)
        return ev

    def _done(self, x, y, t):
        """BITTI -> BEKLEME (rearm): yeterince yol gidildi ve butun waypoint'lerden uzak."""
        if self.max_laps > 0 and self.laps >= self.max_laps:
            return []
        if self.exit_travel < self.rearm_travel_m:
            return []
        d_min = min(math.hypot(px - x, py - y) for px, py in self.points)
        if d_min <= self.rearm_dist_m:
            return []
        self.lap_logs.append(self.log)
        travel = self.exit_travel
        last = self.last
        self._new_lap()
        self.last = last                 # sicrama kontrolu kesintisiz surer
        return [('yeniden', f'tur {self.laps} tamam ({travel:.0f} m yol, en yakin waypoint {d_min:.1f} m): '
                            f'waypoint modu yeniden BEKLEME (tur {self.laps + 1})')]

    def _standby(self, x, y, yaw, t):
        d_first = math.hypot(self.points[0][0] - x, self.points[0][1] - y)
        d_last = math.hypot(self.points[-1][0] - x, self.points[-1][1] - y)
        rev = d_last < d_first
        if self.prev_rev is not None and abs(d_last - d_first) < 1.0:
            rev = self.prev_rev          # iki uc neredeyse esit uzak: onceki turun yonu
        entry = self.points[-1] if rev else self.points[0]
        d = min(d_first, d_last)
        self.hist.append((t, d))
        # hist[0]: en az approach_window_s eski olan en yeni ornek (henuz yoksa en eski)
        while len(self.hist) > 1 and t - self.hist[1][0] >= self.approach_window_s:
            self.hist.popleft()
        t0, d0 = self.hist[0]
        if d > self.activation:
            if self.state == STANDBY and self.approach and d <= self.approach_m and \
                    (self.app_cooldown_until is None or t >= self.app_cooldown_until):
                bearing = math.atan2(entry[1] - y, entry[0] - x)
                if abs(_wrap(bearing - yaw)) <= math.radians(self.approach_fov_deg):
                    self.state = APPROACH
                    self.app_entry = self.n - 1 if rev else 0
                    self.app_best, self.app_t = d, t
                    return [('yaklasma', f'YAKLASMA: giris {self.names[self.app_entry]} {d:.1f} m '
                                         f'({"ters" if rev else "duz"} sira bekleniyor)')]
            return []
        bearing = math.atan2(entry[1] - y, entry[0] - x)
        ahead = abs(_wrap(bearing - yaw)) <= self.fov or d < self.reach
        approaching = (t - t0) >= 0.5 * self.approach_window_s and d0 - d >= self.approach_min_m
        if not (ahead and approaching):
            return []
        self.order = list(range(self.n))[::-1] if rev else list(range(self.n))
        self.state, self.i = ACTIVE, 0
        self.app_entry, self.app_best, self.app_t = None, math.inf, None
        self._start_target(t)
        names = ' -> '.join(self.names[k] for k in self.order)
        return [('aktif', f'waypoint modu AKTIF ({"ters" if rev else "duz"} sira: {names}), '
                          f'giris {d:.1f} m')] + self._active(x, y, yaw, t)

    def _approach(self, x, y, t):
        """YAKLASMA: uzaklasma / ilerleme yok -> BEKLEME."""
        px, py = self.points[self.app_entry]
        d = math.hypot(px - x, py - y)
        name = self.names[self.app_entry]
        if d > self.approach_m + self.approach_hyst_m:
            self._leave_approach()
            return [('yaklasma_iptal', f'YAKLASMA birakildi: {name} {d:.1f} m (menzil disi), standart surus')]
        if d < self.app_best - self.approach_progress_m:
            self.app_best, self.app_t = d, t
        elif self.app_t is not None and t - self.app_t > self.approach_timeout_s:
            self._leave_approach()
            self.app_cooldown_until = t + self.approach_cooldown_s
            return [('yaklasma_iptal', f'YAKLASMA birakildi: {self.approach_timeout_s:.0f} s ilerleme yok '
                                       f'({name} {d:.1f} m), {self.approach_cooldown_s:.0f} s standart surus')]
        return []

    def _leave_approach(self):
        self.state, self.app_entry, self.app_best, self.app_t = STANDBY, None, math.inf, None

    def _start_target(self, t):
        self.attempts, self.d_min, self.t_start = 0, math.inf, t

    def _advance(self, t, result):
        k = self.order[self.i]
        self.log.append((t, self.names[k], result))
        self.i += 1
        if self.i >= self.n:
            self.state = DONE
            self.laps += 1
            self.prev_rev = self.order[0] != 0
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
        if self.rearm:
            s += f' tur={self.laps + (0 if self.state == DONE else 1)}'
            if self.max_laps > 0:
                s += f'/{self.max_laps}'
        if k is not None:
            s += f' hedef={self.names[k]} ({self.i + 1}/{self.n}) deneme={self.attempts}'
            if x is not None:
                s += f' mesafe={self.distance(x, y):.2f}'
        ka = self.approach_index()
        if ka is not None:
            s += f' giris={self.names[ka]}'
            if x is not None:
                s += f' mesafe={math.hypot(self.points[ka][0] - x, self.points[ka][1] - y):.1f}'
        if self.log:
            s += ' | ' + ', '.join(f'{n}:{r}' for _, n, r in self.log)
        return s
