"""
path_tracker'in ROS'tan bagimsiz cekirdegi (test edilebilir): dinamik lookahead'li pure pursuit.

Planner her ~0.1 s yeni bir yol yollar; uzunlugu ve sekli surekli degisir (frontier: menzil
sinirina kadar uzun; en-uzun-yol modu: dar alanda kisa ve kivrimli). Sabit lookahead (eski 0.8 m)
dar S'lerde virajin icinden kesip dubaya surtunuyordu. Burada her kontrol adiminda:

  1) Yol 5 cm araliklarla yeniden orneklenir; en yakin nokta yalnizca yolun ilk
     closest_search_m metresinde aranir (S seklindeki yolun ileri kismina atlamasin).
  2) Ilerideki egrilik: en yakin noktadan curvature_window_m boyunca en buyuk egrilik (1/m).
     Planner'in yolu 8 yonlu izgara: 45 deg kirilimlar (merdiven) 3 noktali egrilikte 2-5 1/m
     gorunur ve hizi ~0.3 m/s'ye kilitler. Egrilik, +-path_smoothing kayan ortalamayla
     yumusatilmis yoldan hesaplanir (yalnizca hiz/lookahead icin; hedef nokta ham yolda kalir,
     kose kesip engele yaklasmaz).
  3) Lookahead = lookahead_min + lookahead_time * |hiz|, ust sinir lookahead_max,
     egrilik siniri lookahead_curvature_factor / egrilik (dar virajda kisalir), alt sinir
     lookahead_min. Kalan yol daha kisaysa hedef yolun sonu.
  4) Acisal hiz = kp * yon hatasi (eski kontrolcu, kp 1.0 ayari korunur).
  5) Dogrusal hiz:
       v = max_v * (1 - |hata| / slowdown_yaw_error)            (yon hatasi, eskisi gibi)
       v = min(v, sqrt(max_lateral_accel / egrilik))             (viraj hiz siniri)
       v = v * (1 - turn_slowdown * |w| / max_w)                 (acisal hiz verirken ek yavaslama)
     alt sinir min_v; |hata| > in_place_yaw_error ise 0 (yerinde don, ileri itme).
  5b) clearance_slowdown > 0 ise: yolun onumuzdeki clearance_window metresindeki en yuksek harita
     maliyeti c (planner pose.z'ye yazar): v <= max_v * (1 - clearance_slowdown * (c / 99)^clearance_power).
     clearance_power > 1: orta maliyette (serit kenari, genis gecit) az, oldurucu sinira yakin cok yavaslar.
  6) Geri vites yalnizca yol bir RECOVERY yoluysa (reverse=True: yol geldigi anda aracin
     tamamen arkasindaydi; bunu cagiran karar verir). Ileri bir yolun hedefi arac donerken
     arkada kalirsa geri gitmez, yerinde doner.
"""
import math

import numpy as np


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def resample(path, step=0.05):
    """[(x, y), ...] -> esit aralikli Nx2 dizi (ilk ve son nokta korunur)."""
    p = np.asarray(path, float).reshape(-1, 2)
    if len(p) < 2:
        return p
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return p[:1]
    t = np.append(np.arange(0.0, s[-1], step), s[-1])
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def smooth(p, half_window, step=0.05):
    """Kayan ortalama (+-half_window m); uclarda pencere daralir."""
    k = int(round(half_window / step))
    if k < 1 or len(p) < 3:
        return p
    c = np.vstack([np.zeros((1, 2)), np.cumsum(p, axis=0)])
    i = np.arange(len(p))
    lo, hi = np.maximum(0, i - k), np.minimum(len(p), i + k + 1)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def curvature(p, i0, window, step=0.05):
    """p[i0:] uzerinde window metre boyunca en buyuk egrilik (uc nokta cemberi, 1/m)."""
    span = max(2, int(round(0.15 / step)))       # 15 cm aralikli uc nokta: piksel merdivenini yumusatir
    i1 = min(len(p) - 1, i0 + int(window / step))
    best = 0.0
    for i in range(i0, i1 - 2 * span + 1):
        a, b, c = p[i], p[i + span], p[i + 2 * span]
        ab, bc, ca = np.hypot(*(b - a)), np.hypot(*(c - b)), np.hypot(*(a - c))
        cross = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if ab * bc * ca > 1e-9:
            best = max(best, 2.0 * cross / (ab * bc * ca))
    return best


DEFAULTS = dict(
    kp=1.0, max_v=1.0, min_v=0.1, max_w=2.0, slowdown_yaw_error=0.8,
    lookahead_min=0.35, lookahead_max=1.0, lookahead_time=0.6, lookahead_curvature_factor=0.5,
    curvature_window=1.2, max_lateral_accel=0.5, turn_slowdown=0.5, in_place_yaw_error=1.2,
    closest_search=1.0, goal_tolerance=0.1, path_smoothing=0.25,
    clearance_slowdown=0.0, clearance_window=1.5, clearance_power=1.0,
)


def is_recovery_path(local_path):
    """Planner'in RECOVERY yolu mu: birkac noktali (planner 3 nokta yollar: 0, -0.5, -1 m) ve tamami
    arkada. Izgaradan cizilen yollar (5 cm aralikli, onlarca nokta) arkada olsa da ileri vitesle
    izlenir: iz hafizasi cikmazda arka yaridan planlayinca arac yerinde donup ileri gider."""
    xs = [x for x, _ in local_path]
    return 2 <= len(xs) <= 4 and xs[-1] < -0.1 and all(x < 0.05 for x in xs[1:])


def cost_ahead(path, costs, pose, window, closest_search=1.0):
    """Yolda aracin en yakin noktasindan window metre boyunca en yuksek maliyet (0..99)."""
    if not costs or len(costs) != len(path):
        return 0.0
    p = np.asarray(path, float)
    seg = np.hypot(*np.diff(p, axis=0).T) if len(p) > 1 else np.zeros(0)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    near = s <= closest_search
    i0 = int(np.argmin(np.where(near, (p[:, 0] - pose[0]) ** 2 + (p[:, 1] - pose[1]) ** 2, np.inf)))
    sel = (s >= s[i0]) & (s <= s[i0] + window)
    return float(np.max(np.asarray(costs, float)[sel])) if sel.any() else 0.0


def compute_command(path, pose, speed, prm=None, reverse=False, costs=None):
    """
    path: [(x, y), ...] sabit cercevede (odom); pose: (x, y, yaw) ayni cercevede; speed: m/s.
    reverse: yol RECOVERY yolu (is_recovery_path), geri vitesle izlenir.
    Dondurur: (v, w, info) info = {'lookahead', 'curvature', 'target', 'mode'}.
    """
    k = dict(DEFAULTS, **(prm or {}))
    rx, ry, ryaw = pose
    c, s = math.cos(ryaw), math.sin(ryaw)
    p = resample(path)
    info = {'lookahead': 0.0, 'curvature': 0.0, 'target': None, 'mode': 'stop'}
    if len(p) == 0:
        return 0.0, 0.0, info

    # 1) en yakin nokta (yalnizca yolun basinda)
    n_search = max(1, min(len(p), int(k['closest_search'] / 0.05) + 1))
    d2 = (p[:n_search, 0] - rx) ** 2 + (p[:n_search, 1] - ry) ** 2
    i0 = int(np.argmin(d2))

    # 2-3) egrilik ve lookahead
    kappa = curvature(smooth(p, k['path_smoothing']), i0, k['curvature_window'])
    ld = min(k['lookahead_max'], k['lookahead_min'] + k['lookahead_time'] * abs(speed))
    if kappa > 1e-6:
        ld = min(ld, k['lookahead_curvature_factor'] / kappa)
    ld = max(ld, k['lookahead_min'])
    j = min(len(p) - 1, i0 + max(1, int(round(ld / 0.05))))
    tx, ty = p[j]
    info.update(lookahead=ld, curvature=kappa, target=(float(tx), float(ty)))

    dx, dy = tx - rx, ty - ry
    dist = math.hypot(dx, dy)
    if j == len(p) - 1 and dist < k['goal_tolerance']:
        return 0.0, 0.0, info

    # 6) geri vites: RECOVERY yolu
    if reverse:
        yaw_err = wrap(math.atan2(dy, dx) - ryaw - math.pi)
        info['mode'] = 'reverse'
        return -k['min_v'], clamp(k['kp'] * yaw_err, k['max_w']), info

    # 4) acisal hiz
    yaw_err = wrap(math.atan2(dy, dx) - ryaw)
    w = clamp(k['kp'] * yaw_err, k['max_w'])
    if abs(yaw_err) > k['in_place_yaw_error']:
        info['mode'] = 'rotate'
        return 0.0, w, info

    # 5) dogrusal hiz
    v = k['max_v'] * max(0.0, 1.0 - abs(yaw_err) / k['slowdown_yaw_error'])
    if kappa > 1e-6:
        v = min(v, math.sqrt(k['max_lateral_accel'] / kappa))
    v *= max(0.0, 1.0 - k['turn_slowdown'] * abs(w) / k['max_w'])
    if k['clearance_slowdown'] > 0 and costs is not None:
        # Engele yakinlik: yolun onumuzdeki kismi sisirme bolgesinden geciyorsa yavasla (dar gecit,
        # sik duba). Genis seritte maliyet 0, etkisi yok.
        c = cost_ahead(path, costs, pose, k['clearance_window'], k['closest_search'])
        v = min(v, k['max_v'] * max(0.0, 1.0 - k['clearance_slowdown'] * (c / 99.0) ** k['clearance_power']))
        info['cost_ahead'] = c
    if j == len(p) - 1 and dist < 0.5:              # yolun sonuna yaklasirken yavasla
        v *= dist / 0.5
    info['mode'] = 'forward'
    return max(v, k['min_v']), w, info


def clamp(x, lim):
    return max(-lim, min(lim, x))
