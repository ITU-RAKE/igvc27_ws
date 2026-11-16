"""
Occupancy grid -> maliyet haritasi (inflation). ROS'tan bagimsiz, test edilebilir.

A* araci nokta olarak planlar; engellerin etrafi arac boyutuna gore sisirilir:
  d <= lethal_radius                 -> 100 (oldurucu: arac merkezi burada ise govde engele
                                             kesin carpar)
  lethal_radius < d <= inflation_radius -> 99 ... 1 azalan maliyet (engelden uzak dur)
      profile='linear': dogrusal.
      profile='exp'   : ustel (disbukey): siyahin hemen disinda hizla duser, disarida yavasca
                        sifira iner. Genis gradyanda dar gecitler gereksiz pahali olmaz, genis
                        alanda yolu ortalayan hafif bir egim kalir. 99 * exp(-ln(99) * u),
                        u = (d - lethal) / (inflation - lethal): u=0'da 99, u=1'de 1.
  d > inflation_radius               -> hucrenin kendi degeri (0 bos, -1 bilinmiyor)
d: hucre merkezinden en yakin dolu (100) hucre merkezine uzaklik [m].
Maliyet hucrenin bilinip bilinmemesinden bagimsizdir: bilinen bir engele yakin bilinmeyen hucre de
sisirilir (arac merkezi orada olsa yine carpar).

Varsayilan yaricaplar arac izdusumunden (robot_geometry.inflation_radii):
  lethal    = ic teget yaricap (yari genislik 0.395) + 5 cm pay = 0.445 m
  inflation = cevrel yaricap sqrt((L/2)^2 + (W/2)^2) = 0.618 m
Ornek: 2 m'lik seritte merkezde ~0.76 m bembeyaz koridor kalir, oldurucu bolge araca degmez.
Not: siyah (100) bolge arac MERKEZININ giremeyecegi yerdir; govdeyle karsilastirilmaz.
"""
import math

import cv2
import numpy as np

LETHAL = 100


def inflate(grid, resolution, lethal_radius, inflation_radius, profile='linear'):
    """grid: (H, W) int8 (-1 / 0 / 100). Donus: ayni boyutta int8 maliyet haritasi."""
    if lethal_radius <= 0 and inflation_radius <= 0:
        return grid
    obstacles = grid >= LETHAL
    if not obstacles.any():
        return grid
    # En yakin engele uzaklik (engel hucresinde 0), metre
    d = cv2.distanceTransform((~obstacles).view(np.uint8), cv2.DIST_L2,
                              cv2.DIST_MASK_PRECISE) * resolution
    cost = np.zeros(grid.shape, np.int8)
    outer = max(inflation_radius, lethal_radius)
    if outer > lethal_radius:
        ramp = (d > lethal_radius) & (d <= outer)
        u = (d[ramp] - lethal_radius) / (outer - lethal_radius)
        if profile == 'exp':
            c = np.ceil(99.0 * np.exp(-np.log(99.0) * u))
        else:
            c = np.ceil(99.0 * (1.0 - u))
        cost[ramp] = np.clip(c, 1, 99).astype(np.int8)
    cost[d <= lethal_radius] = LETHAL
    out = grid.copy()
    # Yalnizca sisirme bolgesinde ve maliyeti artiriyorsa yaz. (cost > 0 sart: aksi halde
    # bolge disindaki bilinmeyen -1 hucreler 0 > -1 diye "bos" olurdu.)
    m = (cost > 0) & (cost > out)
    out[m] = cost[m]
    return out


def shadow_mask(occupied, margin_cells=2, n_angles=720):
    """
    Grid merkezinden (arac) bakinca bir engelin ARKASINDA kalan hucreler (bool).

    occupied: (N, N) bool, engeller (lidar); merkez = grid merkezi (base_footprint orijini).
    Her isinda ilk engelden margin_cells sonrasi golge. Kutupsal donusumle (cv2.warpPolar):
    satir = aci, sutun = yaricap; ilk engelden itibaren kumulatif or, geri donusum.
    """
    n = occupied.shape[0]
    if not occupied.any():
        return np.zeros(occupied.shape, bool)
    c = (n / 2.0 - 0.5, n / 2.0 - 0.5)
    r_max = n / 2.0 * math.sqrt(2.0)
    n_r = int(math.ceil(r_max))
    flags = cv2.WARP_POLAR_LINEAR | cv2.INTER_NEAREST
    pol = cv2.warpPolar(occupied.astype(np.uint8) * 255, (n_r, n_angles), c, r_max, flags) > 0
    sh = np.maximum.accumulate(pol, axis=1)
    k = max(0, int(margin_cells))
    if k:
        sh[:, k:] = sh[:, :-k].copy()
        sh[:, :k] = False
    back = cv2.warpPolar(sh.astype(np.uint8) * 255, (n, n), c, r_max,
                         flags | cv2.WARP_INVERSE_MAP)
    return back > 0
