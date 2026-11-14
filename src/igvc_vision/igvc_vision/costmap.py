"""
Occupancy grid -> maliyet haritasi (inflation). ROS'tan bagimsiz, test edilebilir.

A* araci nokta olarak planlar; engellerin etrafi arac boyutuna gore sisirilir:
  d <= lethal_radius                 -> 100 (oldurucu: arac merkezi burada ise govde engele
                                             kesin carpar)
  lethal_radius < d <= inflation_radius -> 99 ... 1 dogrusal azalan maliyet (yerinde donerken
                                             kose carpabilir; engelden uzak dur)
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
import cv2
import numpy as np

LETHAL = 100


def inflate(grid, resolution, lethal_radius, inflation_radius):
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
        c = np.ceil(99.0 * (outer - d[ramp]) / (outer - lethal_radius))
        cost[ramp] = np.clip(c, 1, 99).astype(np.int8)
    cost[d <= lethal_radius] = LETHAL
    out = grid.copy()
    # Yalnizca sisirme bolgesinde ve maliyeti artiriyorsa yaz. (cost > 0 sart: aksi halde
    # bolge disindaki bilinmeyen -1 hucreler 0 > -1 diye "bos" olurdu.)
    m = (cost > 0) & (cost > out)
    out[m] = cost[m]
    return out
