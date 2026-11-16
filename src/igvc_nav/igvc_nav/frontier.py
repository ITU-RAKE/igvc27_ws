"""
Frontier tabanli hedef secimi + yol (path_planner goal_mode='frontier').

Sorun (eski BFS hedef secimi): skor yalnizca "ne kadar ileri" diye bakar. Sonu kapali bir cep
(orn. agzi acik duba cebi) ileri eksende en uzak bos hucreyi verdigi icin hedef olur; arac cebe
girer, RECOVERY ile geri cikar, ayni hedefi yine secer ve ileri geri yapar. Arama derinligini
artirmak cozmez: daha derin arama da cebin ucunu bulur, gercek cikis daha uzakta ve yanda kalir.

Cozum: hedef yalnizca FRONTIER hucrelerinden secilir: robottan ulasilabilen, gecilebilir ve
"acik bilinmeyen" alana (sensor menzilinin otesi, harita kenari) komsu hucre. Kapali bir cebin
icinde frontier yoktur (her yani engel), yani cep hedef olamaz; devam eden yolun sonu ise
menzil sinirina ulastigi icin her zaman frontier'dir. Hafiza gerekmez: her haritada yeniden.

Yalnizca haritanin UST YARISI ("ileri" yonunde robotun onu; varsayilan aracin burnu, planner
son metrelerde gidilen yonu verir) kullanilir: arama da hedef
de oradadir. Boylece yol robotun arkasindan dolasip seridin disindan ileri cikamaz.

Adimlar (hepsi vektorel / C, 240x240 haritada ~20-40 ms):
  1) Ust yarinin gecilebilir hucreleri uzerinde 8 komsulu, maliyet agirlikli Dijkstra (scipy.csgraph),
     hucre maliyeti A* ile ayni: adim * (1 + deger * path_find_cost_coeff / 100).
  2) Frontier: ulasilabilir gecilebilir hucre, buyuk bir bilinmeyen bolgeye ya da harita
     kenarina komsu. Kucuk bilinmeyen adacıklar (gurultu) sayilmaz; robota min_goal_dist'ten
     yakin hucreler sayilmaz.
  3) Skor (hucre birimi, eski skorla ayni katsayilar):
        ileri * x_coeff - |yanal| * y_coeff - deger * cost_coeff - (jeodezik - kus ucusu) * detour_coeff
     Dolambac cezasi, serit cizgisindeki bir bosluktan seridin disina dolasan yollari eler.
  4) Ust yarida harita sinirina ulasan yol yoksa (frontier yok, orn. dubalarin sisirmesi seridi
     kapatiyor): ust yarida cizilebilen EN UZUN yolun ucu hedef (yol uzunlugu, maliyetsiz;
     mumkunse maliyeti fallback_max_cost'u gecmeyen hucreler). O hedefe giden yol yine maliyetli
     Dijkstra'dan: engellerden uzak, ortalanmis. Arac oraya kadar ilerler, harita ilerledikce
     yenilenir.
  5) Yol: Dijkstra'nin onculeri (A* gerekmez, hedefe her zaman ulasir).
"""
import math

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def plan(data, start, *, cost_threshold=99, cost_coeff=8.5, x_coeff=2.0, y_coeff=1.1,
         goal_cost_coeff=5.0, detour_coeff=1.0, min_goal_dist=20,
         min_unknown_area=100, fallback_max_cost=50, use_frontier=True, frontier_radius=0,
         forward=(1.0, 0.0), escape_radius=0, trail_back=None, max_backtrack=np.inf):
    """
    data: HxW int8 occupancy (data[y, x]; -1 bilinmiyor, 0..100 maliyet); +x = ileri.
    start: (x, y) hucre. Mesafeler hucre cinsinden.
    Dondurur: (goal, path, info) ya da aday yoksa (None, None, info).
      path: [(x, y), ...] start -> goal.
      info: {'frontier': HxW bool, 'candidates': int, 'fallback': bool}
        fallback = True: frontier yok, en uzun yolun ucu secildi.
    use_frontier=False: frontier hic aranmaz, her zaman en uzun yolun ucu (adim 4).
    forward: "ileri" birim vektoru (grid x, y). Varsayilan aracin burnu (+x). Planner son
      metrelerde gidilen yonu verir: arac yerinde donse de ust yari parkurun ilerisinde kalir.
    escape_radius (hucre, 0 = kapali: eski davranis, None -> planner RECOVERY): robotun hucresi gecilemezse (oldurucu bolge: baslangic cizgisinin
      ustunde, dar gecitte sisirmeye kaymis) ust yaridaki en yakin gecilebilir hucreden
      planlanir, yolun basina oraya duz bir parca eklenir. Yoksa None (planner RECOVERY).
    frontier_radius > 0 (hucre; planner plan_radius verir): frontier yalnizca robottan bu
      uzakliktaki halka (plan dairesinin kenari). Ic bolgedeki bilinmeyenler (duba arkasi
      golgeler, gorulmeyen yerler) frontier sayilmaz, yalnizca gecilemez.
    trail_back (HxW float, hucre; None = kapali): aracin geldigi izin koridoru. Koridordaki hucrede
      izin sonundan (aracin en son yeni zemine ciktigi yer) o noktaya kadar iz boyunca geri mesafe,
      koridor disinda inf. YOL MALIYETINI DEGISTIRMEZ (yol izin ustunden gecebilir, dar alan
      manevrasi etkilenmez); yalnizca HEDEF secimi:
        1) iz disindaki frontier adaylari (normal skor); izdeki halka hucreleri elenir,
        2) yoksa en uzun yolun ucu (eski davranis; yalnizca izde max_backtrack'ten geride kalan
           hucreler elenir, dar alan manevrasi iz kapaliyla ayni), hicbiri yoksa None + blocked.
      info['trail'] = elenen frontier sayisi, info['backtrack'] = hedef eski izin uzerinde,
      info['backtrack_blocked'] = gidilebilir her yer max_backtrack'ten geride (planner durur).
    """
    h, w = data.shape
    sx, sy = start
    ys, xs = np.mgrid[0:h, 0:w]
    ux, uy = forward
    norm = math.hypot(ux, uy) or 1.0
    ux, uy = ux / norm, uy / norm
    along = (xs - sx) * ux + (ys - sy) * uy                     # ileri bileşen (hucre)
    across = -(xs - sx) * uy + (ys - sy) * ux                   # yanal bileşen
    passable = (data != -1) & (data <= cost_threshold) & (along >= 0)   # yalnizca ust yari
    info = {'frontier': np.zeros((h, w), bool), 'candidates': 0, 'fallback': False, 'escape': False,
            'trail': 0, 'backtrack': False, 'backtrack_blocked': False}
    on_trail = np.isfinite(trail_back) if trail_back is not None else np.zeros((h, w), bool)
    lead = []                                   # kacis: robot -> ilk gecilebilir hucre
    if not passable[sy, sx]:
        r = escape_radius
        y0, y1, x0, x1 = max(0, sy - r), min(h, sy + r + 1), max(0, sx - r), min(w, sx + r + 1)
        win = passable[y0:y1, x0:x1] & (np.hypot(xs[y0:y1, x0:x1] - sx, ys[y0:y1, x0:x1] - sy) <= r)
        if not win.any():
            return None, None, info
        wy, wx = np.nonzero(win)
        k = int(np.argmin((wx + x0 - sx) ** 2 + (wy + y0 - sy) ** 2))
        ex, ey = int(wx[k] + x0), int(wy[k] + y0)
        n_lead = max(abs(ex - sx), abs(ey - sy))
        lead = [(int(round(sx + (ex - sx) * t / n_lead)), int(round(sy + (ey - sy) * t / n_lead)))
                for t in range(n_lead)]
        info['escape'] = True
        sx, sy = ex, ey

    # --- 1) Dijkstra -------------------------------------------------------------------------
    idx = np.full((h, w), -1, np.int64)
    n = int(passable.sum())
    idx[passable] = np.arange(n)
    step_cost = 1.0 + np.where(data < 0, 0, data).astype(np.float64) * cost_coeff / 100.0
    rows, cols, wts, steps = [], [], [], []
    for dx, dy in NEIGHBORS:
        # kaynak [y, x] -> hedef [y+dy, x+dx]
        ys0, ys1 = max(0, -dy), h - max(0, dy)
        xs0, xs1 = max(0, -dx), w - max(0, dx)
        src = passable[ys0:ys1, xs0:xs1]
        dst = passable[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        m = src & dst
        if dx and dy:
            # kose kesme yok: iki dik komsu da gecilebilir olmali
            m &= passable[ys0 + dy:ys1 + dy, xs0:xs1] & passable[ys0:ys1, xs0 + dx:xs1 + dx]
        rows.append(idx[ys0:ys1, xs0:xs1][m])
        cols.append(idx[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx][m])
        wts.append(step_cost[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx][m] * math.hypot(dx, dy))
        steps.append(np.full(int(m.sum()), math.hypot(dx, dy)))
    graph = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n, n))
    s = int(idx[sy, sx])
    dist, pred = dijkstra(graph, indices=s, return_predecessors=True)
    reach = np.zeros((h, w), bool)
    reach[passable] = np.isfinite(dist)
    geo = np.full((h, w), np.inf)
    geo[passable] = dist

    # --- 2) Frontier -------------------------------------------------------------------------
    unknown = (data == -1).astype(np.uint8)
    k, lab, stats, _ = cv2.connectedComponentsWithStats(unknown, connectivity=8)
    open_unknown = np.zeros((h, w), bool)
    if k > 1:
        big = np.zeros(k, bool)
        big[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_unknown_area
        # harita kenarina degen bilinmeyen bolge her zaman "acik" (menzil disi)
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        big[border[border > 0]] = True
        open_unknown = big[lab]
    near_open = cv2.dilate(open_unknown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    near_open[0, :] = near_open[-1, :] = near_open[:, 0] = near_open[:, -1] = True  # harita kenari
    fx, fy = xs - sx, ys - sy
    euclid = np.hypot(fx, fy)
    if frontier_radius > 0:
        near_open = euclid >= frontier_radius - 2.0
    frontier = reach & near_open & (euclid >= min_goal_dist)
    if not use_frontier:
        frontier[:] = False
    info['frontier'] = frontier
    info['trail'] = int((frontier & on_trail).sum())

    cand = np.flatnonzero(frontier & ~on_trail)
    info['candidates'] = int(cand.size)
    if cand.size:
        # --- 3) Skor --------------------------------------------------------------------------
        fxc, fyc = along.ravel()[cand], across.ravel()[cand]
        score = (fxc * x_coeff - np.abs(fyc) * y_coeff
                 - data.ravel()[cand].astype(np.float64) * goal_cost_coeff
                 - (geo.ravel()[cand] - euclid.ravel()[cand]) * detour_coeff)
        best = cand[int(np.argmax(score))]
    else:
        # --- 4) En uzun yol: ayni grafik, agirlik = yalnizca adim uzunlugu ----------------------
        info['fallback'] = True
        length_graph = csr_matrix(
            (np.concatenate(steps), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        d2 = dijkstra(length_graph, indices=s)       # yol yine maliyetli Dijkstra'dan (pred)
        plen = np.full((h, w), -1.0)
        plen[passable] = np.where(np.isfinite(d2), d2, -1.0)
        reach_far = (plen >= 0) & (euclid >= min_goal_dist)
        # Iz burada yalnizca max_backtrack'ten geride kalan hucreleri eler. Dar alanda (sik duba
        # slalomu) yol S cizip katlanir: birkac metre onceki iz ileriki cikisa 1-2 m yakin kalir;
        # yakin izi elemek cikisi kapatip araci dubalara sikistiriyordu (iz kapaliyla ayni kalmali).
        far_back = on_trail & (trail_back > max_backtrack) if trail_back is not None else on_trail
        ok = reach_far & ~far_back
        if not ok.any():
            # gidilebilir yer var ama hepsi izde max_backtrack'ten geride: planner durur
            info['backtrack_blocked'] = bool((reach_far & on_trail).any())
            return None, None, info
        cheap = ok & (data <= fallback_max_cost)
        pick = cheap if cheap.any() else ok
        best = int(np.argmax(np.where(pick, plen, -1.0)))
        info['backtrack'] = bool(on_trail.ravel()[best])    # hedef eski izde: geri gidis
    gy, gx = divmod(int(best), w)

    # --- 5) Yol (Dijkstra onculeri) ------------------------------------------------------------
    cell_y, cell_x = np.nonzero(passable)          # idx sirasiyla ayni (satir oncelikli)
    path = []
    node = int(idx[gy, gx])
    while node >= 0:
        path.append((int(cell_x[node]), int(cell_y[node])))
        if node == s:
            break
        node = int(pred[node])
    path.reverse()
    return (gx, gy), lead + path, info
