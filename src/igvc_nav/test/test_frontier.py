"""frontier.plan testleri: sonu kapali cep hedef olmaz, cikisa gider; cepten kacis; hiz."""
import time

import cv2
import numpy as np

from igvc_nav.frontier import plan

N, C, R = 240, 120, 98          # 12 m harita, 5 cm; robot merkezde; gorus yaricapi 4.9 m


def world(walls, inflate=8):
    """walls: [(x0, y0, x1, y1)] dolu dikdortgenler. Engel 100, sisirme 99 -> 1, menzil disi -1."""
    occ = np.zeros((N, N), np.uint8)
    for x0, y0, x1, y1 in walls:
        occ[y0:y1 + 1, x0:x1 + 1] = 1
    d = cv2.distanceTransform((1 - occ).astype(np.uint8), cv2.DIST_L2, 5)
    g = np.where(d <= inflate, 100, np.clip(99 * np.exp(-(d - inflate) / 6), 0, 99)).astype(np.int8)
    g[occ == 1] = 100
    ys, xs = np.mgrid[0:N, 0:N]
    g[np.hypot(xs - C, ys - C) > R] = -1
    return g


# Serit: sag duvar y=90 (x 40..170'te biter -> saga cikis), sol duvar y=150.
# Cep: cikisin hemen ilerisinde, agzi robota bakan, sonu kapali (x=212 duvar, y=110 bolme).
POCKET = [(40, 150, 230, 152), (40, 88, 170, 90), (180, 108, 212, 110), (210, 108, 212, 152)]


def test_dead_end_pocket_is_not_goal():
    g = world(POCKET)
    goal, path, info = plan(g, (C, C))
    assert goal is not None and not info['fallback']
    gx, gy = goal
    assert not (180 <= gx <= 212 and 110 <= gy <= 152), f'hedef cebin icinde: {goal}'
    assert gy < 100, f'hedef saga cikista olmali: {goal}'
    assert path[0] == (C, C) and path[-1] == goal
    assert all(g[y, x] <= 99 for x, y in path)


def test_never_goes_behind_robot():
    """Frontier yalnizca robotun arkasindan dolasarak ulasilabilir (serit disi): kullanilmaz."""
    g = world([(100, 140, 200, 142), (100, 98, 200, 100), (150, 98, 152, 142)])
    goal, path, info = plan(g, (115, C))
    assert all(x >= 115 for x, _ in path)
    assert info['fallback'] and goal[0] > 115
    # duvarin dibinde (1 m'den uzak gidilebilir hucre yok): None -> planner eski BFS/RECOVERY'ye duser
    assert plan(g, (138, C))[0] is None


def test_blocked_lane_goes_to_end_of_longest_path():
    """
    Viraj: serit saga donuyor ama dubalarin sisirmesi ileride seridi tamamen kapatiyor.
    Ust yarida sinira ulasan yol yok -> seritte cizilebilen en uzun yolun ucu (viraji takip eder).
    """
    walls = [(0, 150, 202, 152), (0, 88, 160, 90),          # serit, x=160'tan sonra saga aciliyor
             (160, 60, 162, 90), (200, 60, 202, 152),        # saga (-y) donen kol
             (160, 60, 202, 62)]                             # dubalar kolun sonunu kapatiyor
    g = world(walls)
    goal, path, info = plan(g, (C, C))
    assert info['fallback']
    gx, gy = goal
    assert 162 < gx < 200 and gy < 90, f'hedef saga donen kolun sonunda olmali: {goal}'
    assert all(x >= C for x, _ in path)


def test_open_lane_goes_straight():
    g = world([(0, 150, N - 1, 152), (0, 88, N - 1, 90)])
    goal, _, _ = plan(g, (C, C))
    assert goal[0] > C + 80 and abs(goal[1] - C) < 15


def test_small_unknown_holes_ignored():
    """Ileride bilinmeyen kucuk bir delik (gurultu) frontier sayilmaz: hedef uzak ucta."""
    g = world([(0, 150, N - 1, 152), (0, 88, N - 1, 90)])
    g[118:122, 160:164] = -1
    goal, _, info = plan(g, (C, C))
    assert goal[0] > C + 80
    assert not info['frontier'][118:122, 158:166].any()


def test_fast_enough():
    g = world(POCKET)
    plan(g, (C, C))
    t = time.perf_counter()
    for _ in range(5):
        plan(g, (C, C))
    assert (time.perf_counter() - t) / 5 < 0.08


def test_use_frontier_false_always_longest_path():
    """Frontier kapali: cikis gorunse de hedef en uzun yolun ucu (frontier hic aranmaz)."""
    g = world(POCKET)
    goal_f, _, info_f = plan(g, (C, C))
    goal, path, info = plan(g, (C, C), use_frontier=False)
    assert not info_f['fallback'] and info['fallback']
    assert not info['frontier'].any() and goal != goal_f
    assert path[0] == (C, C) and path[-1] == goal


def test_frontier_ring_ignores_inner_unknown():
    """Halka modu: ic bolgedeki buyuk bilinmeyen (duba golgesi) frontier degil, gecilemez."""
    g = world([(0, 150, N - 1, 152), (0, 88, N - 1, 90)])
    g[95:145, 175:200] = -1                                  # seridin icinde golge
    goal, path, info = plan(g, (C, C), frontier_radius=R)
    assert not info['frontier'][95:145, 170:205].any()
    assert all(g[y, x] != -1 for x, y in path)
    # eski mod: golge kenari frontier olurdu
    assert plan(g, (C, C))[2]['frontier'][95:145, 170:205].any()


def test_forward_vector_defines_upper_half():
    """Arac seride dik (burnu duvara bakiyor) ama gidilen yon +x: ust yari +x'e gore, hedef ilerde."""
    g = world([(0, 150, N - 1, 152), (0, 88, N - 1, 90)])
    goal_nose, _, _ = plan(g, (C, C), forward=(0.0, 1.0))          # burun duvara (+y)
    goal, path, _ = plan(g, (C, C), forward=(1.0, 0.0))            # gidilen yon
    assert goal[0] > C + 80
    assert all(x >= C for x, _ in path)
    assert goal_nose[1] >= C                                        # burun yonunde ust yari: +y tarafi


def test_escape_when_start_in_lethal():
    """Robot baslangic cizgisinin ustunde (hucresi oldurucu): RECOVERY yerine yakindaki bos hucreden planla."""
    g = world([(0, 150, N - 1, 152), (0, 88, N - 1, 90), (C - 1, 92, C + 1, 148)])   # robotun altindan gecen cizgi
    assert g[C, C] == 100
    assert plan(g, (C, C))[0] is None                                  # varsayilan: kapali
    goal, path, info = plan(g, (C, C), escape_radius=12)
    assert info['escape'] and goal[0] > C + 80
    assert path[0] == (C, C) and all(g[y, x] <= 99 for x, y in path[len(path) // 4:])
    # gercekten sikismis (cevrede bos hucre yok): None
    g2 = np.full((N, N), 100, np.int8)
    assert plan(g2, (C, C), escape_radius=12)[0] is None


# ---------------------------------------------------------------- iz hafizasi (trail_back)
LANE = [(0, 150, N - 1, 152), (0, 88, N - 1, 90)]


def trail_corridor(x0, x1, end_x, radius=24):
    """Serit ortasindan (y=C) x0..x1 boyunca gelinmis iz; geri mesafe = end_x - x (hucre)."""
    tb = np.full((N, N), np.inf)
    ys, xs = np.mgrid[0:N, 0:N]
    m = (xs >= x0) & (xs <= x1) & (np.abs(ys - C) <= radius)
    tb[m] = (end_x - xs[m]).astype(float)
    return tb


def test_trail_none_is_unchanged():
    g = world(POCKET)
    a = plan(g, (C, C))
    b = plan(g, (C, C), trail_back=None)
    assert a[0] == b[0] and a[1] == b[1]


def test_turned_around_does_not_pick_trail_goal():
    """Arac geldigi yola donmus (burun -x). Arkasi (+x) acik ama burun yarisinda her sey eski iz:
    iz disi hedef yok -> en az geri gidis (backtrack); planner (burun gidis yonune ters) arka
    yariya bakar ve oradaki yeni zemini secer."""
    g = world(LANE)
    tb = trail_corridor(0, C, C)
    goal, _, info = plan(g, (C, C), forward=(-1.0, 0.0), trail_back=tb)
    assert info['backtrack'] and info['trail'] > 0
    goal2, _, info2 = plan(g, (C, C), forward=(1.0, 0.0), trail_back=tb)
    assert not info2['backtrack'] and goal2[0] > C + 80


def test_trail_excludes_back_frontier_prefers_side_exit():
    """Cepte donmus arac: onde (burun -x) geldigi yol (iz) ve yanda yeni bir kol (asagi, -y) var.
    Iz olmadan kus ucusu en ileri olan geldigi yol secilir; izle yan kol secilir."""
    walls = [(0, 150, N - 1, 152),                 # sol duvar
             (0, 88, 60, 90), (100, 88, N - 1, 90),  # sag duvarda x 60..100 arasi kol agzi
             (60, 0, 62, 90), (98, 0, 100, 90)]     # kol -y yonune gidiyor
    g = world(walls)
    start = (85, C)
    no_trail, _, _ = plan(g, start, forward=(-1.0, 0.0), y_coeff=0.3)
    assert no_trail[0] < 60, f'izsiz: geldigi yola gitmeli (testin on kosulu): {no_trail}'
    tb = trail_corridor(0, 70, 85)
    goal, path, info = plan(g, start, forward=(-1.0, 0.0), y_coeff=0.3, trail_back=tb)
    assert info['trail'] > 0 and not info['backtrack']
    assert 62 < goal[0] < 98 and goal[1] < 88, f'yan kola gitmeli: {goal}'


def test_dead_end_backtracks_along_trail():
    """Gercek cikmaz (onde duvar), geri tek yol izin kendisi: hedef izde (geri gidis), engellenmez."""
    g = world(LANE + [(170, 88, 172, 152)])
    tb = trail_corridor(0, 150, 150)
    goal, path, info = plan(g, (150, C), forward=(-1.0, 0.0), trail_back=tb)
    assert info['backtrack'] and np.isfinite(tb[goal[1], goal[0]]) and path[-1] == goal


def test_backtrack_limit_blocks():
    g = world(LANE + [(170, 88, 172, 152)])
    tb = trail_corridor(0, 150, 200)               # her iz hucresi >= 50 hucre geride
    goal, _, info = plan(g, (150, C), forward=(-1.0, 0.0), trail_back=tb, max_backtrack=40)
    assert goal is None and info['backtrack_blocked']


def test_fallback_ignores_recent_trail_fold():
    """Dar alan: frontier yok (serit kapali), iz yakinda katlaniyor. En uzun yol hedefi iz
    kapaliyla ayni olmali (yakin iz fallback'i etkilemez)."""
    walls = [(0, 150, 202, 152), (0, 88, 160, 90), (160, 60, 162, 90), (200, 60, 202, 152),
             (160, 60, 202, 62)]
    g = world(walls)
    tb = np.full((N, N), np.inf)
    tb[60:150, 150:200] = 60.0                     # katlanan iz, 3 m geride
    assert plan(g, (C, C))[0] == plan(g, (C, C), trail_back=tb)[0]


# ---------------------------------------------------------------- standart nav modu (adim 6)
STD = dict(standard_depth=70, route_slack=20, frontier_radius=R - 2)


def lane_with_barrel(bx):
    """3 m serit (y 90..150), ortada bx hucre ileride duba."""
    return world([(0, 88, 239, 90), (0, 150, 239, 152), (bx - 6, C - 5, bx + 6, C + 7)])


def lateral_at(path, x):
    return next(abs(py - C) for px, py in path if px >= x)


def test_standard_off_is_unchanged():
    g = lane_with_barrel(C + 60)
    a = plan(g, (C, C), frontier_radius=R - 2)
    b = plan(g, (C, C), frontier_radius=R - 2, standard_depth=0)
    assert a[0] == b[0] and a[1] == b[1] and not b[2]['standard']


def test_standard_goal_used_on_route():
    g = lane_with_barrel(C + 60)                  # duba 3 m onde
    goal0, path0, _ = plan(g, (C, C), frontier_radius=R - 2)
    goal, path, info = plan(g, (C, C), **STD)
    assert info['standard'] and info['route_goal'] == goal0
    assert goal[0] < goal0[0]                     # standart hedef daha yakin (BFS derinligi)
    assert g[goal[1], goal[0]] <= 99 and path[-1] == goal


def test_standard_goal_in_pocket_is_rejected():
    """Duz ileride sonu kapali cep, gercek yol sola kol: dogrulamasiz standart BFS cebe gider."""
    walls = [(0, 88, 216, 90), (214, 88, 216, 152),            # sag cizgi + 4.7 m'de duvar (cep)
             (0, 150, 135, 152),                                # sol cizgi 0.75 m'de biter
             (135, 150, 137, 239), (160, 150, 162, 239), (160, 150, 216, 152)]   # sola kol
    g = world(walls)
    goal0, _, _ = plan(g, (C, C), frontier_radius=R - 2)
    goal, _, info = plan(g, (C, C), **STD)
    assert goal0[1] > 150, f'frontier hedefi sol kolda olmali: {goal0}'
    assert not info['standard'] and goal == goal0     # dogrulamasiz standart BFS (185, 119): cepte


def test_standard_not_used_in_fallback():
    g = world(POCKET, inflate=30)                 # sisirme seridi kapatir -> fallback
    goal0, path0, i0 = plan(g, (C, C))
    goal, path, info = plan(g, (C, C), standard_depth=70)
    if i0['fallback']:
        assert not info['standard'] and goal == goal0


# ---------------------------------------------------------------- waypoint YAKLASMA: egilim + acik alan
from igvc_nav.frontier import open_arc_deg, ray_clear  # noqa: E402


def fork_world():
    """Serit 3 m ileride ikiye catallaniyor (sol ve sag kol), ortada bolme."""
    return world([(0, 88, 180, 90), (0, 150, 180, 152),              # serit
                  (180, 30, 182, 88), (180, 152, 182, 210),           # catal agzi duvarlari yok: acik
                  (185, 115, 239, 125)])                              # ortadaki bolme


def test_bias_none_is_unchanged():
    g = fork_world()
    a = plan(g, (C, C), frontier_radius=R - 2)
    b = plan(g, (C, C), frontier_radius=R - 2, bias_target=None, bias_coeff=1.0)
    c = plan(g, (C, C), frontier_radius=R - 2, bias_target=(C + 150, C + 80), bias_coeff=0.0)
    assert a[0] == b[0] == c[0] and a[1] == b[1] == c[1]


def test_bias_picks_side_toward_target():
    g = fork_world()
    left, _, _ = plan(g, (C, C), frontier_radius=R - 2, bias_target=(C + 150, C + 120), bias_coeff=1.0)
    right, _, _ = plan(g, (C, C), frontier_radius=R - 2, bias_target=(C + 150, C - 120), bias_coeff=1.0)
    assert left[1] > C and right[1] < C                               # hedef hangi yandaysa o kol


def test_open_arc_lane_vs_open_field():
    lane = world([(0, 88, 239, 90), (0, 150, 239, 152)])
    field = world([])
    assert open_arc_deg(lane, (C, C), (1, 0), R - 4) < 60
    assert open_arc_deg(field, (C, C), (1, 0), R - 4) > 170


def test_ray_clear():
    g = world([(150, 100, 156, 140)])                                 # 1.5 m ileride duvar
    assert not ray_clear(g, (C, C), (C + 90, C), 90)
    assert ray_clear(g, (C, C), (C, C + 90), 90)                     # sola dogru bos
