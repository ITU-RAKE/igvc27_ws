"""
Waypoint modu birim testleri (igvc_nav/waypoints.py, frontier.plan_to_target).
Calistirma:  colcon test --packages-select igvc_nav && colcon test-result --verbose
"""
import math
import os

import numpy as np
import pytest
import yaml

from igvc_nav import frontier
from igvc_nav import waypoints as W

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.normpath(os.path.join(HERE, '..', 'config', 'waypoints.yaml'))


# ------------------------------------------------------------------ dosya + donusum
def test_config_file_converts_to_sim_positions():
    """config/waypoints.yaml'daki lat/lon, datum'da map'e cevrilince dunya modellerinin konumu (sim_xy)."""
    wf = W.load_waypoints(CFG)
    assert len(wf['waypoints']) == 4 and wf['reach_radius_m'] == 1.0
    pts = W.to_map(wf['waypoints'], *wf['datum'])
    with open(CFG) as f:
        sim = [w['sim_xy'] for w in yaml.safe_load(f)['waypoints']]
    np.testing.assert_allclose(pts, sim, atol=1e-3)


def test_missing_or_bad_file(tmp_path):
    with pytest.raises(W.WaypointFileError, match='bulunamadi'):
        W.load_waypoints(str(tmp_path / 'yok.yaml'))
    bad = tmp_path / 'bad.yaml'
    bad.write_text('waypoints:\n  - name: a\n    latitude: 1.0\n')
    with pytest.raises(W.WaypointFileError, match='latitude/longitude'):
        W.load_waypoints(str(bad))


# ------------------------------------------------------------------ durum makinesi
# Sim yerlesimi: No Man's Land guneye dogru, waypoint'ler x~33, y = 7, 1.6, -1.3
PTS = [(33.06, 7.04), (33.21, 1.64), (33.02, -1.32)]
NAMES = ['giris', 'orta', 'cikis']
SOUTH = -math.pi / 2


def drive(m, path, yaw, t0=0.0, dt=0.1):
    """path boyunca (x, y) noktalari dt araliklarla ver; olaylari topla."""
    ev, t = [], t0
    for x, y in path:
        ev += m.update(x, y, yaw, t)
        t += dt
    return ev, t


def line(a, b, step=0.05):
    n = max(2, int(math.dist(a, b) / step))
    return [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n + 1)]


def test_standby_far_away_no_influence():
    m = W.WaypointManager(PTS, NAMES)
    ev, _ = drive(m, line((0, 0), (0, 20)), math.pi / 2)        # parkurun sol duzlugu, 33 m uzak
    assert m.state == W.STANDBY and m.target() is None and ev == []


def test_activation_requires_ahead_and_approaching():
    m = W.WaypointManager(PTS, NAMES)
    # Yakin (4 m) ama waypoint'e sirtini donmus, duruyor: aktif olmamali
    _, t = drive(m, [(33.06, 11.0)] * 30, math.pi / 2)
    assert m.state == W.STANDBY
    # Ayni yerde guneye donup yaklasinca aktif, sira duz
    ev, _ = drive(m, line((33.06, 11.0), (33.06, 10.0)), SOUTH, t0=t)
    assert m.state == W.ACTIVE and m.order == [0, 1, 2] and m.target() == PTS[0]
    assert ev[0][0] == 'aktif'


def test_reverse_direction_order():
    m = W.WaypointManager(PTS, NAMES)
    drive(m, line((33.0, -6.0), (33.0, -4.5)), math.pi / 2)     # guneyden kuzeye geliyor
    assert m.state == W.ACTIVE and m.order == [2, 1, 0] and m.target() == PTS[2]


def test_full_pass_reaches_all_then_done_with_exit_hint():
    m = W.WaypointManager(PTS, NAMES, exit_hint_m=4.0)
    route = (line((33.06, 12.0), PTS[0]) + line(PTS[0], PTS[1])[1:] + line(PTS[1], PTS[2])[1:]
             + line(PTS[2], (33.0, -3.0))[1:])
    ev, _ = drive(m, route, SOUTH, dt=0.05)
    kinds = [k for k, _ in ev]
    assert kinds.count('varildi') == 3 and kinds[-1] == 'bitti' and m.state == W.DONE
    assert [r for _, _, r in m.log] == ['varildi'] * 3
    hint = m.exit_hint()                                         # son iki waypoint yonu (guney)
    assert hint is not None and hint[1] < -0.99
    drive(m, line((33.0, -3.0), (33.0, -8.0)), SOUTH)            # 5 m sonra ipucu biter
    assert m.exit_hint() is None
    drive(m, line((33.0, -8.0), (33.06, 6.0)), math.pi / 2)      # geri gelse bile tekrar aktif olmaz
    assert m.state == W.DONE


def test_reach_uses_margin():
    m = W.WaypointManager(PTS, NAMES, reach_radius=1.0, reach_margin=0.3)
    _, t = drive(m, line((33.06, 12.0), (33.06, 8.5)), SOUTH)
    assert m.state == W.ACTIVE
    m.update(33.06 + 0.8, 7.04, SOUTH, t + 1.5)                  # 0.8 m: 1.0 icinde ama paydan disari
    assert m.i == 0
    m.update(33.06 + 0.6, 7.04, SOUTH, t + 1.6)
    assert m.i == 1 and m.target() == PTS[1]


def test_missed_waypoint_retries_then_skips():
    m = W.WaypointManager(PTS, NAMES, retries=1)
    drive(m, line((33.06, 12.0), (34.3, 9.0)), SOUTH)            # aktif
    assert m.state == W.ACTIVE
    # 1.25 m yanindan gecip uzaklas: kacirildi -> tekrar
    ev, t = drive(m, line((34.3, 9.0), (34.3, 4.0)), SOUTH, t0=10.0)
    assert ('tekrar' in [k for k, _ in ev]) and m.i == 0 and m.attempts == 1
    # Tekrar denemede yine yanindan gecip uzaklas: atlandi -> sonraki
    ev, _ = drive(m, line((34.3, 4.0), (34.3, 9.0)) + line((34.3, 9.0), (34.3, 4.5)), SOUTH, t0=t)
    assert 'atlandi' in [k for k, _ in ev] and m.target() == PTS[1]
    assert m.log[0][1:] == ('giris', 'atlandi')


def test_timeout_skips():
    m = W.WaypointManager(PTS, NAMES, timeout_s=5.0)
    _, t = drive(m, line((33.06, 12.0), (33.06, 10.0)), SOUTH)
    assert m.state == W.ACTIVE
    ev, _ = drive(m, [(33.06, 10.0)] * 60, SOUTH, t0=t)          # 6 s yerinde (5 s asim)
    assert 'atlandi' in [k for k, _ in ev] and m.target() == PTS[1]


def test_gps_jump_single_sample_not_evaluated():
    m = W.WaypointManager(PTS, NAMES)
    _, t = drive(m, line((33.06, 12.0), (33.06, 10.0)), SOUTH)
    ev = m.update(33.06, 7.04, SOUTH, t + 0.05)                  # 3 m sicrama, waypoint'in tam ustu
    assert ev[0][0] == 'sicrama' and m.i == 0                    # sicrama orneginde varis sayilmaz


def test_gps_jump_relocalization_does_not_freeze():
    """EKF yeniden kilitlenmesi: global poz 2.6 m kayar ve oyle kalir. Yonetici yeni konumu esas
    almali; eskiden her ornek eski referansa gore 'sicrama' sayilip yonetici donuyordu."""
    m = W.WaypointManager(PTS, NAMES)
    _, t = drive(m, line((33.06, 14.0), (33.06, 12.0)), SOUTH)  # BEKLEME, yaklasiyor
    ev, t = drive(m, [(35.66, 12.0)], SOUTH, t0=t)               # 2.6 m yana kayma
    assert [k for k, _ in ev] == ['sicrama']
    ev, t = drive(m, line((35.66, 12.0), (33.06, 7.04)), SOUTH, t0=t)   # yeni konumdan devam
    kinds = [k for k, _ in ev]
    assert 'sicrama' not in kinds and 'aktif' in kinds and 'varildi' in kinds
    assert m.target() == PTS[1]


def test_jump_does_not_cause_false_miss():
    """Waypoint'e yaklasirken poz uzaga ziplarsa (d - d_min buyur) 'kacirildi' sayilmamali."""
    m = W.WaypointManager(PTS, NAMES)
    _, t = drive(m, line((33.06, 12.0), (33.06, 8.5)), SOUTH)    # aktif, d_min ~1.5
    assert m.state == W.ACTIVE
    ev, t = drive(m, [(33.06, 11.2)], SOUTH, t0=t)               # 2.7 m geri ziplama
    ev2, _ = drive(m, line((33.06, 11.2), (33.06, 10.6)), SOUTH, t0=t)
    assert 'tekrar' not in [k for k, _ in ev + ev2] and m.attempts == 0


def test_no_logger_severity_switch_on_one_line():
    """rclpy, ayni kaynak satirindan farkli seviyede log basilirsa ValueError ile dugumu cokertir
    ('Logger severity cannot be changed between calls'). Seviye secen tek satirlik desen yasak."""
    import re
    src = open(os.path.join(HERE, '..', 'igvc_nav', 'path_planner.py')).read()
    bad = re.findall(r'=\s*self\.get_logger\(\)\.\w+\s+if\b', src)
    assert bad == [], bad


# ------------------------------------------------------------------ plan_to_target
def grid(h=80, w=80):
    return np.zeros((h, w), np.int8)


def test_plan_to_reachable_target_exact():
    g = grid()
    goal, path, info = frontier.plan_to_target(g, (10, 40), (60.5, 40.5))
    assert info['exact'] and goal == (60, 40) and path[0] == (10, 40) and path[-1] == (60, 40)


def test_target_behind_obstacle_goes_around():
    g = grid()
    g[30:51, 30:36] = 100                                       # duvar, hedef arkasinda
    goal, path, info = frontier.plan_to_target(g, (10, 40), (60.5, 40.5))
    assert info['exact'] and goal == (60, 40)
    assert all(g[y, x] <= 99 for x, y in path)                  # engelden gecmez
    assert max(abs(y - 40) for _, y in path) >= 10              # duvarin etrafindan dolasir


def test_target_in_unknown_shadow_approaches():
    g = grid()
    g[35:46, 40:44] = 100                                       # duba
    g[35:46, 44:70] = -1                                        # arkasi golge (bilinmiyor)
    goal, path, info = frontier.plan_to_target(g, (10, 40), (55.5, 40.5), approach_coeff=2.0)
    assert not info['exact'] and goal is not None
    d_goal = math.hypot(goal[0] + 0.5 - 55.5, goal[1] + 0.5 - 40.5)
    assert d_goal < math.hypot(10 - 55.5, 40 - 40.5) - 30        # hedefe cok yaklasti
    assert all(g[y, x] != -1 and g[y, x] <= 99 for x, y in path)


def test_target_outside_map():
    goal, path, info = frontier.plan_to_target(grid(), (40, 40), (200.0, 40.5))
    assert not info['exact'] and goal[0] == 79                  # haritanin o yondeki kenari


def test_chaining_continues_through_target():
    g = grid()
    far, info = frontier.plan_to_target(g, (10, 40), (40.5, 40.5), next_target=(70.5, 20.5),
                                        chain_dist=20)[1:]
    assert not info['chained']                                  # 30 hucre uzak: zincir yok
    goal, path, info = frontier.plan_to_target(g, (25, 40), (40.5, 40.5),
                                               next_target=(70.5, 20.5), chain_dist=20)
    assert info['chained'] and info['via'] == (40, 40) and goal == (70, 20)
    assert (40, 40) in path and path[-1] == (70, 20)            # waypoint'in ustunden gecer


def test_start_blocked_without_escape_returns_none():
    g = grid()
    g[38:43, 8:13] = 100
    assert frontier.plan_to_target(g, (10, 40), (60.5, 40.5))[0] is None
    goal, path, info = frontier.plan_to_target(g, (10, 40), (60.5, 40.5), escape_radius=5)
    assert info['escape'] and goal == (60, 40)


def test_clock_reset_does_not_break_activation():
    m = W.WaypointManager(PTS, NAMES)
    drive(m, line((33.06, 14.0), (33.06, 12.0)), SOUTH, t0=100.0)
    ev, _ = drive(m, line((33.06, 11.0), (33.06, 10.0)), SOUTH, t0=0.0)   # saat geri gitti
    assert m.state == W.ACTIVE


# ------------------------------------------------------------------ ikinci tur (rearm)
def one_pass(m, t0=0.0):
    """Kuzeyden girip uc waypoint'i gecen ve No Man's Land'den guneye cikan rota."""
    route = (line((33.06, 12.0), PTS[0]) + line(PTS[0], PTS[1])[1:] + line(PTS[1], PTS[2])[1:]
             + line(PTS[2], (33.0, -8.0))[1:])
    return drive(m, route, SOUTH, t0=t0, dt=0.05)


def loop_back(m, t0):
    """No Man's Land'den uzaklasip parkuru dolasir (guney -> bati -> kuzey -> dogu), girise doner."""
    route = (line((33.0, -8.0), (33.0, -13.0)) + line((33.0, -13.0), (0.0, -13.0))[1:]
             + line((0.0, -13.0), (0.0, 13.0))[1:] + line((0.0, 13.0), (33.06, 13.0))[1:])
    ev, t = [], t0
    for (x0, y0), (x1, y1) in zip(route, route[1:]):
        ev += m.update(x1, y1, math.atan2(y1 - y0, x1 - x0), t)
        t += 0.05
    return ev, t


def test_rearm_off_stays_done_like_before():
    m = W.WaypointManager(PTS, NAMES)                            # varsayilan: rearm kapali
    _, t = one_pass(m)
    assert m.state == W.DONE
    ev, t = loop_back(m, t)
    ev2, _ = one_pass(m, t)
    assert m.state == W.DONE and m.laps == 1 and 'yeniden' not in [k for k, _ in ev + ev2]


def test_rearm_second_lap_activates_again():
    m = W.WaypointManager(PTS, NAMES, rearm=True, rearm_travel_m=30.0, rearm_dist_m=10.0)
    ev, t = one_pass(m)
    assert m.state == W.DONE and m.laps == 1
    ev, t = loop_back(m, t)
    assert [k for k, _ in ev].count('yeniden') == 1 and m.state == W.STANDBY
    assert m.lap_logs and [r for _, _, r in m.lap_logs[0]] == ['varildi'] * 3 and m.log == []
    ev, _ = one_pass(m, t)
    kinds = [k for k, _ in ev]
    assert kinds.count('aktif') == 1 and kinds.count('varildi') == 3 and m.state == W.DONE and m.laps == 2
    assert m.order == [0, 1, 2]                                  # ikinci turda da duz sira


def test_rearm_needs_distance_not_just_travel():
    """Yeterince yol gidilse de waypoint'lerin yakininda kalinirsa (geri donus) yeniden kurulmaz."""
    m = W.WaypointManager(PTS, NAMES, rearm=True, rearm_travel_m=30.0, rearm_dist_m=10.0)
    _, t = one_pass(m)
    back_and_forth = (line((33.0, -8.0), (33.0, -10.0)) + line((33.0, -10.0), (33.0, -8.0))[1:]) * 10  # 40 m, son waypoint'e <= 8.7 m
    ev, _ = drive(m, back_and_forth, SOUTH, t0=t, dt=0.05)
    assert m.state == W.DONE and 'yeniden' not in [k for k, _ in ev]


def test_rearm_respects_max_laps():
    m = W.WaypointManager(PTS, NAMES, rearm=True, max_laps=1)
    _, t = one_pass(m)
    ev, _ = loop_back(m, t)
    assert m.state == W.DONE and 'yeniden' not in [k for k, _ in ev]
    assert 'tur=1/1' in m.status()


def test_reset_clears_laps():
    m = W.WaypointManager(PTS, NAMES, rearm=True)
    _, t = one_pass(m)
    loop_back(m, t)
    m.reset()
    assert m.laps == 0 and m.lap_logs == [] and m.state == W.STANDBY


# ------------------------------------------------------------------ uzaktan yonelme (YAKLASMA)
def test_approach_off_unchanged():
    m = W.WaypointManager(PTS, NAMES)                            # varsayilan kapali
    ev, _ = drive(m, line((33.06, 25.0), (33.06, 12.0)), SOUTH)
    assert m.state == W.STANDBY and ev == [] and m.approach_target() is None


def test_approach_enters_at_20m_then_activates():
    m = W.WaypointManager(PTS, NAMES, approach=True, approach_m=20.0)
    ev, t = drive(m, line((33.06, 30.0), (33.06, 26.5)), SOUTH)   # 19.5 m, giris onde
    assert m.state == W.APPROACH and m.approach_target() == PTS[0] and [k for k, _ in ev] == ['yaklasma']
    ev, _ = drive(m, line((33.06, 26.5), (33.06, 10.0))[1:], SOUTH, t0=t)
    assert m.state == W.ACTIVE and m.order == [0, 1, 2] and 'aktif' in [k for k, _ in ev]


def test_approach_requires_entry_ahead():
    m = W.WaypointManager(PTS, NAMES, approach=True)
    ev, _ = drive(m, line((33.06, 20.0), (33.06, 22.0)), math.pi / 2)   # giris arkada, uzaklasiyor
    assert m.state == W.STANDBY and ev == []


def test_approach_released_out_of_range():
    m = W.WaypointManager(PTS, NAMES, approach=True, approach_m=20.0, approach_hyst_m=3.0)
    _, t = drive(m, line((33.06, 26.0), (33.06, 25.0)), SOUTH)
    assert m.state == W.APPROACH
    ev, _ = drive(m, line((33.06, 25.0), (33.06, 32.0))[1:], SOUTH, t0=t)   # geri gidiyor (yon guneyde kalsa da)
    assert m.state == W.STANDBY and 'yaklasma_iptal' in [k for k, _ in ev]


def test_approach_no_progress_timeout_and_cooldown():
    m = W.WaypointManager(PTS, NAMES, approach=True, approach_timeout_s=20.0, approach_cooldown_s=30.0)
    _, t = drive(m, line((33.06, 22.0), (33.06, 21.0)), SOUTH)
    assert m.state == W.APPROACH
    stuck = [(33.06, 21.0)] * 250                                 # 25 s ayni yerde (duvara yaslanmis)
    ev, t = drive(m, stuck, SOUTH, t0=t)
    assert m.state == W.STANDBY and 'yaklasma_iptal' in [k for k, _ in ev]
    ev, t = drive(m, [(33.06, 21.0)] * 100, SOUTH, t0=t)          # 10 s: bekleme suresinde girilmez
    assert m.state == W.STANDBY and 'yaklasma' not in [k for k, _ in ev]
    ev, _ = drive(m, line((33.06, 21.0), (33.06, 19.0)), SOUTH, t0=t + 25.0)   # sure doldu
    assert m.state == W.APPROACH


def test_approach_reverse_entry_and_rearm_lap():
    m = W.WaypointManager(PTS, NAMES, approach=True, rearm=True)
    ev, t = drive(m, line((33.0, -19.0), (33.0, -3.5)), math.pi / 2)    # guneyden kuzeye
    kinds = [k for k, _ in ev]
    assert kinds[0] == 'yaklasma' and 'aktif' in kinds and m.order == [2, 1, 0]
