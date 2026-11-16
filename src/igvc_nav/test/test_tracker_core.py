"""tracker_core testleri: dinamik lookahead, viraj/donus yavaslamasi, yerinde donus, geri vites."""
import math

import numpy as np
import pytest

from igvc_nav.tracker_core import compute_command, cost_ahead, curvature, is_recovery_path, resample, smooth


def line(x0, x1, y=0.0, n=50):
    return [(x, y) for x in np.linspace(x0, x1, n)]


def arc(r, ang, n=80):
    """(0,0)'dan +x yonunde baslayip sola donen r yaricapli yay."""
    t = np.linspace(0, ang, n)
    return list(zip(r * np.sin(t), r * (1 - np.cos(t))))


def test_curvature_of_arc():
    p = resample(arc(0.5, math.pi))
    assert curvature(p, 0, 1.2) == pytest.approx(2.0, rel=0.1)
    assert curvature(resample(line(0, 3)), 0, 1.2) < 1e-6


def test_straight_full_speed_long_lookahead():
    v, w, info = compute_command(line(0, 5), (0, 0, 0), speed=1.0)
    assert v == pytest.approx(1.0) and abs(w) < 1e-6
    assert info['lookahead'] == pytest.approx(0.95)            # 0.35 + 0.6 * 1.0


def test_tight_turn_shortens_lookahead_and_slows():
    v, w, info = compute_command(arc(0.5, math.pi), (0, 0, 0), speed=1.0)
    assert info['lookahead'] <= 0.36                           # 0.5 / 2.0 -> alt sinir 0.35
    assert v <= math.sqrt(0.5 / 1.8) + 1e-6                    # viraj hiz siniri ~0.5 m/s
    assert w > 0


def test_turning_slows_down():
    """Ayni yon hatasinda acisal hiz buyudukce dogrusal hiz duser (turn_slowdown)."""
    path = [(0, 0), (1.0, 0.35), (3.0, 1.05)]
    v_slow, w, _ = compute_command(path, (0, 0, 0), speed=0.5)
    v_free, _, _ = compute_command(path, (0, 0, 0), speed=0.5, prm={'turn_slowdown': 0.0})
    assert w > 0.2 and v_slow < v_free


def test_rotate_in_place_when_target_behind_but_path_forward():
    """Yol ileride ama arac 120 deg yanlis bakiyor: geri gitmez, yerinde doner."""
    v, w, info = compute_command(line(0, 3), (0, 0, math.radians(120)), speed=0.0)
    assert info['mode'] == 'rotate' and v == 0.0 and w < 0


def test_recovery_path_reverses():
    path = [(0, 0), (-0.5, 0), (-1.0, 0)]
    assert is_recovery_path(path) and not is_recovery_path(line(0, 3))
    # arka yaridan cizilmis izgara yolu (iz hafizasi): geri vites degil, yerinde donup ileri
    assert not is_recovery_path([(-0.05 * i, 0.0) for i in range(60)])
    v, w, info = compute_command(path, (0, 0, 0), speed=0.0, reverse=True)
    assert info['mode'] == 'reverse' and v < 0


def test_closest_point_does_not_jump_to_later_part_of_s_path():
    """S yolunun ileri kismi aracin yanindan geciyor: en yakin nokta yolun basinda kalir."""
    path = line(0, 1.0) + [(1.0, y) for y in np.linspace(0, 0.6, 10)] + line(1.0, -0.2, 0.6)
    _, _, info = compute_command(path, (0.0, 0.0, 0.0), speed=0.3)
    assert info['target'][1] < 0.3                              # hedef ilk kolda (y ~ 0)


def test_stops_at_end():
    v, w, _ = compute_command([(0, 0), (0.05, 0)], (0.0, 0.0, 0.0), speed=0.0)
    assert v == 0.0 and w == 0.0


def test_grid_staircase_does_not_cap_speed():
    """8 yonlu izgara yolu (duz + 45 deg kirilimlar): yumusatmayla viraj hiz siniri devreye girmez."""
    cells = [(i, 0) for i in range(10)] + [(10 + i, i) for i in range(6)] + [(16 + i, 6) for i in range(20)]
    path = [(x * 0.05, y * 0.05) for x, y in cells]
    p = resample(path)
    raw = curvature(p, 0, 1.2)
    assert raw > 3.0                                            # ham: sahte keskin viraj
    assert curvature(smooth(p, 0.25), 0, 1.2) < raw / 2
    v, _, _ = compute_command(path, (0, 0, 0), speed=1.0)
    v_raw, _, _ = compute_command(path, (0, 0, 0), speed=1.0, prm={'path_smoothing': 0.0})
    assert v > v_raw


def test_clearance_slowdown():
    """Onumuzdeki yol sisirme bolgesinden geciyorsa yavasla; kapaliyken (0) etkisiz."""
    path = line(0, 5)
    costs = [0] * 20 + [90] * 10 + [0] * 20                  # ~2-3 m ileride engele yakin bolum
    assert cost_ahead(path, costs, (1.5, 0, 0), 1.5) == 90
    assert cost_ahead(path, costs, (0.0, 0, 0), 1.0) == 0
    prm = {'max_v': 1.5, 'clearance_slowdown': 0.6}
    v_near, _, _ = compute_command(path, (1.5, 0, 0), 1.0, prm, costs=costs)
    v_off, _, _ = compute_command(path, (1.5, 0, 0), 1.0, dict(prm, clearance_slowdown=0.0), costs=costs)
    assert v_near == pytest.approx(1.5 * (1 - 0.6 * 90 / 99), rel=1e-3) and v_off > v_near


# ---------------------------------------------------------------- standart mod
from igvc_nav.tracker_core import compute_command_standard  # noqa: E402


def test_standard_fixed_lookahead_and_speed():
    v, w, info = compute_command_standard(line(0, 5), (0, 0, 0), 0.0, {'max_v': 1.0})
    assert info['lookahead'] == pytest.approx(0.8) and v == pytest.approx(1.0) and abs(w) < 1e-6
    # viraj freni yok: dar yayda da hiz yalnizca yon hatasiyla duser
    v2, w2, _ = compute_command_standard(arc(1.0, math.pi / 2), (0, 0, 0), 1.0, {'max_v': 1.0})
    assert w2 > 0 and v2 > 0.5


def test_standard_dynamic_lookahead():
    prm = {'dynamic_lookahead': True, 'lookahead_min': 0.5, 'lookahead_time': 0.5, 'lookahead_max': 1.2}
    assert compute_command_standard(line(0, 5), (0, 0, 0), 0.0, prm)[2]['lookahead'] == pytest.approx(0.5)
    assert compute_command_standard(line(0, 5), (0, 0, 0), 2.0, prm)[2]['lookahead'] == pytest.approx(1.2)


def test_standard_reverse_only_for_recovery():
    back = [(0, 0), (-0.5, 0), (-1.0, 0)]
    v, _, info = compute_command_standard(back, (0, 0, 0), 0.0, reverse=True)
    assert v < 0 and info['mode'] == 'reverse'
    v, w, info = compute_command_standard(line(0, -3), (0, 0, 0), 0.0, reverse=False)
    assert v > 0 and info['mode'] == 'forward'      # uzun arkadaki yol: geri vites yok
