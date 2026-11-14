"""Inflation (igvc_vision/costmap.py) testleri."""
import numpy as np

from igvc_vision.costmap import inflate

RES = 0.05
LETHAL, INFL = 0.445, 0.618


def _grid(fill=0, n=101):
    return np.full((n, n), fill, np.int8)


def _at(g, dx_m, c=50):
    """Merkezdeki engelden +x yonunde dx_m uzaktaki hucre."""
    return int(g[c, c + int(round(dx_m / RES))])


def test_cost_profile():
    g = _grid()
    g[50, 50] = 100
    out = inflate(g, RES, LETHAL, INFL)
    assert _at(out, 0.0) == 100
    assert _at(out, 0.40) == 100                     # oldurucu bolge icinde
    c45 = _at(out, 0.45)
    assert 90 <= c45 <= 99                           # rampanin basi
    assert _at(out, 0.55) < c45                      # uzaklastikca azalir
    assert 1 <= _at(out, 0.60) <= 15                 # rampanin sonu
    assert _at(out, 0.65) == 0                       # inflation disi: dokunulmaz
    # Simetri: L2 uzaklik (kosegen yonde de ayni yaricap)
    k = int(round(0.4 / RES / np.sqrt(2)))
    assert out[50 + k, 50 + k] == 100


def test_unknown_near_obstacle_is_inflated_far_stays_unknown():
    g = _grid(-1)
    g[50, 50] = 100
    out = inflate(g, RES, LETHAL, INFL)
    assert _at(out, 0.4) == 100                      # bilinmiyor ama engele cok yakin
    assert _at(out, 0.55) > 0
    assert out[0, 0] == -1                           # uzak bilinmeyen bilinmiyor kalir


def test_no_obstacle_or_disabled_returns_input():
    g = _grid()
    g[10:20, 10:20] = -1
    assert np.array_equal(inflate(g, RES, LETHAL, INFL), g)
    g[50, 50] = 100
    assert np.array_equal(inflate(g, RES, 0.0, 0.0), g)


def test_input_costs_below_100_are_not_obstacles():
    """Girdi zaten sisirilmis (1..99) ise yeniden engel kaynagi olmamali (cift sisirme yok)."""
    g = _grid()
    g[50, 40:61] = 50
    assert np.array_equal(inflate(g, RES, LETHAL, INFL), g)
