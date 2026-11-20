"""Inflation (igvc_vision/costmap.py) testleri."""
import numpy as np

from igvc_vision.costmap import inflate, shadow_mask

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


def test_exp_profile_convex_and_same_lethal():
    """Ustel profil: siyah ayni, uclar 99 ve 1, ortada dogrusaldan dusuk (disbukey)."""
    g = _grid()
    g[50, 50] = 100
    lin = inflate(g, RES, LETHAL, 1.1)
    exp = inflate(g, RES, LETHAL, 1.1, profile='exp')
    assert np.array_equal(lin == 100, exp == 100)
    near = _at(exp, 0.50)                                   # siyahin hemen disi
    assert 50 < near <= 99
    assert _at(exp, 0.80) < _at(lin, 0.80)                  # ortada daha ucuz
    assert 1 <= _at(exp, 1.05) <= 3 and _at(exp, 1.20) == 0


def test_shadow_behind_obstacle_only():
    """Merkezden bakinca engelin arkasi golge, onu ve yanlari degil."""
    occ = np.zeros((101, 101), bool)
    occ[48:53, 70:73] = True                                  # merkezin +x yonunde 1 m
    sh = shadow_mask(occ, margin_cells=2)
    assert sh[50, 80] and sh[50, 99]                          # arkasi
    assert not sh[50, 60] and not sh[50, 30]                  # onu, ters yon
    assert not sh[20, 85] and not sh[80, 85]                  # yanlar
    assert not shadow_mask(np.zeros((101, 101), bool)).any()


# ---------------------------------------------------------------- golgede bilinmeyen kalsin
from igvc_vision.costmap import remove_thin_free  # noqa: E402


def test_unknown_cost_false_keeps_unknown_except_lethal():
    g = np.full((40, 40), -1, np.int8)
    g[:, :20] = 0                                    # sol yari bilinen bos, sag yari bilinmiyor (golge)
    g[20, 19] = 100                                  # engel bilinenin kenarinda
    old = inflate(g, 0.05, 0.2, 0.5)                 # eski: bilinmeyen de maliyet alir
    new = inflate(g, 0.05, 0.2, 0.5, unknown_cost=False)
    right = np.zeros_like(g, bool); right[:, 20:] = True
    assert ((old > 0) & (old < 100) & right).any()                 # eski: golgede gecilebilir halka
    assert not ((new >= 0) & (new < 100) & right).any()            # yeni: golgede yalnizca -1 / 100
    np.testing.assert_array_equal(old == 100, new == 100)          # oldurucu bolge ayni
    np.testing.assert_array_equal(old[:, :20], new[:, :20])        # bilinen tarafta degisiklik yok


def test_remove_thin_free_strips_next_to_unknown():
    g = np.full((30, 30), 0, np.int8)                # genis bos alan
    g[:, 20:] = -1                                   # bilinmeyen (golge)
    g[:, 17] = 100                                   # engel (sisirilmis bant temsili)
    g[5:25, 18:20] = 0                               # engel ile golge arasinda 2 hucrelik bos serit
    g[:5, 18:20] = 100; g[25:, 18:20] = 100
    out = remove_thin_free(g, 3)
    assert (out[5:25, 18:20] == -1).all()            # serit butunuyle bilinmeyen (2 sutun da)
    np.testing.assert_array_equal(out[:, :17], g[:, :17])   # genis bos alan aynen
    assert (out == 100).sum() == (g == 100).sum()


def test_remove_thin_free_keeps_gaps_away_from_unknown():
    g = np.full((20, 20), 0, np.int8)
    g[:, 8] = 100; g[:, 10] = 100                    # iki engel arasinda 1 hucrelik bosluk, bilinmeyen yok
    np.testing.assert_array_equal(remove_thin_free(g, 3), g)
    g2 = g.copy(); g2[0:2, 0:2] = -1                 # uzakta bilinmeyen var ama bosluga degmiyor
    np.testing.assert_array_equal(remove_thin_free(g2, 3)[:, 9], g2[:, 9])



def test_remove_thin_free_also_thin_cost_cells():
    """Golge kenari ile oldurucu arasinda kalan ince MALIYETLI (1..99) serit de bilinmeyen olur."""
    g = np.full((30, 30), 0, np.int8)
    g[:, 20:] = -1
    g[:, 17] = 100
    g[5:25, 18:20] = 40                              # bilinen ama ince (2 hucre) maliyet seridi, golgeye degiyor
    g[:5, 18:20] = 100; g[25:, 18:20] = 100
    out = remove_thin_free(g, 3)
    assert (out[5:25, 18:20] == -1).all() and (out == 100).sum() == (g == 100).sum()
    np.testing.assert_array_equal(out[:, :17], g[:, :17])
