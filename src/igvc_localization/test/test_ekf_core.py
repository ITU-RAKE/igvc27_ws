"""ekf_core.py testleri: model tutarliligi, entegrasyon, aci sarmasi, sapma tahmini, kapi."""
import math

import numpy as np
import pytest

from igvc_localization.ekf_core import BG, Ekf, N, PX, PY, V, W, YAW, wrap

Q = [1e-4, 1e-4, 1e-5, 0.5, 0.5, 1e-7]


def numeric_jacobian(x, dt, eps=1e-6):
    def f(xx):
        e = Ekf(Q, x0=xx)
        e.predict(dt)
        return e.x
    J = np.zeros((N, N))
    for i in range(N):
        d = np.zeros(N); d[i] = eps
        a, b = f(x + d), f(x - d)
        diff = a - b
        diff[YAW] = wrap(diff[YAW])
        J[:, i] = diff / (2 * eps)
    return J


@pytest.mark.parametrize('x', [[0, 0, 0.3, 1.2, 0.4, 0.01], [2, -1, -2.9, 0.5, -1.1, 0.0],
                               [0, 0, 3.1, 2.0, 2.0, 0.0]])
def test_jacobian_matches_model(x):
    """Analitik Jacobian sayisal turevle ayni (eski kodda tutarsizdi)."""
    x = np.array(x, float)
    e = Ekf(Q, x0=x, p0=np.zeros((N, N)))
    e.q[:] = 0.0
    e.predict(0.05)
    J = numeric_jacobian(x, 0.05)
    np.testing.assert_allclose(e._F, J, atol=1e-6)


def drive(e, v, w, seconds, dt=0.01, gyro_bias=0.0):
    for _ in range(int(round(seconds / dt))):
        e.predict(dt)
        e.update_wheel(v, w, 0.05, 0.1)
        e.update_gyro(w + gyro_bias, 0.002)


def test_straight_line():
    e = Ekf(Q)
    drive(e, 1.0, 0.0, 10.0)
    assert e.x[PX] == pytest.approx(10.0, abs=0.05)
    assert abs(e.x[PY]) < 1e-6 and abs(e.x[YAW]) < 1e-6


def test_circle_matches_analytic():
    """v = 1, w = 0.5: tam tur (4 pi s) sonunda baslangica doner; yaricap v/w = 2 m."""
    e = Ekf(Q)
    e.x[V], e.x[W] = 1.0, 0.5
    drive(e, 1.0, 0.5, math.pi / 0.5)            # yarim tur
    assert e.x[PX] == pytest.approx(0.0, abs=0.03)
    assert e.x[PY] == pytest.approx(4.0, abs=0.03)
    assert abs(wrap(e.x[YAW] - math.pi)) < 0.01
    drive(e, 1.0, 0.5, math.pi / 0.5)            # tam tur
    assert math.hypot(e.x[PX], e.x[PY]) < 0.05


def test_yaw_wraps_across_pi():
    """yaw ~ +pi, olcum ~ -pi: inovasyon kucuk, sicrama yok (eski kodda 2 pi hata)."""
    e = Ekf(Q, x0=[0, 0, math.pi - 0.01, 0, 0, 0])
    ok, d2 = e.update_yaw(-math.pi + 0.01, 0.05)
    assert ok and d2 < 1.0
    assert abs(wrap(e.x[YAW] - math.pi)) < 0.02
    # sarmali yon donusu
    e2 = Ekf(Q, x0=[0, 0, 3.0, 0, 1.0, 0])
    e2.predict(0.5)
    assert -math.pi <= e2.x[YAW] <= math.pi
    assert e2.x[YAW] == pytest.approx(wrap(3.5), abs=1e-9)


def test_gyro_bias_is_estimated():
    """Arac duruyor, jiroskop 0.01 rad/s sapmali: yon kaymaz, b_g ~ 0.01 olur."""
    e = Ekf(Q)
    for _ in range(3000):
        e.predict(0.01)
        e.update_wheel(0.0, 0.0, 0.05, 0.1)
        e.update_gyro(0.01, 0.002)
        e.update_yaw(0.0, 0.02)
    assert e.x[BG] == pytest.approx(0.01, abs=0.002)
    assert abs(e.x[YAW]) < 0.01


def test_gate_rejects_outlier_position():
    e = Ekf(Q, p0=np.eye(N) * 0.01)
    ok, d2 = e.update_position(0.05, -0.03, np.eye(2) * 0.02 ** 2)
    assert ok
    before = e.x.copy()
    ok, d2 = e.update_position(25.0, 0.0, np.eye(2) * 0.02 ** 2)     # 25 m sicrama
    assert not ok and d2 > 100
    np.testing.assert_array_equal(e.x, before)


def test_lever_arm():
    """Anten 0.5 m onde, arac kuzeye bakiyor: anten (0, 0.5)'te olculurse merkez (0, 0)."""
    e = Ekf(Q, x0=[0.3, -0.2, math.pi / 2, 0, 0, 0], p0=np.diag([1, 1, 1e-6, 1, 1, 1e-6]))
    for _ in range(20):
        e.update_position(0.0, 0.5, np.eye(2) * 1e-4, lever=(0.5, 0.0))
    assert e.x[PX] == pytest.approx(0.0, abs=1e-3)
    assert e.x[PY] == pytest.approx(0.0, abs=1e-3)


def test_covariance_stays_symmetric_positive():
    rng = np.random.default_rng(1)
    e = Ekf(Q)
    for _ in range(2000):
        e.predict(rng.uniform(0.001, 0.05))
        k = rng.integers(4)
        if k == 0:
            e.update_wheel(rng.normal(1, 0.1), rng.normal(0, 0.1), 0.05, 0.1)
        elif k == 1:
            e.update_gyro(rng.normal(0, 0.05), 0.002)
        elif k == 2:
            e.update_yaw(rng.uniform(-math.pi, math.pi), 0.05, gate=False)
        else:
            e.update_position(*rng.normal(0, 1, 2), np.eye(2) * 0.01, gate=False)
    np.testing.assert_allclose(e.P, e.P.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(e.P) > 0)


def test_gate_lockout_and_reset():
    """
    Filtre GPS'ten 3 m kopmus ve kendinden emin: kapi dogru GPS'i de surekli reddeder
    (kilitlenme). reset_position konumu GPS'e kilitler, yon/sapma korunur.
    """
    e = Ekf(Q, x0=[3.0, 0.0, 0.7, 0.0, 0.0, 0.002], p0=np.diag([1e-4, 1e-4, 1e-4, 0.1, 0.1, 1e-6]))
    gps_cov = np.eye(2) * 0.02 ** 2
    rejected = sum(not e.update_position(0.0, 0.0, gps_cov)[0] for _ in range(20))
    assert rejected == 20                                     # kilitlenme
    ok, _ = e.reset_position(0.0, 0.0, gps_cov)
    assert ok
    assert math.hypot(e.x[PX], e.x[PY]) < 0.05
    assert e.x[YAW] == pytest.approx(0.7) and e.x[BG] == pytest.approx(0.002)
    assert e.update_position(0.01, -0.01, gps_cov)[0]         # sonraki olcumler yine kabul
    np.testing.assert_allclose(e.P, e.P.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(e.P) > 0)


def test_tilt_lever():
    """Anten 1 m yukarida: 10 deg burun yukari (pitch -10 deg, ROS) -> anten 17 cm geride."""
    from igvc_localization.ekf_node import tilt_lever
    lx, ly = tilt_lever((0.0, 0.0, 1.0), 0.0, math.radians(-10.0))
    assert lx == pytest.approx(-math.sin(math.radians(10.0)), abs=1e-9) and abs(ly) < 1e-12
    lx, ly = tilt_lever((0.0, 0.0, 1.0), math.radians(5.0), 0.0)      # saga yatis (roll +)
    assert ly == pytest.approx(-math.sin(math.radians(5.0)), abs=1e-9)
    assert tilt_lever((0.3, -0.1, 1.0), 0.0, 0.0) == pytest.approx((0.3, -0.1))
