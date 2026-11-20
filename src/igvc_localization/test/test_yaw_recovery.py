"""Yon kilitlenmesinden kurtarma testleri (ekf_core: reset_yaw, CourseTracker, arbitrate_yaw)."""
import math

import numpy as np
import pytest

from igvc_localization.ekf_core import (CourseTracker, Ekf, PX, PY, V, W, YAW,
                                        arbitrate_yaw, wrap)

Q = [1e-4, 1e-4, 1e-5, 0.5, 1.0, 1e-8]
D = math.radians


def test_reset_yaw_only_touches_yaw():
    e = Ekf(Q, x0=[1.0, 2.0, 0.1, 0.5, 0.0, 0.003], p0=np.eye(6) * 0.01)
    e.predict(0.5)
    x_before = e.x.copy()
    e.reset_yaw(D(170), 0.02)
    assert e.x[YAW] == pytest.approx(D(170))
    np.testing.assert_allclose(np.delete(e.x, YAW), np.delete(x_before, YAW))
    assert e.P[YAW, YAW] == pytest.approx(0.02 ** 2)
    assert np.all(np.delete(e.P[YAW], YAW) == 0) and np.all(np.delete(e.P[:, YAW], YAW) == 0)
    assert np.all(np.linalg.eigvalsh(e.P) > 0)


# ------------------------------------------------------------------ rota
def feed(ct, pts, v=0.5, w=0.0, std=0.02, dt=0.1):
    out = None
    for k, (x, y) in enumerate(pts):
        out = ct.add(k * dt, x, y, v, w, std)
    return out


def test_course_straight_and_min_distance():
    ct = CourseTracker(min_dist=1.5, dist_sigma=16.0)
    pts = [(0.05 * k, 0.05 * k) for k in range(20)]         # 1.34 m: henuz degil
    assert feed(ct, pts) is None
    pts += [(0.05 * k, 0.05 * k) for k in range(20, 25)]    # 1.70 m
    assert feed(ct, pts[-1:]) == pytest.approx(D(45), abs=1e-6)


def test_course_distance_grows_with_gps_noise():
    ct = CourseTracker(min_dist=1.5, dist_sigma=16.0)
    pts = [(0.1 * k, 0.0) for k in range(51)]               # 5 m duz
    assert feed(ct, pts, std=1.0) is None                    # 1 m'lik GPS: 16 m gerekir
    ct.reset()
    assert feed(ct, pts, std=0.02) == pytest.approx(0.0, abs=1e-9)


def test_course_turn_or_stop_resets():
    ct = CourseTracker()
    feed(ct, [(0.1 * k, 0.0) for k in range(10)])
    assert ct.add(1.0, 1.0, 0.0, 0.5, 0.6, 0.02) is None and ct.track == []   # donus
    feed(ct, [(0.1 * k, 0.0) for k in range(10)])
    assert ct.add(1.0, 1.0, 0.0, 0.05, 0.0, 0.02) is None and ct.track == []  # durdu


def test_course_reverse_points_to_nose():
    ct = CourseTracker()
    pts = [(-0.1 * k, 0.0) for k in range(20)]              # bati yonune geri gidiyor
    assert abs(wrap(feed(ct, pts, v=-0.5) - 0.0)) < 1e-9    # burun doguya bakiyor


# ------------------------------------------------------------------ hakem
@pytest.mark.parametrize('course, imu, filt, expect', [
    (D(90), D(91), D(0), 'imu'),         # filtre 90 deg kopmus, IMU dogru
    (D(0), D(60), D(1), 'filter'),       # IMU bozuk (manyetik), filtre dogru
    (D(-170), D(175), D(60), 'imu'),     # aci sarmasi
    (D(45), D(0), D(90), None),          # ikisi de uzak
    (D(0), D(4), D(-5), None),           # fark margin'den kucuk
    (D(0), D(2), D(15), 'imu'),          # ikisi de tol icinde ama IMU belirgin yakin
])
def test_arbitrate(course, imu, filt, expect):
    assert arbitrate_yaw(course, imu, filt, D(20), D(10)) == expect


# ------------------------------------------------------------------ senaryo (dugum akisinin taklidi)
def run_scenario(true_yaw, filt_yaw, imu_yaw, steps=80, v=0.5, reset_after=100):
    """Filtre (0,0) ve filt_yaw'da, arac gercekte true_yaw yonunde v ile duz gidiyor; IMU imu_yaw der.
    100 Hz tekerlek+jiro+IMU yaw, 10 Hz GPS (2 cm). Dugumdeki sira: yaw kapisi -> ust uste ret ->
    supheli -> GPS rotasi -> hakem."""
    rng = np.random.default_rng(0)
    e = Ekf(Q, x0=[0, 0, filt_yaw, v, 0, 0], p0=np.diag([1e-4, 1e-4, 1e-4, 0.01, 0.01, 1e-6]))
    ct, rejects, suspect, verdicts = CourseTracker(), 0, False, []
    gps_cov = np.eye(2) * 0.02 ** 2
    for k in range(steps * 10):
        t = k * 0.01
        e.predict(0.01)
        e.update_wheel(v, 0.0, 0.05, 0.02)
        e.update_gyro(0.0, 0.005)
        ok, _ = e.update_yaw(imu_yaw, 0.02)
        rejects = 0 if ok else rejects + 1
        if rejects >= reset_after:
            suspect = True
        if k % 10 == 0:
            px, py = v * t * math.cos(true_yaw), v * t * math.sin(true_yaw)
            px, py = px + rng.normal(0, 0.02), py + rng.normal(0, 0.02)
            course = ct.add(t, px, py, e.x[V], e.x[W], 0.02)
            if suspect and course is not None:
                verdict = arbitrate_yaw(course, imu_yaw, e.x[YAW], D(20), D(10))
                verdicts.append(verdict)
                if verdict == 'imu':
                    e.reset_yaw(imu_yaw, 0.02)
                    e.reset_position(px, py, gps_cov)
                    suspect, rejects = False, 0
                    ct.reset()
                    continue
            e.update_position(px, py, gps_cov)
    return e, verdicts


def test_scenario_teleported_robot_recovers():
    """Gazebo'da 90 deg dondurulmus arac: filtre 0, gercek ve IMU 90 deg -> yon ve konum toparlanir."""
    e, verdicts = run_scenario(true_yaw=D(90), filt_yaw=0.0, imu_yaw=D(90))
    assert 'imu' in verdicts
    assert abs(wrap(e.x[YAW] - D(90))) < D(2)
    assert math.hypot(e.x[PX] - 0.0, e.x[PY] - 0.5 * 79 * 0.1) < 0.1   # GPS'le ortusur
    # Sonrasinda IMU yaw'i yine kabul ediliyor
    assert e.update_yaw(D(90), 0.02)[0]


def test_scenario_magnetic_disturbance_keeps_filter():
    """Filtre dogru (0), IMU 60 deg bozuk: hakem 'filter' der, yon hic degismez."""
    e, verdicts = run_scenario(true_yaw=0.0, filt_yaw=0.0, imu_yaw=D(60))
    assert verdicts and set(verdicts) == {'filter'}
    assert abs(wrap(e.x[YAW])) < D(1)


def test_normal_driving_never_suspect():
    """Uyumlu IMU: hic ust uste ret yok, hakeme hic gidilmez."""
    e, verdicts = run_scenario(true_yaw=D(30), filt_yaw=D(30), imu_yaw=D(30))
    assert verdicts == []
