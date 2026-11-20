"""
Gercek kamera ic kalibrasyonu (config/camera_calibration.yaml) testleri.
Calistirma:  colcon test --packages-select igvc_vision && colcon test-result --verbose

1) Varsayilan yaml (hepsi enabled: false) ve "enabled: true + simulasyon degerleri" -> kus bakisi
   LUT'u simulasyon modeliyle BIREBIR ayni (mevcut cikti bozulmaz).
2) Distorsiyon formulleri OpenCV (cv2.projectPoints / cv2.fisheye.projectPoints) ile ayni.
3) Uctan uca: OpenCV'nin BAGIMSIZ ters distorsiyonuyla (undistortPoints) zemindeki bilinen bir dama
   tahtasinin "gercek lens" goruntuleri uretilir (110 deg genis aci, kaymis cx/cy, kalibrasyon
   cozunurlugu != isleme cozunurlugu). Kalibrasyonla kurulan kus bakisi dama tahtasini dogru
   metrik konuma koymali; distorsiyonu yok sayan model belirgin bicimde bozuk olmali.
4) Yaml dogrulamasi: hatali dosyalar anlasilir hatayla reddedilir.
"""
import copy
import os

import cv2
import numpy as np
import pytest
import yaml

from igvc_vision import robot_geometry as rg
from igvc_vision.surround_view_node import (
    CAMS, SurroundLUT, _pose, distort_normalized, intrinsic_matrix, vehicle_body,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_YAML = os.path.normpath(os.path.join(HERE, '..', 'config', 'camera_calibration.yaml'))
BEV_PX, RES, RANGE = 500, 0.02, 4.9


@pytest.fixture(scope='module')
def props():
    return rg.read_props()


def _lut(v, calib):
    p = rg.surround_view_params(v, calib)
    veh = {k.split('.', 1)[1]: val for k, val in p.items() if k.startswith('vehicle.')}
    cams = [{'K': intrinsic_matrix(*p[f'{n}.intrinsics']), 'dist': p[f'{n}.dist_coeffs'],
             'pose': p[f'{n}.pose'], 'model': p[f'{n}.distortion_model']} for n in CAMS]
    lut = SurroundLUT(cams, p['image_width'], p['image_height'], BEV_PX, RES, RANGE, 45.0,
                      vehicle_body(veh))
    return lut, p, cams


def _yaml_doc():
    with open(CALIB_YAML) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ 1) mevcut cikti bozulmaz
def test_default_yaml_is_disabled_and_identical_to_sim(props):
    calib = rg.load_calibration(CALIB_YAML)
    assert all(calib[n] is None for n in rg.CAMERA_LAYOUT), 'varsayilan yaml simulasyon olmali'
    L0, p0, _ = _lut(props, None)
    L1, p1, _ = _lut(props, calib)
    assert p0 == p1
    assert np.array_equal(L0.map1, L1.map1) and np.array_equal(L0.map2, L1.map2)
    assert np.array_equal(L0.owner, L1.owner)


def test_enabled_with_sim_values_is_identical(props):
    """Yaml'daki ornek degerler simulasyonun birebir karsiligi: etkinlestirmek ciktiyi degistirmez."""
    doc = _yaml_doc()
    for n in rg.CAMERA_LAYOUT:
        doc[n]['enabled'] = True
    calib = rg.parse_calibration(doc)
    assert all(calib[n] for n in rg.CAMERA_LAYOUT)
    L0, _, _ = _lut(props, None)
    L1, p1, _ = _lut(props, calib)
    assert (p1['image_width'], p1['image_height']) == (int(props['camera_img_w']),
                                                       int(props['camera_img_h']))
    assert np.array_equal(L0.map1, L1.map1) and np.array_equal(L0.map2, L1.map2)
    assert np.array_equal(L0.owner, L1.owner)


# ------------------------------------------------------------------ 2) formuller == OpenCV
K_TEST = np.array([[520.0, 0.0, 482.3], [0.0, 521.5, 297.6], [0.0, 0.0, 1.0]])
MODELS = {
    'plumb_bob': [-0.28, 0.07, 0.0008, -0.0006, -0.006],
    'rational_polynomial': [0.8, 0.15, 0.0005, -0.0004, 0.0, 1.15, 0.35, 0.0],
    'equidistant': [0.05, -0.02, 0.004, -0.001],
}


@pytest.mark.parametrize('model', list(MODELS))
def test_distortion_matches_opencv(model):
    rng = np.random.default_rng(0)
    th = rng.uniform(0, np.radians(62), 2000)
    ph = rng.uniform(-np.pi, np.pi, 2000)
    P = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    D = np.array(MODELS[model])
    xd, yd = distort_normalized(P[:, 0] / P[:, 2], P[:, 1] / P[:, 2], D, model)
    ours = np.c_[K_TEST[0, 0] * xd + K_TEST[0, 2], K_TEST[1, 1] * yd + K_TEST[1, 2]]
    if model == 'equidistant':
        ref, _ = cv2.fisheye.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K_TEST, D)
    else:
        ref, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K_TEST, D)
    np.testing.assert_allclose(ours, ref.reshape(-1, 2), atol=1e-6)


def test_scale_intrinsics_matches_resize():
    """cv2.resize ile ayni piksel merkezi kurali: ayni fiziksel nokta ayni yere duser."""
    K = [1040.0, 1043.0, 964.6, 595.2]
    Ks = rg.scale_intrinsics(K, (1920, 1200), (640, 400))
    np.testing.assert_allclose(Ks, [1040 / 3, 1043 / 3, (964.6 + 0.5) / 3 - 0.5,
                                    (595.2 + 0.5) / 3 - 0.5])


# ------------------------------------------------------------------ 3) uctan uca sentetik lens
CAL_SIZE, SQUARE = (960, 600), 0.25   # kalibrasyon cozunurlugu (isleme 640x400), dama karesi [m]


def _calib_doc(model, dist):
    doc = {'process_width': None}
    for k, n in enumerate(rg.CAMERA_LAYOUT):
        K = K_TEST.copy()
        K[0, 2] += 3.0 * k - 4.0      # kamera basina farkli merkez: kamera ayrimi dogru mu?
        K[1, 2] -= 2.0 * k
        doc[n] = {'enabled': True, 'image_width': CAL_SIZE[0], 'image_height': CAL_SIZE[1],
                  'camera_matrix': {'rows': 3, 'cols': 3, 'data': K.ravel().tolist()},
                  'distortion_model': model,
                  'distortion_coefficients': {'rows': 1, 'cols': len(dist), 'data': list(dist)}}
    return doc


def _checker(X, Y):
    return ((np.floor(X / SQUARE) + np.floor(Y / SQUARE)) % 2).astype(np.uint8) * 255


def _render(doc_cam, pose, model, dist):
    """OpenCV ters distorsiyonu ile 'gercek' kamera goruntusu: piksel -> isin -> zemin -> renk."""
    w, h = CAL_SIZE
    K = np.array(doc_cam['camera_matrix']['data']).reshape(3, 3)
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pts = np.c_[uu.ravel(), vv.ravel()].reshape(-1, 1, 2)
    if model == 'equidistant':
        n = cv2.fisheye.undistortPoints(pts, K, np.array(dist)).reshape(-1, 2)
    else:
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)
        n = cv2.undistortPointsIter(pts, K, np.array(dist), None, None, crit).reshape(-1, 2)
    x, y, z, R = _pose(pose)
    d = R @ np.c_[n, np.ones(len(n))].T             # base cercevesinde isin yonu
    with np.errstate(divide='ignore', invalid='ignore'):
        t = -z / d[2]
    X, Y = x + t * d[0], y + t * d[1]
    img = np.where(t > 0, _checker(X, Y), 128).astype(np.uint8).reshape(h, w)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _bev_accuracy(lut, imgs):
    bev = lut.warp(imgs)[:, :, 0].astype(int)
    c = BEV_PX / 2.0
    vv, uu = np.mgrid[0:BEV_PX, 0:BEV_PX].astype(np.float64)
    X, Y = (c - vv) * RES, (c - uu) * RES
    edge = np.minimum(np.abs(X / SQUARE - np.round(X / SQUARE)),
                      np.abs(Y / SQUARE - np.round(Y / SQUARE))) * SQUARE
    sel = (lut.owner >= 0) & (np.hypot(X, Y) <= 3.0) & (edge > 0.05)
    return float((np.abs(bev - _checker(X, Y).astype(int)) < 64)[sel].mean()), int(sel.sum())


@pytest.mark.parametrize('model', list(MODELS))
def test_end_to_end_synthetic_lens(props, model):
    dist = MODELS[model]
    doc = _calib_doc(model, dist)
    calib = rg.parse_calibration(doc)
    lut, p, _ = _lut(props, calib)
    assert (p['image_width'], p['image_height']) == (640, 400)   # 960x600 -> 640x400
    imgs = [cv2.resize(_render(doc[n], p[f'{n}.pose'], model, dist), (640, 400))
            for n in CAMS]
    acc, n_px = _bev_accuracy(lut, imgs)
    assert n_px > 20000
    assert acc > 0.995, f'{model}: kalibrasyonlu kus bakisi dogrulugu %{acc * 100:.2f}'

    # Ayni K, distorsiyonu yok sayan model (kalibrasyon girilmemis gibi) belirgin bicimde bozuk
    naive = copy.deepcopy(doc)
    for n in rg.CAMERA_LAYOUT:
        naive[n]['distortion_model'] = 'plumb_bob'
        naive[n]['distortion_coefficients'] = {'rows': 1, 'cols': 5, 'data': [0.0] * 5}
    lut_n, _, _ = _lut(props, rg.parse_calibration(naive))
    acc_n, _ = _bev_accuracy(lut_n, imgs)
    assert acc_n < acc - 0.05, f'{model}: distorsiyonsuz %{acc_n * 100:.1f} vs %{acc * 100:.1f}'


# ------------------------------------------------------------------ 4) yaml dogrulamasi
def _one(**over):
    doc = _calib_doc('plumb_bob', MODELS['plumb_bob'])
    doc['front'].update(over)
    return doc


@pytest.mark.parametrize('bad, msg', [
    (_one(distortion_coefficients=[0.1, 0.2]), 'katsayi'),
    (_one(distortion_model='kannala'), 'desteklenmiyor'),
    (_one(camera_matrix=[1, 0, 0, 0, 1, 0, 0, 0]), '9 elemanli'),
    (_one(camera_matrix=[500, 1, 320, 0, 500, 200, 0, 0, 1]), 'bekleniyor'),
    (_one(image_width=0), 'pozitif'),
    (_one(image_width=800, image_height=600), 'en-boy'),
])
def test_invalid_calibration_rejected(props, bad, msg):
    with pytest.raises(rg.CalibrationError, match=msg):
        rg.camera_models(props, rg.parse_calibration(bad))


def test_mixed_aspect_with_sim_cameras_rejected(props):
    doc = _calib_doc('plumb_bob', MODELS['plumb_bob'])   # 16:10, sim 3:2
    doc['left']['enabled'] = False
    with pytest.raises(rg.CalibrationError, match='hepsini'):
        rg.camera_models(props, rg.parse_calibration(doc))


def test_missing_explicit_file_rejected():
    with pytest.raises(rg.CalibrationError, match='bulunamadi'):
        rg.load_calibration('/yok/boyle/bir/dosya.yaml')


def test_plumb_bob_four_coeffs_padded_and_process_width(props):
    doc = _calib_doc('plumb_bob', [-0.2, 0.05, 0.0, 0.0])
    doc['process_width'] = 800
    size, models = rg.camera_models(props, rg.parse_calibration(doc))
    assert size == (800, 500)
    assert models['front']['dist'] == [-0.2, 0.05, 0.0, 0.0, 0.0]
