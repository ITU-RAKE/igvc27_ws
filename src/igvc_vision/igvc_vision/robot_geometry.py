"""
Arac geometrisini igvc_description/urdf/igvc_macros.xacro'dan okur (tek kaynak).

Launch dosyalari bunu kullanarak dugumlere arac olculerini, kamera pozlarini ve
intrinsic'lerini parametre olarak verir; boylece URDF'te bir olcu degisince lidar
haritasi ve kus bakisi kendiliginden guncellenir. ROS'a bagimli degildir (test edilebilir).

Tum konumlar base_footprint cercevesinde: zeminde, aks merkezinin altinda, x ileri, y sol.
"""
import math
import os

XACRO_NS = 'http://www.ros.org/wiki/xacro'

PROPS = (
    'chassis_bottom_length', 'chassis_bottom_width', 'chassis_top_length', 'chassis_top_width',
    'chassis_height', 'ground_clearance',
    'wheel_radius', 'wheel_width', 'wheel_separation', 'wheel_x_offset',
    'camera_height', 'camera_x_offset', 'camera_y_offset',
    'camera_pitch', 'camera_roll', 'camera_yaw', 'camera_hfov', 'camera_img_w', 'camera_img_h',
)

# Kamera adi (dugumdeki) -> URDF'teki yerlesim: (x isareti, y isareti, yaw)
CAMERA_LAYOUT = {
    'front': (1, 0, 0.0),
    'back': (-1, 0, math.pi),
    'left': (0, 1, math.pi / 2),
    'right': (0, -1, -math.pi / 2),
}


def default_macros_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('igvc_description'),
                            'urdf', 'igvc_macros.xacro')
    except Exception:  # noqa: BLE001  kaynak agacindan (test) calisirken
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(here, '..', '..', 'igvc_description',
                                             'urdf', 'igvc_macros.xacro'))


def read_props(macros_path=None, names=PROPS):
    """Macros dosyasindaki property'leri xacro ile degerlendir (ifadeler dahil)."""
    import xacro
    macros_path = macros_path or default_macros_path()
    attrs = ' '.join(f'{p}="${{{p}}}"' for p in names)
    doc = xacro.parse(f'<robot xmlns:xacro="{XACRO_NS}">'
                      f'<xacro:include filename="{os.path.abspath(macros_path)}"/>'
                      f'<v {attrs}/></robot>')
    xacro.process_doc(doc)
    el = doc.getElementsByTagName('v')[0]
    return {p: float(el.getAttribute(p)) for p in names}


def chassis_center_x(v):
    """Govde merkezinin base_footprint'e gore x'i (chassis_joint)."""
    return -v['wheel_x_offset']


def footprint(v):
    """Tekerlekler dahil en buyuk izdusum: (uzunluk, genislik, merkez_x)."""
    cx = chassis_center_x(v)
    half_len = max(v['chassis_bottom_length'], v['chassis_top_length']) / 2
    x0, x1 = min(cx - half_len, -v['wheel_radius']), max(cx + half_len, v['wheel_radius'])
    width = max(v['chassis_bottom_width'], v['chassis_top_width'],
                v['wheel_separation'] + v['wheel_width'])
    return x1 - x0, width, (x0 + x1) / 2


def footprint_rect(v):
    """Ayak izi [x0, x1, y0, y1] base_footprint'te (tekerlekler dahil)."""
    length, width, cx = footprint(v)
    return [cx - length / 2, cx + length / 2, -width / 2, width / 2]


def circumscribed_radius(v):
    """Cevrel yaricap: base_footprint'ten ayak izinin en uzak kosesine (yarim kosegen) [m].
    Merkezli arac icin sqrt((L/2)^2 + (W/2)^2)."""
    x0, x1, y0, y1 = footprint_rect(v)
    return max(math.hypot(x, y) for x in (x0, x1) for y in (y0, y1))


def inflation_radii(v, safety_margin=0.05):
    """A* maliyet haritasi icin (lethal, inflation) yaricaplari [m], base_footprint merkezli.
    lethal    = ic teget yaricap (yari genislik) + pay: merkez buraya girerse govde kesin carpar.
    inflation = cevrel yaricap (yarim kosegen): yerinde donerken kose carpabilir -> buraya kadar
                azalan maliyet. Arac boyunca ilerlerken yari genislik yeterli; cevrel yaricap
                yalnizca donus icin gerekli (Nav2 ile ayni mantik)."""
    lethal = inscribed_radius(v) + safety_margin
    return lethal, max(circumscribed_radius(v), lethal)


def inscribed_radius(v):
    """Ic teget yaricap: base_footprint'ten ayak izinin en yakin kenarina [m]."""
    x0, x1, y0, y1 = footprint_rect(v)
    return min(x1, -x0, y1, -y0)


def vehicle_params(v):
    """surround_view icin 'vehicle.*' parametreleri (govde + tekerlek, kamera gorusunu kapatan)."""
    return {
        'vehicle.chassis_bottom_length': v['chassis_bottom_length'],
        'vehicle.chassis_bottom_width': v['chassis_bottom_width'],
        'vehicle.chassis_top_length': v['chassis_top_length'],
        'vehicle.chassis_top_width': v['chassis_top_width'],
        'vehicle.chassis_height': v['chassis_height'],
        'vehicle.ground_clearance': v['ground_clearance'],
        'vehicle.chassis_center_x': chassis_center_x(v),
        'vehicle.wheel_radius': v['wheel_radius'],
        'vehicle.wheel_width': v['wheel_width'],
        'vehicle.wheel_separation': v['wheel_separation'],
        'vehicle.wheel_x': 0.0,  # tekerlekler base_link (aks) uzerinde
    }


def camera_intrinsics(v):
    """Gazebo pinhole (kare piksel, distorsiyonsuz): [fx, fy, cx, cy], 0-tabanli piksel."""
    w, h = int(v['camera_img_w']), int(v['camera_img_h'])
    fx = (w / 2.0) / math.tan(v['camera_hfov'] / 2.0)
    return w, h, [fx, fx, (w - 1) / 2.0, (h - 1) / 2.0]


def camera_poses(v):
    """Her kamera icin [x, y, z, yaw_deg, pitch_deg, roll_deg] (pitch asagi +), base_footprint."""
    cx = chassis_center_x(v)
    poses = {}
    for name, (sx, sy, yaw) in CAMERA_LAYOUT.items():
        poses[name] = [
            cx + sx * v['camera_x_offset'],
            sy * v['camera_y_offset'],
            v['camera_height'],
            math.degrees(yaw + v['camera_yaw']),
            math.degrees(v['camera_pitch']),
            math.degrees(v['camera_roll']),
        ]
    return poses


# ===================== Gercek kamera kalibrasyonu (config/camera_calibration.yaml) =====================
# Kamera POZLARI her zaman xacro'dan; buradan yalnizca kameranin ic ayarlari gelir.
# distortion_model -> izin verilen katsayi sayilari (ROS camera_info / OpenCV sirasi)
DIST_MODELS = {
    'plumb_bob': (4, 5),            # k1 k2 p1 p2 [k3]
    'rational_polynomial': (8,),    # k1 k2 p1 p2 k3 k4 k5 k6
    'equidistant': (4,),            # fisheye: k1 k2 k3 k4 (OpenCV cv2.fisheye)
}
ASPECT_TOL = 0.01  # kalibrasyon / isleme / gelen goruntu en-boy orani toleransi (%1)


class CalibrationError(ValueError):
    """camera_calibration.yaml hatali ya da tutarsiz."""


def default_calibration_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('igvc_vision'),
                            'config', 'camera_calibration.yaml')
    except Exception:  # noqa: BLE001  kaynak agacindan (test) calisirken
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(here, '..', 'config', 'camera_calibration.yaml'))


def _matrix_data(val, n, what):
    """camera_calibration ciktisi ({rows, cols, data}) ya da duz liste -> float listesi."""
    data = val.get('data') if isinstance(val, dict) else val
    if not isinstance(data, (list, tuple)) or (n is not None and len(data) != n):
        raise CalibrationError(f'{what}: {n} elemanli liste bekleniyor, gelen: {data!r}')
    try:
        return [float(x) for x in data]
    except (TypeError, ValueError) as e:
        raise CalibrationError(f'{what}: sayi olmayan eleman: {data!r}') from e


def parse_calibration(doc):
    """YAML sozlugu -> {'process_width': int|None, kamera: {...} | None (etkin degil)}.
    Etkin kamera: {'size': (w, h), 'K': [fx, fy, cx, cy], 'model': str, 'dist': [..]}."""
    doc = doc or {}
    out = {'process_width': doc.get('process_width')}
    if out['process_width'] is not None:
        out['process_width'] = int(out['process_width'])
        if out['process_width'] <= 0:
            raise CalibrationError('process_width pozitif olmali')
    for name in CAMERA_LAYOUT:
        c = doc.get(name) or {}
        if not c.get('enabled', False):
            out[name] = None
            continue
        where = f'{name}'
        try:
            w, h = int(c['image_width']), int(c['image_height'])
        except (KeyError, TypeError, ValueError) as e:
            raise CalibrationError(f'{where}: image_width / image_height eksik ya da hatali') from e
        if w <= 0 or h <= 0:
            raise CalibrationError(f'{where}: goruntu boyutu pozitif olmali')
        m = _matrix_data(c.get('camera_matrix'), 9, f'{where}.camera_matrix')
        fx, skew, cx, _, fy, cy, r20, r21, r22 = m
        if fx <= 0 or fy <= 0:
            raise CalibrationError(f'{where}.camera_matrix: fx, fy pozitif olmali')
        if abs(skew) > 1e-9 or abs(r20) > 1e-9 or abs(r21) > 1e-9 or abs(r22 - 1.0) > 1e-9:
            raise CalibrationError(f'{where}.camera_matrix: [fx 0 cx; 0 fy cy; 0 0 1] bekleniyor')
        model = c.get('distortion_model', 'plumb_bob')
        if model not in DIST_MODELS:
            raise CalibrationError(f'{where}.distortion_model: {model!r} desteklenmiyor '
                                   f'({", ".join(DIST_MODELS)})')
        dist = _matrix_data(c.get('distortion_coefficients', []), None,
                            f'{where}.distortion_coefficients')
        if len(dist) not in DIST_MODELS[model]:
            raise CalibrationError(f'{where}: {model} icin {DIST_MODELS[model]} katsayi '
                                   f'bekleniyor, {len(dist)} geldi')
        if model == 'plumb_bob' and len(dist) == 4:
            dist = dist + [0.0]
        out[name] = {'size': (w, h), 'K': [fx, fy, cx, cy], 'model': model, 'dist': dist}
    return out


def load_calibration(path=None):
    """camera_calibration.yaml'i oku ve dogrula. Varsayilan dosya yoksa hepsi etkin degil (sim);
    acikca verilen dosya yoksa hata (yanlis yol sessizce simulasyon modeline dusmesin)."""
    import yaml
    if path and not os.path.exists(path):
        raise CalibrationError(f'Kalibrasyon dosyasi bulunamadi: {path}')
    path = path or default_calibration_path()
    if not os.path.exists(path):
        return parse_calibration({})
    with open(path) as f:
        return parse_calibration(yaml.safe_load(f))


def scale_intrinsics(K, src_size, dst_size):
    """[fx, fy, cx, cy] cozunurluk degisimi (cv2.resize ile ayni: piksel merkezi 0.5 kaymali)."""
    fx, fy, cx, cy = K
    sx, sy = dst_size[0] / src_size[0], dst_size[1] / src_size[1]
    return [fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5]


def camera_models(v, calib=None):
    """Kamera basina homografi icin ic model: (isleme (w, h), {kamera: {K, model, dist}}).
    Etkin olmayan kamera: xacro'dan ideal pinhole (simulasyon, distorsiyonsuz) -> eski davranis.
    Etkin kamera: kalibrasyon K'si isleme cozunurlugune olceklenir. Isleme genisligi process_width
    (yoksa xacro camera_img_w), yukseklik kalibrasyonun en-boy oranindan."""
    calib = calib or {}
    sim_w, sim_h, sim_K = camera_intrinsics(v)
    enabled = {n: calib.get(n) for n in CAMERA_LAYOUT if calib.get(n)}
    if not enabled:
        size = (sim_w, sim_h)
    else:
        aspects = {n: c['size'][0] / c['size'][1] for n, c in enabled.items()}
        a0 = next(iter(aspects.values()))
        if any(abs(a / a0 - 1) > ASPECT_TOL for a in aspects.values()):
            raise CalibrationError(f'Etkin kameralarin en-boy oranlari farkli: {aspects}')
        W = calib.get('process_width') or sim_w
        size = (int(W), int(round(W / a0)))
        if len(enabled) < len(CAMERA_LAYOUT) and abs((sim_w / sim_h) / a0 - 1) > ASPECT_TOL:
            raise CalibrationError(
                f'Etkin kameralarin en-boy orani ({a0:.3f}) etkin olmayan (simulasyon, '
                f'{sim_w}x{sim_h}) kameralardan farkli: ya hepsini etkinlestir ya hicbirini')
    models = {}
    for name in CAMERA_LAYOUT:
        c = enabled.get(name)
        if c is None:
            K = sim_K if size == (sim_w, sim_h) else scale_intrinsics(sim_K, (sim_w, sim_h), size)
            models[name] = {'K': K, 'model': 'plumb_bob', 'dist': [0.0] * 5}
        else:
            models[name] = {'K': scale_intrinsics(c['K'], c['size'], size),
                            'model': c['model'], 'dist': list(c['dist'])}
    return size, models


def surround_view_params(v, calib=None):
    """surround_view dugumune verilecek geometri parametrelerinin tamami.
    calib: load_calibration() sonucu (None -> hepsi simulasyon, eski davranis)."""
    (w, h), models = camera_models(v, calib)
    params = {'image_width': w, 'image_height': h, 'frame_id': 'base_footprint'}
    params.update(vehicle_params(v))
    for name, pose in camera_poses(v).items():
        params[f'{name}.pose'] = pose
        params[f'{name}.intrinsics'] = models[name]['K']
        params[f'{name}.distortion_model'] = models[name]['model']
        params[f'{name}.dist_coeffs'] = models[name]['dist']
    return params


def bev_homographies(v, bev_size, bev_meters, cams=('front', 'back', 'left', 'right')):
    """Kamera pikseli -> BEV pikseli homografileri (four_cam_test icin), duz liste (36).
    BEV: bev_size x bev_size px, bev_meters genislik, base_footprint merkezde, yukari = ileri."""
    import numpy as np
    from igvc_vision.surround_view_node import (
        ground_to_bev_matrix, ground_to_image_h, intrinsic_matrix)
    _, _, K = camera_intrinsics(v)
    A = ground_to_bev_matrix(bev_size, bev_meters / bev_size)
    poses = camera_poses(v)
    out = []
    for name in cams:
        H = A @ np.linalg.inv(ground_to_image_h(intrinsic_matrix(*K), poses[name]))
        out.extend((H / H[2, 2]).ravel().tolist())
    return out
