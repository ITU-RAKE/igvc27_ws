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


def surround_view_params(v):
    """surround_view dugumune verilecek geometri parametrelerinin tamami."""
    w, h, K = camera_intrinsics(v)
    params = {'image_width': w, 'image_height': h, 'frame_id': 'base_footprint'}
    params.update(vehicle_params(v))
    for name, pose in camera_poses(v).items():
        params[f'{name}.pose'] = pose
        params[f'{name}.intrinsics'] = K
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
