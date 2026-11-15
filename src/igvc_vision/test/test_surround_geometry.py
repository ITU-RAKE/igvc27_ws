"""
Surround-view geometri testleri.
Çalıştırma:  colcon test --packages-select igvc_vision && colcon test-result --verbose

1) Homografi matematigi: docs/sistem_ozeti.md'deki elle hesaplanmis degerlerle (Muharrem'in
   referans araci: 720x480, x=0.35, y=0.21). Kodun varsayilanlarindan bagimsiz.
2) Bizim arac: kamera modeli URDF'teki gercek TF zinciriyle (xacro -> joint origin'leri)
   ayni pikselleri vermeli; kapsama ve menzil yeterli olmali.
"""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from igvc_vision import robot_geometry as rg
from igvc_vision.surround_view_node import (
    CAMS, SurroundLUT, ground_to_bev_matrix, ground_to_image_h, intrinsic_matrix, vehicle_body,
)

# ---------------------------------------------------------------- 1) referans (sistem_ozeti.md)
SPEC_INTRINSICS = [312.943226, 312.774089, 360.0, 240.0]  # 720x480
SPEC_POSES = {
    "front": [0.35, 0.0, 1.0, 0.0, 43.0],
    "back": [-0.35, 0.0, 1.0, 180.0, 43.0],
    "left": [0.0, 0.21, 1.0, 90.0, 43.0],
    "right": [0.0, -0.21, 1.0, -90.0, 43.0],
}
SPEC_G2I = {
    "front": [[618.009745, -734.566153, 360.0], [-88.695655, 0, 952.183074], [1.716694, 0, 1]],
    "back": [[-618.009745, 734.566153, 360.0], [88.695655, 0, 952.183074], [-1.716694, 0, 1]],
    "left": [[592.231047, 498.259492, 360.0], [0, -71.509312, 757.66956], [0, 1.384054, 1]],
    "right": [[-592.231047, -498.259492, 360.0], [0, 71.509312, 757.66956], [0, -1.384054, 1]],
}
SPEC_I2BEV = {
    "front": [[1.322508, 4.83872, -226.102798], [0, 5.402445, -286.769854], [0, 0.019355, 1]],
    "back": [[-1.322508, 4.83872, 726.102798], [0, 4.274994, 786.769854], [0, 0.019355, 1]],
    "left": [[0, 5.53793, -279.769854], [-1.322508, 4.83872, 726.102798], [0, 0.019355, 1]],
    "right": [[0, 4.13951, 779.769854], [1.322508, 4.83872, -226.102798], [0, 0.019355, 1]],
}


def _spec_cams():
    K = intrinsic_matrix(*SPEC_INTRINSICS)
    return [{"K": K, "dist": [0.0] * 5, "pose": SPEC_POSES[n]} for n in CAMS]


def test_homographies_match_spec():
    A = ground_to_bev_matrix(500, 0.02)
    for name, cam in zip(CAMS, _spec_cams()):
        G = ground_to_image_h(cam["K"], cam["pose"])
        np.testing.assert_allclose(G / G[2, 2], SPEC_G2I[name], atol=1e-4)
        I = A @ np.linalg.inv(G)
        np.testing.assert_allclose(I / I[2, 2], SPEC_I2BEV[name], atol=1e-4)


def test_bottom_and_center_pixels():
    expect = {"front": (0.519, 1.422), "left": (0.379, 1.282)}
    for name, cam in zip(CAMS, _spec_cams()):
        if name not in expect:
            continue
        Gi = np.linalg.inv(ground_to_image_h(cam["K"], cam["pose"]))
        axis = 0 if name == "front" else 1
        for (u, v), want in zip([(360, 479.5), (360, 240)], expect[name]):
            g = Gi @ np.array([u, v, 1.0])
            assert abs(g[axis] / g[2] - want) < 0.005


def test_spec_coverage():
    lut = SurroundLUT(_spec_cams(), 720, 480, 500, 0.02, 4.9, 45.0, (0.465, 0.325, 0.34))
    rng, cov = lut.coverage_stats()
    assert rng >= 4.8
    assert cov >= 0.99


# ---------------------------------------------------------------- 2) bizim arac (xacro)
def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @
            np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @
            np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def _urdf():
    xacro = pytest.importorskip("xacro")
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory("igvc_description"), "urdf", "igvc.urdf.xacro")
    except Exception:  # noqa: BLE001
        pytest.skip("igvc_description bulunamadi (workspace source edilmemis)")
    return ET.fromstring(xacro.process_file(path).toxml())


def _frame_in_base(root, child, base="base_footprint"):
    """URDF joint origin'lerini zincirleyip child'in base'e gore 4x4 donusumu."""
    # Yalnizca robotun ust duzey joint'leri (<ros2_control> icindeki <joint>'lerin <child>'i yok)
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    T = np.eye(4)
    while child != base:
        j = joints[child]
        o = j.find("origin")
        xyz = [float(a) for a in o.get("xyz", "0 0 0").split()]
        rpy = [float(a) for a in o.get("rpy", "0 0 0").split()]
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = _rpy(*rpy), xyz
        T = M @ T
        child = j.find("parent").get("link")
    return T


URDF_NAMES = {"front": "front", "back": "rear", "left": "left", "right": "right"}


def test_camera_model_matches_urdf():
    """Dugumun H_g2i'si, URDF optik cercevesiyle yapilan izdusumle ayni pikseli vermeli."""
    root = _urdf()
    v = rg.read_props()
    P = rg.surround_view_params(v)
    K = intrinsic_matrix(*P["front.intrinsics"])
    rng = np.random.default_rng(0)
    for name in CAMS:
        T = _frame_in_base(root, f"{URDF_NAMES[name]}_camera_optical_frame")
        G = ground_to_image_h(K, P[f"{name}.pose"])
        sx, sy, _ = rg.CAMERA_LAYOUT[name]
        # Kameranin onundeki zemin noktalari
        for _ in range(20):
            d = rng.uniform(0.6, 4.0)
            lat = rng.uniform(-1.0, 1.0) * d
            fwd = np.array([sx, sy]) if (sx or sy) else np.array([1.0, 0.0])
            side = np.array([-fwd[1], fwd[0]])
            gx, gy = T[:2, 3] + fwd * d + side * lat * 0.5
            pc = np.linalg.inv(T) @ np.array([gx, gy, 0.0, 1.0])  # optik cercevede
            uv_urdf = (K @ pc[:3])[:2] / pc[2]
            g = G @ np.array([gx, gy, 1.0])
            np.testing.assert_allclose(g[:2] / g[2], uv_urdf, atol=1e-6)


def test_our_vehicle_coverage():
    v = rg.read_props()
    P = rg.surround_view_params(v)
    veh = {k.split(".", 1)[1]: val for k, val in P.items() if k.startswith("vehicle.")}
    body = vehicle_body(veh)
    length, width, cx = rg.footprint(v)
    fp = body["footprint"]
    assert fp[1] - fp[0] == pytest.approx(length) and fp[3] - fp[2] == pytest.approx(width)
    cams = [{"K": intrinsic_matrix(*P[f"{n}.intrinsics"]), "dist": [0.0] * 5,
             "pose": P[f"{n}.pose"]} for n in CAMS]
    lut = SurroundLUT(cams, P["image_width"], P["image_height"], 500, 0.02, 4.9, 45.0, body)
    rng, cov = lut.coverage_stats()
    assert rng >= 4.85
    assert cov >= 0.995
