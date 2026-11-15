"""geodesy.py testleri."""
import math

import pytest

from igvc_localization.geodesy import (
    EnuFrame, ecef_to_geodetic, geodetic_to_ecef, ll2enu, ll2xy, utm_convergence_deg, xy2ll)

OAKLAND = (42.668, -83.218)    # IGVC sahasi
ITU = (41.1055, 29.0237)       # ITU Ayazaga


def ellipsoid_radii(lat_deg):
    """WGS84: meridyen (M, kuzey-guney) ve dogu-bati (N, prime vertical) egrilik yaricaplari."""
    a, e2 = 6378137.0, 6.69437999014e-3
    s2 = math.sin(math.radians(lat_deg)) ** 2
    return a * (1 - e2) / (1 - e2 * s2) ** 1.5, a / math.sqrt(1 - e2 * s2)


@pytest.mark.parametrize('origin', [OAKLAND, ITU])
def test_ecef_roundtrip(origin):
    lat, lon, alt = ecef_to_geodetic(*geodetic_to_ecef(origin[0], origin[1], 123.4))
    assert lat == pytest.approx(origin[0], abs=1e-9)
    assert lon == pytest.approx(origin[1], abs=1e-9)
    assert alt == pytest.approx(123.4, abs=1e-4)


@pytest.mark.parametrize('origin', [OAKLAND, ITU])
def test_enu_axes_and_scale(origin):
    f = EnuFrame(*origin)
    assert f.to_enu(*origin) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    m, nr = ellipsoid_radii(origin[0])
    d = 1e-4
    e, n, _ = f.to_enu(origin[0] + d, origin[1])             # kuzey
    assert abs(e) < 1e-3 and n == pytest.approx(m * math.radians(d), rel=1e-4)
    e, n, _ = f.to_enu(origin[0], origin[1] + d)             # dogu
    assert abs(n) < 1e-3 and e == pytest.approx(nr * math.cos(math.radians(origin[0])) * math.radians(d), rel=1e-4)


@pytest.mark.parametrize('origin', [OAKLAND, ITU])
def test_enu_roundtrip_100m(origin):
    f = EnuFrame(*origin)
    for e, n in ((100.0, 0.0), (0.0, -100.0), (-70.0, 70.0), (0.01, 0.02)):
        lat, lon, alt = f.to_geodetic(e, n, 0.0)
        e2, n2, u2 = f.to_enu(lat, lon, alt)
        assert (e2, n2, u2) == pytest.approx((e, n, 0.0), abs=1e-6)
    assert ll2enu(*f.to_geodetic(30.0, 40.0)[:2], *origin) == pytest.approx((30.0, 40.0), abs=1e-3)


@pytest.mark.parametrize('origin', [OAKLAND, ITU])
def test_utm_roundtrip_and_convergence(origin):
    x, y = 120.0, -45.0
    lat, lon = xy2ll(x, y, *origin)
    assert ll2xy(lat, lon, *origin) == pytest.approx((x, y), abs=1e-3)
    # UTM ve ENU ayni noktaya ayni mesafede, aralarinda meridyen yakinsamasi kadar aci var
    f = EnuFrame(*origin)
    lat, lon, _ = f.to_geodetic(0.0, 100.0)                   # gercek kuzeye 100 m
    ux, uy = ll2xy(lat, lon, *origin)
    assert math.hypot(ux, uy) == pytest.approx(100.0, rel=1e-3)   # UTM olcek hatasi < %0.1
    grid_angle = math.degrees(math.atan2(ux, uy))              # gercek kuzeyin grid'deki acisi
    assert grid_angle == pytest.approx(-utm_convergence_deg(*origin), abs=0.02)


def test_oakland_convergence_is_significant():
    """IMU yaw'i ile UTM'i karistirmanin neden hata oldugunu belgeler (~1.5 deg)."""
    assert abs(utm_convergence_deg(*OAKLAND)) > 1.0
