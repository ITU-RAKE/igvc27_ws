"""
Cografi koordinat donusumleri (ROS'tan bagimsiz, test edilebilir).

EKF ve GPS odometrisi YEREL ENU teget duzlemini kullanir:
  enlem/boylam/yukseklik -> ECEF -> datum noktasinda ENU (x dogu, y kuzey, z yukari).
  Kesin donusum, meridyen yakinsamasi yok: x ekseni GERCEK doguya bakar. IMU'nun mutlak
  yaw'i (BNO085, REP-145 ENU) ile ayni eksen -> konum ve yon tutarli.

UTM fonksiyonlari (ll2xy / xy2ll) gecen seneki koddan geriye uyumluluk icin korundu.
UTM grid kuzeyi gercek kuzeyden meridyen yakinsamasi kadar donuktur (Oakland'da ~1.5 deg);
IMU yaw'iyla birlikte kullanilacak konumlar icin ll2enu tercih edilmeli.
"""
import math

# WGS84
WGS84_A = 6378137.0                     # buyuk yari eksen [m]
WGS84_F = 1.0 / 298.257223563           # basiklik
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)    # birinci eksantriklik^2

# UTM
UTM_K0 = 0.9996


# ============================================================ yerel ENU (EKF bunu kullanir)
def geodetic_to_ecef(lat_deg, lon_deg, alt=0.0):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
    return ((n + alt) * cl * math.cos(lon),
            (n + alt) * cl * math.sin(lon),
            (n * (1.0 - WGS84_E2) + alt) * sl)


def ecef_to_geodetic(x, y, z):
    """Bowring yontemi + iki Newton adimi (mm altinda hata)."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(5):
        sl = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
        alt = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    sl = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
    alt = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), alt


class EnuFrame:
    """Datum (enlem, boylam, yukseklik) noktasinda yerel ENU cercevesi."""

    def __init__(self, lat0_deg, lon0_deg, alt0=0.0):
        self.lat0, self.lon0, self.alt0 = lat0_deg, lon0_deg, alt0
        self.x0, self.y0, self.z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0)
        lat, lon = math.radians(lat0_deg), math.radians(lon0_deg)
        sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
        # ECEF -> ENU donme matrisi (satirlar: dogu, kuzey, yukari)
        self.r = ((-so, co, 0.0),
                  (-sl * co, -sl * so, cl),
                  (cl * co, cl * so, sl))

    def to_enu(self, lat_deg, lon_deg, alt=None):
        """(enlem, boylam[, yukseklik]) -> (dogu, kuzey, yukari) [m]."""
        x, y, z = geodetic_to_ecef(lat_deg, lon_deg, self.alt0 if alt is None else alt)
        d = (x - self.x0, y - self.y0, z - self.z0)
        return tuple(sum(self.r[i][k] * d[k] for k in range(3)) for i in range(3))

    def to_geodetic(self, e, n, u=0.0):
        """(dogu, kuzey, yukari) [m] -> (enlem, boylam, yukseklik)."""
        r = self.r
        dx = r[0][0] * e + r[1][0] * n + r[2][0] * u
        dy = r[0][1] * e + r[1][1] * n + r[2][1] * u
        dz = r[0][2] * e + r[1][2] * n + r[2][2] * u
        return ecef_to_geodetic(self.x0 + dx, self.y0 + dy, self.z0 + dz)


def ll2enu(lat, lon, origin_lat, origin_lon):
    """Kisayol: datuma gore (dogu, kuzey) [m]. Cok cagriliyorsa EnuFrame kullan."""
    e, n, _ = EnuFrame(origin_lat, origin_lon).to_enu(lat, lon)
    return e, n


# ============================================================ UTM (geriye uyumluluk)
def utm_zone(lat, lon):
    lon = (lon + 180.0) % 360.0 - 180.0
    zone = int((lon + 180.0) / 6.0) + 1
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        zone = 32
    if 72.0 <= lat < 84.0:
        for lo, hi, z in ((0, 9, 31), (9, 21, 33), (21, 33, 35), (33, 42, 37)):
            if lo <= lon < hi:
                zone = z
    return zone


def ll_to_utm(lat, lon, zone=None):
    """(enlem, boylam) -> (dogu, kuzey, zon). USGS Bulletin 1532 (gecen seneki kodla ayni)."""
    if zone is None:
        zone = utm_zone(lat, lon)
    lon = (lon + 180.0) % 360.0 - 180.0
    e2 = WGS84_E2
    ep2 = e2 / (1.0 - e2)
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    lon0 = math.radians((zone - 1) * 6.0 - 180.0 + 3.0)
    n = WGS84_A / math.sqrt(1.0 - e2 * math.sin(lat_r) ** 2)
    t = math.tan(lat_r) ** 2
    c = ep2 * math.cos(lat_r) ** 2
    a = math.cos(lat_r) * (lon_r - lon0)
    m = WGS84_A * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat_r
                   - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * lat_r)
                   + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * lat_r)
                   - (35 * e2 ** 3 / 3072) * math.sin(6 * lat_r))
    east = UTM_K0 * n * (a + (1 - t + c) * a ** 3 / 6
                         + (5 - 18 * t + t * t + 72 * c - 58 * ep2) * a ** 5 / 120) + 500000.0
    north = UTM_K0 * (m + n * math.tan(lat_r) * (
        a * a / 2 + (5 - t + 9 * c + 4 * c * c) * a ** 4 / 24
        + (61 - 58 * t + t * t + 600 * c - 330 * ep2) * a ** 6 / 720))
    if lat < 0:
        north += 10000000.0
    return east, north, zone


def utm_to_ll(east, north, zone, northern=True):
    e2 = WGS84_E2
    ep2 = e2 / (1.0 - e2)
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    x = east - 500000.0
    y = north if northern else north - 10000000.0
    lon0 = (zone - 1) * 6.0 - 180.0 + 3.0
    mu = y / UTM_K0 / (WGS84_A * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * math.sin(6 * mu))
    n1 = WGS84_A / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    t1 = math.tan(phi1) ** 2
    c1 = ep2 * math.cos(phi1) ** 2
    r1 = WGS84_A * (1 - e2) / (1 - e2 * math.sin(phi1) ** 2) ** 1.5
    d = x / (n1 * UTM_K0)
    lat = phi1 - (n1 * math.tan(phi1) / r1) * (
        d * d / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 * c1 - 9 * ep2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 * t1 - 252 * ep2 - 3 * c1 * c1) * d ** 6 / 720)
    lon = (d - (1 + 2 * t1 + c1) * d ** 3 / 6
           + (5 - 2 * c1 + 28 * t1 - 3 * c1 * c1 + 8 * ep2 + 24 * t1 * t1) * d ** 5 / 120) / math.cos(phi1)
    return math.degrees(lat), lon0 + math.degrees(lon)


def ll2xy(lat, lon, origin_lat, origin_lon):
    """Gecen seneki API: datuma gore UTM (dogu, kuzey) farki [m]. Ayni zon zorunlu."""
    zone = utm_zone(origin_lat, origin_lon)
    e0, n0, _ = ll_to_utm(origin_lat, origin_lon, zone)
    e, n, _ = ll_to_utm(lat, lon, zone)
    return e - e0, n - n0


def xy2ll(x, y, origin_lat, origin_lon):
    zone = utm_zone(origin_lat, origin_lon)
    e0, n0, _ = ll_to_utm(origin_lat, origin_lon, zone)
    return utm_to_ll(e0 + x, n0 + y, zone, northern=origin_lat >= 0)


def utm_convergence_deg(lat, lon):
    """Meridyen yakinsamasi: UTM grid kuzeyi ile gercek kuzey arasi aci [deg]."""
    zone = utm_zone(lat, lon)
    lon0 = (zone - 1) * 6.0 - 180.0 + 3.0
    return math.degrees(math.atan(math.tan(math.radians(lon - lon0)) * math.sin(math.radians(lat))))
