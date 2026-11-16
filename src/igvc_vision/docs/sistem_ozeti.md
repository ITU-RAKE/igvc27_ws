# BU ARAÇ (igvc27_ws) — GÜNCEL DEĞERLER

> Aşağıdaki "SİSTEM ÖZETİ" bölümü paketin geldiği **referans aracı** (Muharrem) anlatır ve
> testlerde homografi matematiğini doğrulamak için tutulur. **Bizim aracın değerleri elle
> girilmez**: `surround_view.launch.py` / `four_cam.launch.py` hepsini
> `igvc_description/urdf/igvc_macros.xacro`'dan hesaplar (`igvc_vision/robot_geometry.py`).
> Aşağıdaki tablo mevcut xacro ile hesaplanmış durumdur; URDF değişirse bu tablo eskir,
> güncel değerler düğümün açılış log'undadır.

| | Bizim araç | Referans araç |
|---|---|---|
| Koordinat orijini | `base_footprint` (zeminde, aks altında) | direk merkezinin zemin izdüşümü |
| Ön/arka kamera | x = ±0.325 | x = ±0.35 |
| Sol/sağ kamera | y = ±0.200 | y = ±0.21 |
| Kamera yüksekliği / eğim | 1.00 m / 43° | 1.0 m / 43° |
| Çözünürlük | 640×427 (simde 1280×854 idi; kamera hızı için düşürüldü) | 720×480 |
| Intrinsic fx = fy, cx, cy | 278.1718, 319.5, 213.0 | 312.94, 360, 240 |
| Gövde modeli | kesik piramit (taban 0.95×0.65, tavan 0.85×0.6, zeminden 0.10–0.40 m) + 2 tekerlek | kutu 0.93×0.65×0.34 |
| Ayak izi (tekerlek dahil) | 0.950 × 0.790 m | 0.93 × 0.65 m |
| Kesintisiz menzil / kapsama (4.9 m) | 4.90 m / %99.9 | 4.88 m / %99.7 |

Simülasyonda zemine çizilen 0.5 m'lik ızgarayla doğrulandı: dört sektörde ve dikişlerde
çizgi konum hatası ortalama < 0.3 cm (1 px = 2 cm), ölçümlerin %99.9'u ±2 cm içinde.

Gazebo `camera_info` cx/cy'yi 1-tabanlı ((w+1)/2 = 640.5) yayınlar; burada 0-tabanlı
(OpenCV) merkez (w−1)/2 = 639.5 kullanılır (fark ~0.1°, kuş bakışında önemsiz).

### H_i2bev (500×500, 2 cm/px), bizim araç
ön:
[[   0.740708,    2.708597, -223.682586],
 [   0.      ,    3.037699, -283.083394],
 [   0.      ,    0.010834,    1.      ]]
arka:
[[ -0.740708,   2.708597, 723.682586],
 [ -0.      ,   2.379494, 783.083394],
 [  0.      ,   0.010834,   1.      ]]
sol:
[[   0.      ,    3.105414, -276.833394],
 [  -0.740708,    2.708597,  723.682586],
 [   0.      ,    0.010834,    1.      ]]
sağ:
[[   0.      ,    2.311779,  776.833394],
 [   0.740708,    2.708597, -223.682586],
 [  -0.      ,    0.010834,    1.      ]]

---

# IGVC 4 KAMERA SURROUND-VIEW / BEV — SİSTEM ÖZETİ

## Araç
- Şasi: 93 cm (X, uzunluk) × 65 cm (Y, genişlik); kenarlar orijinden X=±0.465 m, Y=±0.325 m
- Gövde üst yüksekliği ~34 cm (ölçülmedi, 8.2 cm'lik kör alandan geri hesaplandı)
- Kameralar 1 m'lik orta direğin tepesinde, kollarla dışarı açılmış

## Koordinat sistemi
- Orijin: direk merkezinin zemindeki izdüşümü (base_link zeminde)
- X ileri, Y sol, Z yukarı, birim metre
- Kamera optik çerçevesi (OpenCV): x sağ, y aşağı, z ileri

## Kamera montajı (dört kamera aynı eğimde)
| Kamera | x | y | z | yaw | pitch (aşağı) |
| ön   | +0.35 | 0     | 1.0 | 0°    | 43° |
| arka | -0.35 | 0     | 1.0 | 180°  | 43° |
| sol  | 0     | +0.21 | 1.0 | +90°  | 43° |
| sağ  | 0     | -0.21 | 1.0 | -90°  | 43° |

## Kamera
- FOV: 98° yatay × 75° dikey
- Çözünürlük: 720×480 (3:2). 98°×75° kare pikselle ancak 3:2'de tutuyor (tan49/tan37.5 = 1.499)
- Intrinsic K (ideal pinhole, distorsiyon yok):
  [[312.943226,   0.0,      360.0],
   [  0.0,      312.774089, 240.0],
   [  0.0,        0.0,        1.0]]

## Gazebo SDF (pose: x y z roll pitch yaw, rad; + pitch = aşağı)
- ön:   0.35  0    1.0  0 0.75049 0
- arka: -0.35 0    1.0  0 0.75049 3.14159
- sol:  0     0.21 1.0  0 0.75049 1.5708
- sağ:  0    -0.21 1.0  0 0.75049 -1.5708
- horizontal_fov = 1.71042 (98°), width = 720, height = 480, update_rate = 30, clip near 0.05 / far 50

## Kapsama (hesaplanan)
- Kesintisiz 360° menzil: ~492 cm (sınırı ~44.5° çapraz dikiş belirliyor)
- Gövde kenarından itibaren kör alan: ön/arka ~6 cm, yanlar ~6 cm, köşeler ~10 cm
- Kontrol: ön kameranın alt-orta pikseli zeminde x = 0.519 m'ye, merkez pikseli x = 1.422 m'ye düşüyor
  (yan kameralarda y = 0.379 m ve 1.282 m)
- Menzil yan kamera eğimine çok hassas: 0.5° fark ≈ 10–15 cm menzil değişimi

## BEV canvas
- 10 m × 10 m, 2 cm/px → 500×500 px, araç merkezde (piksel 250,250)
- BEV yönü: yukarı = +X (ileri), sağ = -Y
- Zemin → BEV dönüşümü A:
  [[ 0, -50, 250],
   [-50,  0, 250],
   [ 0,   0,   1]]

## Homografiler (teorik, ideal pinhole + kusursuz montaj)
### H_i2bev: görüntü pikseli → BEV pikseli (cv2.warpPerspective / remap LUT için)
ön:
[[ 1.322508, 4.83872,  -226.102798],
 [ 0.0,      5.402445, -286.769854],
 [ 0.0,      0.019355,    1.0     ]]
arka:
[[-1.322508, 4.83872,   726.102798],
 [ 0.0,      4.274994,  786.769854],
 [ 0.0,      0.019355,    1.0     ]]
sol:
[[ 0.0,      5.53793,  -279.769854],
 [-1.322508, 4.83872,   726.102798],
 [ 0.0,      0.019355,    1.0     ]]
sağ:
[[ 0.0,      4.13951,   779.769854],
 [ 1.322508, 4.83872,  -226.102798],
 [ 0.0,      0.019355,    1.0     ]]

### H_g2i: zemin (X, Y, 1) [m] → görüntü pikseli
ön:
[[ 618.009745, -734.566153, 360.0     ],
 [ -88.695655,    0.0,      952.183074],
 [   1.716694,    0.0,        1.0     ]]
arka:
[[-618.009745,  734.566153, 360.0     ],
 [  88.695655,    0.0,      952.183074],
 [  -1.716694,    0.0,        1.0     ]]
sol:
[[ 592.231047,  498.259492, 360.0     ],
 [   0.0,       -71.509312, 757.66956 ],
 [   0.0,         1.384054,   1.0     ]]
sağ:
[[-592.231047, -498.259492, 360.0     ],
 [   0.0,        71.509312, 757.66956 ],
 [   0.0,        -1.384054,   1.0     ]]

## Zeminde piksel başına çözünürlük, 720×480 (ileri / yanal)
- 1 m: 0.4 / 0.4 cm | 2 m: 1.1 / 0.6 | 3 m: 2.2 / 0.8 | 4 m: 3.6 / 1.1 | 4.9 m: 5.2 / 1.3
- Uzakta ileri yöndeki çözünürlük düşüyor: 7.6 cm'lik şerit çizgisi 4.9 m'de ~1.5 px'e iniyor
  (yana uzanan çizgiler için riskli). 960×640'ta ~2 px, 1200×800'de ~2.5 px.

## Notlar
- BEV maliyeti kamera çözünürlüğüne değil canvas boyutuna bağlı (önceden hesaplanmış LUT + cv2.remap).
- Her BEV pikseli tek bir kamera sektörüne atanmalı (bitwise_or ile birleştirme yapma).
- Simülasyonda bu matrisler doğrudan doğru çıkmalı. Dikişler kayıyorsa SDF pozlarına ve frame tanımına bak.
- Gerçek kamerada: önce satranç tahtasıyla intrinsic + distorsiyon kalibrasyonu, undistort,
  sonra zemindeki bilinen noktalarla cv2.findHomography ile matrisi düzelt. Bu değerler başlangıç noktası.
- Homografi zemini düz varsayar: şerit ve çukur için doğru, yüksek nesneler BEV'de uzar.
  Engel için ayrı bir derinlik kaynağı (LiDAR/stereo) kullan.