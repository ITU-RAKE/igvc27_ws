# igvc_vision

RAKE IGVC aracı için algılama paketi (ROS 2 Humble, `ament_cmake` + Python düğümleri).

| Executable | Modül | Ne yapar |
|---|---|---|
| `lidar_map` | `lidar_map.py` | Ön + arka 2D lidardan anlık, araç merkezli occupancy grid |
| `surround_view` | `surround_view_node.py` | 4 kameradan dairesel kuş bakışı (BEV). Homografiler kamera poz/intrinsic'inden hesaplanır. Model yok. |
| `four_cam_test` | `four_cam_test.py` | BEV + YOLO segmentasyon (TensorRT). `bev_mask` ve `/yolo_coordinates` yayınlar. |
| `camera_map` | `camera_map.py` | Kuş bakışında renk filtresi (kırmızı) → engel, gerisi boş: anlık occupancy grid |
| `fused_map` | `fused_map.py` | `lidar_map` + `camera_map` → tek anlık, şişirilmiş (inflation) maliyet haritası (A* için) |

## Klasör yapısı

```
igvc_vision/
├── CMakeLists.txt, package.xml
├── igvc_vision/                  # Python paketi (düğümler birbirinden import edebilir)
│   ├── lidar_map.py
│   ├── surround_view_node.py
│   ├── four_cam_test.py
│   ├── camera_map.py
│   ├── fused_map.py
│   ├── costmap.py                # inflation (A* maliyet haritası)
│   └── robot_geometry.py         # xacro'dan araç/kamera geometrisi (launch'lar kullanır)
├── config/surround_view.yaml     # kamera pozları, intrinsic'ler, BEV ayarları
├── launch/
│   ├── lidar_map.launch.py       # araç ölçülerini igvc_macros.xacro'dan alır
│   ├── surround_view.launch.py   # sadece surround-view
│   ├── four_cam.launch.py        # BEV ve YOLO iki ayrı süreçte
│   ├── camera_map.launch.py      # surround_view + camera_map
│   └── fused_map.launch.py       # lidar_map + surround_view + camera_map + fused_map
├── test/test_surround_geometry.py
└── docs/sistem_ozeti.md          # kamera ve homografi özeti
```

## Derleme

```bash
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` ile `.py` dosyalarındaki değişiklikler yeniden build gerektirmez.

**Yeni düğüm eklemek:** `igvc_vision/<modul>.py` dosyasını yaz (içinde `main()` olsun),
`CMakeLists.txt` içindeki `PYTHON_NODES` listesine `<executable>:<modul>` satırını ekle.

## Çalıştırma

```bash
ros2 launch igvc_vision lidar_map.launch.py
ros2 launch igvc_vision surround_view.launch.py
ros2 launch igvc_vision four_cam.launch.py engine_path:=/yol/best.engine   # engine yoksa YOLO kapalı
ros2 launch igvc_vision camera_map.launch.py        # surround_view'ı da başlatır (with_surround:=false ile kapat)
ros2 launch igvc_vision fused_map.launch.py         # tüm algılama zinciri (with_inputs:=false: yalnızca fused_map)
```

| Topic | Düğüm | İçerik |
|---|---|---|
| `/lidar_map` | lidar_map | OccupancyGrid, `base_footprint`, anlık |
| `/scan_merged` | lidar_map | 360° birleşik LaserScan |
| `/igvc_vision/lidar_map/debug_image` (+`/compressed`) | lidar_map | Harita görüntüsü, araç ve ön yön oku |
| `/igvc_vision/merged_image/compressed` | surround_view | 4 ham görüntü, ortadaki kutunun etrafında |
| `/igvc_vision/merged_image_homo/compressed` | surround_view, four_cam_test | Dairesel kuş bakışı |
| `/igvc_vision/merged_image_homo` | surround_view | Sıkıştırılmamış bgr8 (`publish_raw_homo: true` ise) |
| `/igvc_vision/merged_image_raw/compressed` | four_cam_test | 4 ham görüntü |
| `/igvc_vision/merged_image_homo_model/compressed` | four_cam_test | BEV + YOLO maskesi bindirilmiş |
| `/igvc_vision/bev_mask` | four_cam_test | mono8, 0 = boş, sınıf_id + 1 |
| `/yolo_coordinates` | four_cam_test | Float32MultiArray: [cam_id, x1, x2, y1, y2, ...] |
| `/igvc_vision/camera_map` | camera_map | OccupancyGrid, `base_footprint`, 2 cm: 100 kırmızı, 0 boş, -1 görülmeyen |
| `/igvc_vision/merged_image_homo/pers_mask` | camera_map | mono8: 255 engel, 0 boş, 100 bilinmiyor |
| `/fused_map` | fused_map | OccupancyGrid, `base_footprint`, 5 cm, 12 m: 100 engel/öldürücü, 99..1 maliyet, 0 boş, -1 bilinmiyor |
| `/igvc_vision/fused_map/debug_image` | fused_map | engel: siyah lidar, magenta kamera, koyu mor ikisi; şişirme ısı haritası (JET); açık gri boş, koyu gri bilinmiyor |

> `surround_view` ve `four_cam_test` aynı `merged_image_homo/compressed` topic'ine yayın
> yapar; ikisini aynı anda çalıştırma.

Görüntüler yalnızca abone varken üretilir.

## surround_view nasıl çalışıyor

- Homografiler elle girilmez. Kamera pozları, çözünürlük, intrinsic'ler ve araç gövdesi
  launch dosyasında `igvc_description/urdf/igvc_macros.xacro`'dan hesaplanır
  (`igvc_vision/robot_geometry.py`); URDF'te bir ölçü değişince kuş bakışı da değişir.
  Hesaplanan değerler ve homografiler açılışta log'a yazılır.
- Açılışta tek bir LUT hazırlanır. Her karede 4 görüntü tek bir `cv2.remap` ile
  birleştirilir.
- **Dikiş:** merkezden `seam_angle_deg` (45°) açıyla çıkan simetrik çapraz çizgiler.
  Sahibi olan kamera bir pikseli göremiyorsa komşu kamera doldurur.
- **Gövde:** kesik piramit gövde ve tekerlekler, kameranın zemini görmesini engelleyen
  dışbükey cisimler olarak modellenir; arkalarında kalan zemin siyah (kör nokta) kalır.
  Ayak izi (tekerlekler dahil) gri dikdörtgen ve ileri oku olarak çizilir.
- Çıktı `max_range_m` yarıçaplı daire.

### Önemli parametreler (`config/surround_view.yaml`)

| Parametre | Açıklama |
|---|---|
| `<kamera>.topic` | Kameranın CompressedImage topic'i |
| `<kamera>.dist_coeffs` | `[k1, k2, p1, p2, k3]` (simülasyonda sıfır) |
| `<kamera>.pose`, `<kamera>.intrinsics`, `vehicle.*` | **launch xacro'dan verir**, yaml'a yazma |
| `max_range_m` | Dairesel çıktının yarıçapı |
| `seam_angle_deg` | Dikiş açısı (X ekseninden) |
| `use_vehicle_model` | `true`: xacro'daki gövde + tekerlek. `false`: `body_box_m` kutusu |
| `draw_seams` | Dikiş çizgilerini çiz (kalibrasyon kontrolü için) |

### Gerçek kamerada kalibrasyon

1. Her kamera için satranç tahtasıyla intrinsic ve distorsiyon kalibrasyonu yap
   (`camera_calibration` paketi), değerleri YAML'a yaz.
2. Zemine bilinen konumlarda bant ve işaretler koy, `draw_seams: true` ile dikişlerde
   çizgilerin kopup kopmadığına bak.
3. Kayma varsa ilgili kameranın `pitch` ve `yaw` değerlerinde ince ayar yap.
   Menzil yan kamera eğimine çok hassas: 0,5° ≈ 10–15 cm.

Homografi zemini düz kabul eder. Şerit ve çukur için doğrudur, yüksek nesneler kuş
bakışında uzar. Engeller için lidar haritasını kullan.

## camera_map

- Girdi sıkıştırılmamış `/igvc_vision/merged_image_homo` (JPEG kırmızı kenarlarda sahte piksel üretir).
- Kuş bakışı pikseli = grid hücresi (aynı çözünürlük, ek dönüşüm yok). Hafızasız, her karede yeniden.
- Kırmızı: HSV iki aralık (`hsv_lower1/upper1`, `hsv_lower2/upper2`), `morph_kernel` ile gürültü temizleme.
- Görülmeyen alan (siyah), menzil dışı ve araç ayak izi `-1`. Çözünürlük/menzil `surround_view.yaml`'dan,
  ayak izi xacro'dan (launch).
- ~3 ms/kare; hız `surround_view`'a (~20 Hz) bağlı.
- Yüksek kırmızı nesneler kuş bakışında kameradan uzağa doğru uzar (homografi zemini düz kabul eder).

## fused_map

- Girdi `/lidar_map` (5 cm) ve `/igvc_vision/camera_map` (2 cm; `pers_mask` ile aynı içerik ama
  çözünürlük/orijin taşıdığı için hizalama bundan yapılır). İkisi de `base_footprint`, hafızasız.
- Kural: herhangi biri dolu → dolu; değilse herhangi biri boş → boş; değilse bilinmiyor.
- İnce kaynak kaba hedefe genişletme (dilate) + `cv2.remap` ile iner: blokta tek dolu hücre bloğu
  dolu yapar (şeritler kaybolmaz, engeller en fazla ~1 hücre şişer).
- **Inflation** (`costmap.py`, A* aracı nokta olarak planlar):
  - `lethal_radius_m` = iç teğet yarıçap (yarı genişlik 0.395) + 5 cm = **0.445 m** → 100. Araç
    merkezi buraya girerse gövde engele kesin çarpar.
  - `inflation_radius_m` = çevrel yarıçap √((L/2)² + (W/2)²) = **0.618 m** → 99'dan 0'a doğrusal
    azalır. Araç yerinde dönerken köşesi çarpabilir; pahalı ama geçilebilir (Nav2 ile aynı mantık).
  - İkisi de launch'ta xacro'dan hesaplanır. 2 m'lik şeritte merkezde ~0.76 m bembeyaz koridor
    kalır; öldürücü bölge araca değmez. Siyah bölge araç **merkezinin** giremeyeceği yerdir.
  - Bilinen bir engele yakın bilinmeyen hücre de şişirilir; uzak bilinmeyen -1 kalır.
  - Girdilerde yalnızca 100 engel sayılır: girdi zaten şişirilmişse çift şişirme olmaz.
  - Çalışırken değiştirilebilir (ör. rampada daraltmak): `ros2 param set /fused_map inflation_radius_m 0.5`
- `lidar_map`'te de aynı inflation var ama varsayılan kapalı (`lethal_radius_m: 0`); A* `/fused_map`'i kullanır.
- Eski girdi (`stale_timeout` 0.5 s) katılmaz. Her girdide yayınlar (`max_publish_rate` 30 Hz). ~3 ms/kare.
- Not: kameranın "boş"u "kırmızı değil" demektir. Yüksek ama kırmızı olmayan bir nesnenin kuş
  bakışındaki izdüşümü kamerada boş görünür; nesnenin kendisini lidar dolu işaretler.
- Simülasyonda tüm düğümler `use_sim_time:=true` (launch varsayılanı); gerçek araçta `false` ver.

## four_cam_test

- YOLO için `torch` ve `ultralytics` (pip) gerekir; yoksa ya da `engine_path` boşsa
  yalnızca BEV çalışır.
- `four_cam.launch.py` homografileri ve kamera çözünürlüğünü xacro'dan hesaplayıp
  `homographies` parametresiyle verir. Elle `homography_file` (npz) de verilebilir; ikisi de
  yoksa **tahmini** (kalibre olmayan) değerler kullanılır.

## Test

```bash
sudo apt install python3-pytest     # bir kez
colcon test --packages-select igvc_vision && colcon test-result --verbose
```

`test_costmap.py` inflation maliyet profilini, bilinmeyen hücre davranışını ve çift şişirme
olmadığını kontrol eder. Testler homografi matematiğini `docs/sistem_ozeti.md`'deki referans değerlerle karşılaştırır,
kamera modelinin URDF'teki gerçek TF zinciriyle aynı pikselleri verdiğini ve bizim araç için
kapsamayı (menzil ≥ 4,85 m, kapsama ≥ %99,5) kontrol eder.
