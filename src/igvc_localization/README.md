# igvc_localization

RAKE IGVC lokalizasyon paketi (ROS 2 Humble, `ament_cmake` + Python). Geçen seneki
`ika_localization`'ın çift EKF mimarisi korunarak yeniden yazıldı (eski paketteki hatalar:
paylaşılan `dt`, Jacobian/model tutarsızlığı, sarılmayan açı, aynı GPS ölçümünün tekrar tekrar
kullanılması, yanlış `map → odom` bileşimi, frame uyumsuzlukları).

> **`sim.launch.py` lokalizasyonu kendisi başlatır** (`ekf:=true`, varsayılan). Ayrıca
> `igvc_localization.launch.py` **açmayın**: iki kopya farklı `odom` çerçeveleri (TF zıplar, RViz'de
> oklar iki konum arasında gider, `path_tracker` şaşırır) ve iki kat GPS ölçümü demek. Düğümler bunu
> algılar: sonradan açılan kopya açıklayıcı bir FATAL mesajıyla kapanır.
> `igvc_localization.launch.py` yalnızca `sim.launch.py ekf:=false` ile ya da gerçek araçta kullanılır.

| Düğüm | Girdi | Çıktı |
|---|---|---|
| `gps_odometry` | `/gps/fix` | `/gps/odometry` (map'te anten ENU konumu), `/gps/datum` (kalıcı) |
| `ekf_local` | `/odom`, `/imu/data` | `/odometry/local`, TF `odom → base_footprint` |
| `ekf_global` | + `/gps/odometry`, mutlak yön | `/odometry/global`, TF `map → odom` |

Ayarlar `config/ekf.yaml`'da. Kontrol ve `path_tracker` sürekli (sıçramasız) `/odometry/local`'i,
GPS waypoint navigasyonu `/odometry/global`'i (ya da TF `map`'i) kullanır.

## Durum

| Aşama | İçerik | Durum |
|---|---|---|
| 1 | Paket iskeleti, `geodesy.py`, simülasyon IMU + GPS, dünya koordinatları | ✅ |
| 2 | `ekf_core.py`, yerel EKF (`odom → base_footprint`) | ✅ |
| 3 | `gps_odometry.py`, global EKF (`map → odom`), launch, config | ✅ |

## Koordinat çerçeveleri (REP-105)

```text
map (ENU, GPS datum'unda) ──global EKF──▶ odom ──yerel EKF──▶ base_footprint
```

- `map`: GPS datum noktasında yerel **ENU** teğet düzlemi (x doğu, y kuzey). `geodesy.EnuFrame`.
- UTM kullanılmaz: UTM grid kuzeyi gerçek kuzeyden meridyen yakınsaması kadar dönüktür
  (Oakland −1.5°, İTÜ +1.3°) ve IMU'nun mutlak yaw'ıyla (BNO085, ENU) tutarsız olurdu.
  `ll2xy` / `xy2ll` (UTM) geçen seneki kodla birebir aynı sonuç verir, geriye uyumluluk için duruyor.

## Yerel EKF (`ekf_node`, `mode: local`)

| | |
|---|---|
| Durum | `[x, y, yaw, v, ω, b_g]`: unicycle (skid steer yana kaymaz), jiroskop sapması dahil |
| Hareket modeli | sabit hız + dönüş hızı, orta nokta entegrasyonu, analitik Jacobian |
| Girdi | `/odom` (igvc_controller tekerlek v, ω), `/imu/data` (jiroskop ω + sapma, yaw) |
| Çıktı | `/odometry/local` (50 Hz), TF `odom → base_footprint` |
| Güncelleme | asenkron sıralı: her mesaj kendi zaman damgasına kadar predict, tek kez kullanılır; açı sarmalı; Joseph formu; χ² aykırı ölçüm kapısı |
| Skid kayması | tekerlek ω std'si = `wheel_w_std + wheel_w_slip·|ω|`: dururken güvenilir (sapmayı öğretir), dönerken zayıf (jiroskop baskın) |

Ayarlar: `config/ekf.yaml`. `sim.launch.py ekf:=true` (varsayılan) EKF'yi başlatır ve
igvc_controller'ın TF'ini kapatır (`publish_tf:=false`); `/odom` ham tekerlek odometrisi olarak kalır.

Simülasyon doğrulaması (2 kare tur, 8 yerinde dönüş + viraj, Gazebo gerçek konumuna göre):

| | Son konum / yaw hatası |
|---|---|
| Ham tekerlek odometrisi | 0.33 m / 9.1° |
| EKF, IMU yaw açık (varsayılan) | **0.026 m / 0.0°** |
| EKF, yalnızca jiroskop (`use_imu_yaw: false`) | 0.16 m / 4.3° |

Not: `odom` çerçevesi sürekli olmalı (REP-105). Gerçek araçta BNO085'in mutlak yaw'ı motorlara yakın
manyetik bozulmayla sıçrıyorsa `imu_yaw_std`'yi büyütün ya da `use_imu_yaw: false` kullanın.

## Global EKF (`ekf_node`, `mode: global`)

- `map`: GPS datum'unda ENU. `gps_odometry` `datum` parametresi; simülasyonda dünya dosyasının
  `spherical_coordinates` noktası, yani `map` = Gazebo dünya çerçevesi.
- Ölçümler: yereldekiler + GPS konumu (NavSatFix kovaryansı, χ² kapısı) + mutlak yön:
  - `heading_source: imu` (varsayılan): BNO085 mutlak yaw. **Gerçek araçta `imu_yaw_offset_deg`
    manyetik sapma** (BNO085 manyetik kuzey verir; Oakland ~ −7°).
  - `heading_source: gps_course`: GPS hız vektörünün yönü (ileri ve > 0.4 m/s); IMU yaw
    güvenilmezse. Araç hareket edene kadar global EKF başlamaz.
- GPS anten ofseti TF'ten (`base_footprint → gps_link`) okunup düzeltilir.
- İlk GPS fix'i ve mutlak yön gelene kadar başlamaz ve yayınlamaz.
- `map → odom` = `T_map_base · T_odom_base⁻¹` (yerel poz TF'ten, aynı zaman damgasıyla).
- **Anten ofseti eğim düzeltmesi:** anten ~1 m yüksekte; araç eğilince (rampa %15 → ~10°) anten
  yatayda ~17 cm kayar. 3B ofset TF'ten okunur, her GPS ölçümünde IMU roll/pitch ile yataya
  izdüşürülür. Rampada global hata 15–17 cm'den 0.2–1.4 cm'ye indi.
- **GPS kilitlenmesinden kurtarma:** χ² kapısı tek tek GPS sıçramalarını reddeder; ama filtre GPS'ten
  bir kez koparsa (tekerlek patinajı, çarpışma, kaçırılma) doğru GPS'i de sonsuza dek reddedebilir.
  GPS `gps_reset_after` (10 → 1 s) kez üst üste reddedilirse konum GPS'e yeniden kilitlenir.
  Doğrulama: çalışırken araç Gazebo'da 5.4 m ışınlandı → global EKF ~1 s'de 1–2 cm hataya döndü.

Simülasyon doğrulaması: başlangıç (3, −2, 45.8°); 2 kare tur, 8 yerinde dönüş + viraj;
`robot_localization` (ekf_node + navsat_transform, aynı sensörler) ile aynı koşuda:

| | Son hata | En büyük | Ortalama |
|---|---|---|---|
| Ham tekerlek odometrisi | 0.246 m / 7.2° | 0.246 m / 7.2° | 0.097 m |
| Yerel EKF | 0.012 m / 0.0° | 0.033 m / 1.3° | 0.014 m |
| **Global EKF** | **0.007 m / 0.0°** | **0.023 m / 1.1°** | **0.010 m** |
| robot_localization | 0.027 m / 0.0° | 0.129 m / 5.9° | 0.042 m |
| Global EKF, `gps_course` | 0.004 m / 0.5° | 0.030 m / 1.3° | 0.010 m |

Başlangıçta global EKF (2.994, −2.000, 45.8°) buldu, yerel EKF (0, 0, 0)'dan başladı.
TF `map → base_footprint` = `/odometry/global`.

## GPS waypoint → map

```python
from igvc_localization.geodesy import EnuFrame
# /gps/datum mesajindan (kalici topic) ya da config'teki datum'dan
enu = EnuFrame(datum.latitude, datum.longitude)
x, y, _ = enu.to_enu(wp_lat, wp_lon)      # map cercevesinde hedef [m]
```

## Sensörler (simülasyon, `igvc_description`)

| Sensör | Topic | Frame | Hız | Gürültü |
|---|---|---|---|---|
| IMU (BNO085) | `/imu/data` | `imu_link` (gövde, zeminden 15 cm) | 100 Hz | jiro 0.002 rad/s + sapma, ivme 0.02 m/s², **mutlak yaw** |
| GPS (RTK) | `/gps/fix`, `/gps/vel` | `gps_link` (direk tepesi, zeminden ~1 m) | 10 Hz | yatay 2 cm, dikey 5 cm |

Konum ve gürültüler `igvc_macros.xacro`'da (`imu_*`, `gps_*`).

## Dünya dosyaları

GPS için dünyada `<spherical_coordinates>` olmalı. `empty.world` ve `parkur.world`: IGVC sahası
(Oakland Üniversitesi, 42.668 N, −83.218 W), `heading_deg` = **180**.

`heading_deg` 180 olmalı: Gazebo Classic'te `heading_deg` 0 iken GPS ENU ekseni dünya ekseninin 180°
dönüğü çıkıyor (ölçüldü: dünyada +x'e 10 m giden araç GPS'te −10 m doğuda). 180 ile GPS = dünya ENU,
IMU yaw'ı etkilenmez. Böylece kodda elle `yaw_offset` yok; simülasyon ve gerçek araç aynı.
Yeni bir dünya kaydederken bu bloğu kopyalayın.

## Performans

Python çekirdek (iki EKF + GPS düğümü birlikte birkaç % CPU): predict ~20 µs, ölçüm güncellemesi ~50 µs; 100 Hz IMU + 100 Hz odom ≈ %1–3 CPU.
`joint_state_broadcaster` 100 Hz'e sınırlandı (controller_manager 1 kHz'de varsayılan 1000 Hz
yayınlıyordu, igvc_controller ~%100 CPU yiyordu).

## Doğruluk testi (simülasyon)

`/ground_truth/odom`: Gazebo'nun gerçek pozu (URDF'te `p3d`, yalnızca simülasyon), `map` ile aynı çerçeve.

```bash
ros2 launch igvc_controller sim.launch.py                         # 1. terminal
ros2 run igvc_localization localization_eval --ros-args -p use_sim_time:=true   # 2. terminal
ros2 run igvc_controller keyboard_controller                      # 3. terminal: sür, Ctrl+C ile özet
```

`localization_eval` 2 s'de bir: global EKF hatası (+ kendi σ'sı), yerel EKF ve ham odom'un
değerlendirme başından beri kayması; Ctrl+C'de ortalama / RMS / en büyük hata özeti.

| Test | Nasıl | Beklenen |
|---|---|---|
| Durağan | araç 60 s dursun | global < 2 cm, yerel kayma ~0 |
| Tur | kare/daire çizip başlangıca dön | global birkaç cm; yerel ham odom'dan çok iyi; ham odom dönüşlerde kayar |
| Kaçırma | Gazebo'da aracı sürükle (Translate) | global ~1 s'de toparlanır ("yeniden kilitleniyor"); yerel toparlanmaz (doğru: GPS'siz) |
| Tutarlılık | global hata vs σ | hata çoğunlukla < 3σ; sürekli üstündeyse gürültü parametreleri fazla iyimser |
| GPS gürültüsü | `igvc_macros.xacro` `gps_h_std` 1.0 (RTK'sız) | global sıçramaz, GPS'ten pürüzsüz |

RViz: Fixed Frame `map`; Odometry display'leri `/ground_truth/odom`, `/odometry/global`,
`/odometry/local` (farklı renkler); TF'te `map → odom → base_footprint` zinciri.

## Test

```bash
colcon test --packages-select igvc_localization && colcon test-result --verbose
```
