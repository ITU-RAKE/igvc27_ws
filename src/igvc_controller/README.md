# igvc_controller

Skid steer araç kontrolcüsü ve klavye sürüşü (ROS 2 Humble, `ament_cmake` + Python).

| Executable | Ne yapar |
|---|---|
| `controller` | `/cmd_vel` → tekerlek hızları (ters kinematik, hız/ivme sınırı, zaman aşımı); enkoder (`/joint_states`) → `/odom` + TF `odom → base_footprint` |
| `keyboard_controller` | WASD → `/cmd_vel` (kendi terminalinde çalışır) |

```
klavye ─/cmd_vel─▶ controller ─/wheel_velocity_controller/commands─▶ ros2_control (Gazebo) ─▶ tekerlekler
                      ▲                                                       │
                      └──────────────── /joint_states (enkoder) ◀─────────────┘
                   controller ─▶ /odom + TF odom → base_footprint   (tek odom kaynağı)
```

## Çalıştırma

```bash
ros2 launch igvc_controller sim.launch.py            # Gazebo + ros2_control + controller
ros2 launch igvc_controller sim.launch.py gui:=false rviz:=true
ros2 run igvc_controller keyboard_controller          # ayrı terminalde
ros2 run igvc_controller keyboard_controller --ros-args -p mode:=hold
```

## Klavye

| Tuş | `increment` (varsayılan) | `hold` |
|---|---|---|
| W / S | ileri / geri hızı bir adım değiştir (0.1 m/s) | basılı tutunca ileri / geri |
| A / D | sola / sağa dönüş hızı bir adım (0.2 rad/s) | basılı tutunca yerinde dön |
| Boşluk | dur | dur |
| Q / E | — | hız çarpanı − / + |
| Ctrl+C | dur komutu gönderip çık | aynı |

Terminal tuş bırakmayı bildirmez ve aynı anda iki tuş okuyamaz: viraj için `increment` modunda
W ile hızlanıp A/D ile eğrilik ver. `hold` modunda tuş bırakılınca `hold_timeout` (0.6 s) sonra durur.

## Kinematik (skid steer)

- Etkin iz `B = wheel_separation × track_scale`
- Ters: `ω_sol = (v − ω·B/2) / r`, `ω_sağ = (v + ω·B/2) / r`
- İleri (odom): `ds = r(Δθ_sol + Δθ_sağ)/2`, `Δψ = r(Δθ_sağ − Δθ_sol)/B`, orta nokta entegrasyonu
- `track_scale`: simülasyon (2 çeker + caster) 1.0. Gerçek skid steer'de yanal kayma yüzünden daha
  büyük (tipik 1.2–2.0): aracı yerinde 360° döndür, odom yaw'ını gerçekle karşılaştırıp ayarla.
- Taraf başına birden fazla tekerlek: `left_wheel_joints` / `right_wheel_joints` listeleri.
- Tekerlek yarıçapı, iz ve azami tekerlek hızı launch'ta `igvc_macros.xacro`'dan okunur.

## Güvenlik

- `max_linear/angular_velocity`, `max_linear/angular_acceleration` (gerçek geçen süreyle)
- Tekerlek hızı doygunluğunda iki taraf birlikte ölçeklenir (eğrilik korunur)
- `cmd_timeout` (0.5 s) boyunca komut gelmezse durur; kapanırken sıfır komut gönderir

## Gerçek araç

Yalnızca "tekerlek hızı gönder / enkoder oku" katmanı değişir: motor sürücüsü
`/wheel_velocity_controller/commands`'ı dinleyip `/joint_states` yayınlarsa (ya da topic adları
parametreden değiştirilirse) kinematik ve odometri aynen çalışır. `use_sim_time:=false` ver.
