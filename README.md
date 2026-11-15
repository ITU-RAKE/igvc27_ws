```text
1. igvc_controller/launch/sim.launch.py
   ├── gzserver / gzclient            (gazebo_ros)
   ├── robot_state_publisher
   ├── spawn_entity.py                (gazebo_ros)
   ├── spawner ×2                     (controller_manager)
   ├── rviz2
   ├── controller.py
   │     skid steer kinematiği, cmd_vel→tekerlek, enkoder→odom
   └── igvc_localization.launch.py    (ekf:=true, varsayılan)
         ├── gps_odometry.py
         │     GPS (lat/lon) → yerel metrik konum
         ├── ekf_node.py (ekf_local)
         │     odom + IMU → /odometry/local, TF odom→base_footprint
         └── ekf_node.py (ekf_global)
               + GPS + mutlak yön → /odometry/global, TF map→odom

2. igvc_vision/launch/igvc_vision.launch.py
   ├── lidar_map.py
   │     iki 2D lidar, anlık araç merkezli occupancy grid
   ├── surround_view_node.py
   │     dört kamera homografisi, 360° kuş bakışı
   ├── camera_map.py
   │     kuş bakışında beyaz şerit filtresi, kamera occupancy grid
   └── fused_map.py
         lidar+kamera birleştirme, inflation, A* maliyet haritası

3. igvc_nav/launch/igvc_nav.launch.py
   ├── path_planner.py
   │     hedef seçimi (BFS), maliyetli A*, geri çekilme
   └── path_tracker.py
         pure pursuit yol takibi, cmd_vel üretimi

```
