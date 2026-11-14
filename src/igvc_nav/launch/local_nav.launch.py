"""Otonom yerel surus: ros2 launch igvc_nav local_nav.launch.py [with_maps:=true]

path_planner (/fused_map -> /igvc_nav/planned_path) + path_tracker (-> /cmd_vel).
with_maps:=true (varsayilan) igvc_vision harita zincirini de baslatir
(lidar_map, surround_view, camera_map, fused_map).
Arac ve kontrolcu: ros2 launch igvc_controller sim.launch.py (ayri terminalde).
Klavye surusunu (/cmd_vel) ayni anda calistirma.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    nav = os.path.join(get_package_share_directory('igvc_nav'), 'launch')
    sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument('with_maps', default_value='true',
                              description='igvc_vision harita zincirini de baslat'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Simulasyon saati (gercek aracta false)'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav, 'path_planner.launch.py')),
            launch_arguments={'with_maps': LaunchConfiguration('with_maps'),
                              'use_sim_time': sim_time}.items()),
        Node(
            package='igvc_nav',
            executable='path_tracker',
            name='path_tracker',
            output='screen',
            parameters=[{'use_sim_time': sim_time}],
        ),
    ])
