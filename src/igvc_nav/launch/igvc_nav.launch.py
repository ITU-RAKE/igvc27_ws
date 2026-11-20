"""Navigasyon: ros2 launch igvc_nav igvc_nav.launch.py

path_planner (/fused_map -> /igvc_nav/planned_path) + path_tracker (-> /cmd_vel).
/fused_map icin once: ros2 launch igvc_vision igvc_vision.launch.py
Standart nav modu: standart_mode:=true (calisirken tek komut, tracker planner'i izler:
  ros2 param set /path_planner standart_mode true|false)
Waypoint modu: waypoint_mode:=true (config/waypoints.yaml; calisirken ros2 param set /path_planner waypoint_mode true)
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
    std = LaunchConfiguration('standart_mode')
    wpm = LaunchConfiguration('waypoint_mode')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Simulasyon saati (gercek aracta false)'),
        DeclareLaunchArgument('standart_mode', default_value='true',
                              description='Standart nav modu: planner + tracker birlikte (iz/frontier korunur)'),
        DeclareLaunchArgument('waypoint_mode', default_value='true',
                              description='GPS waypoint takibi (config/waypoints.yaml)'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav, 'path_planner.launch.py')),
            launch_arguments={'with_maps': 'false', 'use_sim_time': sim_time, 'standart_mode': std,
                              'waypoint_mode': wpm}.items()),
        Node(package='igvc_nav', executable='path_tracker', name='path_tracker',
             output='screen', parameters=[os.path.join(get_package_share_directory('igvc_nav'), 'config', 'path_tracker.yaml'),
                         {'use_sim_time': sim_time, 'standart_mode': std}]),
    ])
