"""Navigasyon: ros2 launch ika_nav igvc_nav.launch.py

path_planner (/fused_map -> /igvc_nav/planned_path) + path_tracker (-> /cmd_vel).
/fused_map icin once: ros2 launch igvc_vision igvc_vision.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    nav = os.path.join(get_package_share_directory('ika_nav'), 'launch')
    sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Simulasyon saati (gercek aracta false)'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav, 'path_planner.launch.py')),
            launch_arguments={'with_maps': 'false', 'use_sim_time': sim_time}.items()),
        Node(package='ika_nav', executable='path_tracker', name='path_tracker',
             output='screen', parameters=[{'use_sim_time': sim_time}]),
    ])
