"""Algilama zinciri: ros2 launch igvc_vision igvc_vision.launch.py

lidar_map, surround_view, camera_map, fused_map -> /fused_map
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share = os.path.join(get_package_share_directory('igvc_vision'), 'launch')
    sim = {'use_sim_time': LaunchConfiguration('use_sim_time')}

    def include(name, **extra):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, name)),
            launch_arguments={**sim, **extra}.items())

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Simulasyon saati (gercek aracta false)'),
        include('lidar_map.launch.py'),
        include('camera_map.launch.py', with_surround='true'),   # surround_view + camera_map
        include('fused_map.launch.py', with_inputs='false'),     # yalnizca fused_map
    ])
