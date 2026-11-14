"""Kamera renk haritasi: ros2 launch igvc_vision camera_map.launch.py [with_surround:=true]

surround_view (kus bakisi) + camera_map (kirmizi -> engel occupancy grid).
BEV cozunurlugu/menzili config/surround_view.yaml'dan, arac ayak izi igvc_macros.xacro'dan
okunur; boylece iki dugum her zaman ayni geometriyi kullanir.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from igvc_vision import robot_geometry


def launch_node(context):
    share = get_package_share_directory('igvc_vision')
    with open(os.path.join(share, 'config', 'surround_view.yaml')) as f:
        sv = yaml.safe_load(f)['surround_view_node']['ros__parameters']
    length, width, cx = robot_geometry.footprint(robot_geometry.read_props())
    return [Node(
        package='igvc_vision',
        executable='camera_map',
        name='camera_map',
        output='screen',
        parameters=[{
            'bev_resolution_m': float(sv['bev_resolution_m']),
            'max_range_m': float(sv['max_range_m']),
            'footprint_m': [cx - length / 2, cx + length / 2, -width / 2, width / 2],
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )]


def generate_launch_description():
    share = get_package_share_directory('igvc_vision')
    return LaunchDescription([
        DeclareLaunchArgument('with_surround', default_value='true',
                              description='surround_view dugumunu de baslat'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Simulasyon saati (gercek aracta false)'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'surround_view.launch.py')),
            condition=IfCondition(LaunchConfiguration('with_surround')),
            launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items(),
        ),
        OpaqueFunction(function=launch_node),
    ])
