"""Yerel yol planlayici: ros2 launch igvc_nav path_planner.launch.py [with_maps:=false]

standart_mode:=true standart nav modunda baslatir (calisirken: ros2 param set /path_planner standart_mode true).
Girdi /fused_map (igvc_vision). with_maps:=true ise igvc_vision fused_map zincirini de baslatir
(lidar_map, surround_view, camera_map, fused_map). Arac ayak izi igvc_macros.xacro'dan,
planlama yaricapi (kamera menzili) igvc_vision/config/surround_view.yaml'dan.
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
    v = robot_geometry.read_props()
    sv_yaml = os.path.join(get_package_share_directory('igvc_vision'), 'config', 'surround_view.yaml')
    with open(sv_yaml) as f:
        cam_range = float(yaml.safe_load(f)['surround_view_node']['ros__parameters']['max_range_m'])
    return [Node(
        package='igvc_nav',
        executable='path_planner',
        name='path_planner',
        output='screen',
        parameters=[{
            'map_topic': '/fused_map',
            'footprint_m': robot_geometry.footprint_rect(v),
            'plan_radius_m': cam_range - 0.1,       # kamera dairesinin kenar karisimini disarida birak
            'standart_mode': LaunchConfiguration('standart_mode'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )]


def generate_launch_description():
    vision = os.path.join(get_package_share_directory('igvc_vision'), 'launch')
    return LaunchDescription([
        DeclareLaunchArgument('with_maps', default_value='false',
                              description='igvc_vision fused_map zincirini de baslat'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Simulasyon saati (gercek aracta false)'),
        DeclareLaunchArgument('standart_mode', default_value='false',
                              description='Standart nav modu (standart engele yaklasma; iz/frontier korunur)'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(vision, 'fused_map.launch.py')),
            condition=IfCondition(LaunchConfiguration('with_maps')),
            launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items()),
        OpaqueFunction(function=launch_node),
    ])
