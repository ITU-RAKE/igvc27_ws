"""Yerel yol planlayici: ros2 launch igvc_nav path_planner.launch.py [with_maps:=false]

Girdi /fused_map (igvc_vision). with_maps:=true ise igvc_vision fused_map zincirini de baslatir
(lidar_map, surround_view, camera_map, fused_map). Arac ayak izi igvc_macros.xacro'dan.
"""
import os

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
    return [Node(
        package='igvc_nav',
        executable='path_planner',
        name='path_planner',
        output='screen',
        parameters=[{
            'map_topic': '/fused_map',
            'footprint_m': robot_geometry.footprint_rect(v),
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
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(vision, 'fused_map.launch.py')),
            condition=IfCondition(LaunchConfiguration('with_maps')),
            launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items()),
        OpaqueFunction(function=launch_node),
    ])
