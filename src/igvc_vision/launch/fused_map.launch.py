"""Lidar + kamera birlesik harita: ros2 launch igvc_vision fused_map.launch.py [with_inputs:=true]

with_inputs:=true (varsayilan) girdileri de baslatir: lidar_map, surround_view, camera_map.
Arac izdusumu ve inflation yaricaplari (oldurucu = ic teget + pay, sifira inis = cevrel yaricap)
igvc_macros.xacro'dan hesaplanir.
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
    lethal, inflation = robot_geometry.inflation_radii(v)
    return [Node(
        package='igvc_vision',
        executable='fused_map',
        name='fused_map',
        output='screen',
        parameters=[{
            'footprint_m': robot_geometry.footprint_rect(v),
            # A* maliyet haritasi: oldurucu = yari genislik + 5 cm, sifira inis = yarim kosegen
            'lethal_radius_m': lethal,
            'inflation_radius_m': inflation,
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )]


def generate_launch_description():
    share = os.path.join(get_package_share_directory('igvc_vision'), 'launch')
    sim = {'use_sim_time': LaunchConfiguration('use_sim_time')}.items()
    inputs = IfCondition(LaunchConfiguration('with_inputs'))
    return LaunchDescription([
        DeclareLaunchArgument('with_inputs', default_value='true',
                              description='lidar_map, surround_view ve camera_map da baslasin'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Simulasyon saati (gercek aracta false)'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'lidar_map.launch.py')),
            condition=inputs, launch_arguments=sim),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'camera_map.launch.py')),
            condition=inputs, launch_arguments=sim),
        OpaqueFunction(function=launch_node),
    ])
