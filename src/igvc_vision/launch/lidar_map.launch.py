"""Anlik lidar haritasi: ros2 launch igvc_vision lidar_map.launch.py

Arac izdusumu (debug goruntudeki dikdortgen ve oz-filtre) igvc_macros.xacro'dan
hesaplanir (igvc_vision/robot_geometry.py); govde/tekerlek olculerini degistirince
burada bir sey yapmaya gerek yok.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from igvc_vision import robot_geometry


def launch_node(context):
    length, width, center_x = robot_geometry.footprint(robot_geometry.read_props())
    return [Node(
        package='igvc_vision',
        executable='lidar_map',
        name='lidar_map',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_length': length,
            'robot_width': width,
            'robot_center_x': center_x,
        }],
        output='screen',
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        OpaqueFunction(function=launch_node),
    ])
