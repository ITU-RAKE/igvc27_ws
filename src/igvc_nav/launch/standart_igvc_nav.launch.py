"""Standart navigasyon (karsilastirma icin): ros2 launch igvc_nav standart_igvc_nav.launch.py

Workspace'e ilk entegre edilen planner/tracker (standart_path_planner.py, standart_path_tracker.py):
BFS hedef secimi + agirlikli A* (10 Hz) ve sabit lookahead'li pure pursuit. Frontier, iz hafizasi,
geri donme duzeltmesi yok. Kendi varsayilan parametreleriyle calisir (tracker tepe hizi 1.0 m/s).

Topic'ler bizim nav ile ayni (/fused_map -> /igvc_nav/planned_path -> /cmd_vel):
igvc_nav.launch.py ile AYNI ANDA calistirma. Dugum adlari ayri:
  ros2 param set /standart_path_planner <ad> <deger>
  ros2 param set /standart_path_tracker <ad> <deger>
with_maps:=true igvc_vision fused_map zincirini de baslatir.
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


def launch_nodes(context):
    sim_time = LaunchConfiguration('use_sim_time')
    footprint = robot_geometry.footprint_rect(robot_geometry.read_props())
    return [
        Node(package='igvc_nav', executable='standart_path_planner', name='standart_path_planner',
             output='screen',
             parameters=[{'map_topic': '/fused_map', 'footprint_m': footprint,
                          'use_sim_time': sim_time}]),
        Node(package='igvc_nav', executable='standart_path_tracker', name='standart_path_tracker',
             output='screen', parameters=[{'use_sim_time': sim_time}]),
    ]


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
        OpaqueFunction(function=launch_nodes),
    ])
