"""4 kamera kus bakisi: ros2 launch igvc_vision surround_view.launch.py

Kamera pozlari, intrinsic'ler, goruntu boyutu ve arac govdesi igvc_macros.xacro'dan
hesaplanir (igvc_vision/robot_geometry.py) ve config/surround_view.yaml'in uzerine yazilir.
URDF'te kamera/govde olcusu degisince burada bir sey yapmaya gerek yok.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from igvc_vision import robot_geometry


def launch_node(context):
    cfg = os.path.join(get_package_share_directory('igvc_vision'), 'config', 'surround_view.yaml')
    geometry = robot_geometry.surround_view_params(robot_geometry.read_props())
    return [Node(
        package='igvc_vision',
        executable='surround_view',
        name='surround_view_node',
        output='screen',
        # sonraki oncekini ezer: geometri her zaman xacro'dan
        parameters=[cfg, geometry, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Simulasyon saati (gercek aracta false)'),
        OpaqueFunction(function=launch_node),
    ])
