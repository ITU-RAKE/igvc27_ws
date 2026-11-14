"""Simulasyon + kontrolcu: ros2 launch igvc_controller sim.launch.py [rviz:=true] [gui:=false] [world:=...]

Gazebo (robot + ros2_control) ve skid steer kontrolcuyu baslatir. Surmek icin ayri terminalde:
  ros2 run igvc_controller keyboard_controller
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    desc = os.path.join(get_package_share_directory('igvc_description'), 'launch')
    ctrl = os.path.join(get_package_share_directory('igvc_controller'), 'launch')
    gz_args = {k: LaunchConfiguration(k) for k in ('rviz', 'world', 'gui')}
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('gui', default_value='true', description='Gazebo arayuzu'),
        DeclareLaunchArgument('world', default_value=os.path.join(
            get_package_share_directory('igvc_description'), 'worlds', 'empty.world')),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(desc, 'gazebo.launch.py')),
                                 launch_arguments=gz_args.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(ctrl, 'controller.launch.py')),
                                 launch_arguments={'use_sim_time': 'true'}.items()),
    ])
