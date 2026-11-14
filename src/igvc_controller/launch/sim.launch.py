"""Simulasyon + kontrolcu: ros2 launch igvc_controller sim.launch.py

Gazebo (parkur.world, robot + ros2_control), RViz (yeni_config.rviz) ve skid steer kontrolcuyu
baslatir. Argumanlar: world:=<dosya> rviz:=false gui:=false rviz_config:=<dosya> x:= y:= yaw:=
parkur.world henuz yoksa (ya da colcon build alinmadiysa) empty.world kullanilir.
Surmek icin ayri terminalde: ros2 run igvc_controller keyboard_controller
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    desc_share = get_package_share_directory('igvc_description')
    desc = os.path.join(desc_share, 'launch')
    ctrl = os.path.join(get_package_share_directory('igvc_controller'), 'launch')

    parkur = os.path.join(desc_share, 'worlds', 'parkur.world')
    default_world = parkur if os.path.exists(parkur) else os.path.join(desc_share, 'worlds', 'empty.world')
    notes = [] if os.path.exists(parkur) else [LogInfo(msg=(
        'parkur.world bulunamadi (worlds/ klasorune koyup colcon build alin) -> empty.world'))]

    gz_args = {k: LaunchConfiguration(k) for k in ('rviz', 'rviz_config', 'world', 'gui', 'x', 'y', 'yaw')}
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value=default_world),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config',
                              default_value=os.path.join(desc_share, 'rviz', 'yeni_config.rviz')),
        DeclareLaunchArgument('gui', default_value='true', description='Gazebo arayuzu'),
        DeclareLaunchArgument('x', default_value='0.0', description='Aracin baslangic konumu'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        *notes,
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(desc, 'gazebo.launch.py')),
                                 launch_arguments=gz_args.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(ctrl, 'controller.launch.py')),
                                 launch_arguments={'use_sim_time': 'true'}.items()),
    ])
