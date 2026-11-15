"""Lokalizasyon: ros2 launch igvc_localization igvc_localization.launch.py

  gps_odometry : /gps/fix -> /gps/odometry (map'te ENU), /gps/datum
  ekf_local    : /odom + /imu/data -> /odometry/local, TF odom -> base_footprint
  ekf_global   : + /gps/odometry + mutlak yon -> /odometry/global, TF map -> odom
Ayarlar: config/ekf.yaml.

DIKKAT: sim.launch.py (ekf:=true, varsayilan) bu launch'i ZATEN baslatir. Ayrica acmayin: iki kopya
iki odom cercevesi ve iki kat GPS demek. Dugumler ikinci kopyayi algilayip kapatir. Bu launch
yalnizca sim.launch.py ekf:=false ile ya da gercek aracta (use_sim_time:=false) kullanilir;
o durumda igvc_controller'in TF'i kapali olmali (controller.launch.py publish_tf:=false).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('igvc_localization'), 'config', 'ekf.yaml')
    sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}

    def node(executable, name):
        return Node(package='igvc_localization', executable=executable, name=name,
                    output='screen', parameters=[cfg, sim_time])

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Simulasyon saati (gercek aracta false)'),
        node('gps_odometry', 'gps_odometry'),
        node('ekf_node', 'ekf_local'),
        node('ekf_node', 'ekf_global'),
    ])
