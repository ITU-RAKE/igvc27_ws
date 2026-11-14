"""Skid steer kontrolcu: ros2 launch igvc_controller controller.launch.py

Tekerlek yaricapi, iz genisligi ve azami tekerlek hizi igvc_macros.xacro'dan okunur;
diger ayarlar config/controller.yaml. Klavye surusu ayri terminalde:
  ros2 run igvc_controller keyboard_controller
"""
import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PROPS = ('wheel_radius', 'wheel_separation', 'max_velocity')


def read_props():
    macros = os.path.join(get_package_share_directory('igvc_description'), 'urdf', 'igvc_macros.xacro')
    attrs = ' '.join(f'{p}="${{{p}}}"' for p in PROPS)
    doc = xacro.parse(f'<robot xmlns:xacro="http://www.ros.org/wiki/xacro">'
                      f'<xacro:include filename="{macros}"/><v {attrs}/></robot>')
    xacro.process_doc(doc)
    el = doc.getElementsByTagName('v')[0]
    return {p: float(el.getAttribute(p)) for p in PROPS}


def launch_node(context):
    v = read_props()
    cfg = os.path.join(get_package_share_directory('igvc_controller'), 'config', 'controller.yaml')
    return [Node(
        package='igvc_controller',
        executable='controller',
        name='igvc_controller',
        output='screen',
        parameters=[cfg, {
            'wheel_radius': v['wheel_radius'],
            'wheel_separation': v['wheel_separation'],
            'max_wheel_velocity': v['max_velocity'],
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Simulasyon saati (gercek aracta false)'),
        OpaqueFunction(function=launch_node),
    ])
