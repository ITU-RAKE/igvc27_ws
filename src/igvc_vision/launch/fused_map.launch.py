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
    margin = float(LaunchConfiguration('lethal_margin_m').perform(context))
    lethal, inflation = robot_geometry.inflation_radii(v, safety_margin=margin)
    extra = float(LaunchConfiguration('inflation_gradient_extra_m').perform(context))
    return [Node(
        package='igvc_vision',
        executable='fused_map',
        name='fused_map',
        output='screen',
        parameters=[{
            'footprint_m': robot_geometry.footprint_rect(v),
            # Maliyet haritasi: oldurucu = yari genislik + 5 cm (degismez),
            # sifira inis = yarim kosegen + gradyan payi, ustel profil
            'lethal_radius_m': lethal,
            'inflation_radius_m': inflation + extra,
            'inflation_profile': 'exp' if extra > 0 else 'linear',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )]


def generate_launch_description():
    share = os.path.join(get_package_share_directory('igvc_vision'), 'launch')
    sim = {'use_sim_time': LaunchConfiguration('use_sim_time')}.items()
    inputs = IfCondition(LaunchConfiguration('with_inputs'))
    return LaunchDescription([
        # Oldurucu bolge = yari genislik + bu pay. 0.05'te arac dubanin ~0.42 m yanindan gecebilir
        # (siyirma riski). Ama igvc_course_2'deki ~1.1 m duba araliklari 0.075 / 0.10 ile kapaniyor
        # (tur bitmedi, 200+ RECOVERY): bu parkur icin 0.05.
        DeclareLaunchArgument('lethal_margin_m', default_value='0.05',
                              description='Oldurucu bolge payi [m] (ic teget yaricapin ustune)'),
        # Tam tur karsilastirmasi (igvc_course): 0.25 m tur suresini iyilestirmedi, 0.5 m dar
        # gecitte arac geri donup turu bozdu -> varsayilan 0 (yarim kosegende biten dogrusal).
        DeclareLaunchArgument('inflation_gradient_extra_m', default_value='0.0',
                              description='Gradyan payi [m], >0 ise ustel profil (0 = dogrusal, yarim kosegene kadar)'),
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
