"""Gazebo Classic'te robotu spawn et: ros2 launch igvc_description gazebo.launch.py

Robot spawn olduktan sonra ros2_control kontrolculeri baslar (simulasyon motor surucusu):
  joint_state_broadcaster   -> /joint_states (tekerlek enkoderleri)
  wheel_velocity_controller -> /wheel_velocity_controller/commands (tekerlek hizlari, rad/s)
cmd_vel -> tekerlek ve /odom: igvc_controller (ros2 launch igvc_controller sim.launch.py hepsini baslatir).
"""
import os
import subprocess
import sys

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def robot_description_xml(xacro_file, cad_visual='true'):
    """xacro -> URDF metni, XML yorumlari temizlenmis.
    gazebo_ros2_control URDF'i '--param robot_description:=...' olarak YAML ayristiricisina verir;
    yorumlardaki ': ' ve ' #' gibi diziler ayristirmayi bozar. Yorumlar dosyada kalir, yalnizca
    yayinlanan aciklamadan cikarilir.
    cad_visual: 'true' -> CAD gorsel mesh'leri, 'false' -> primitif (yalnizca <visual> degisir)."""
    import re

    import xacro
    xml = xacro.process_file(xacro_file, mappings={'cad_visual': cad_visual}).toxml()
    return re.sub(r'<!--.*?-->', '', xml, flags=re.S)


def sync_chassis_mesh(context):
    """Govde mesh'ini igvc_macros.xacro ile esitle (degistiyse yeniden yaz).
    --symlink-install ile macros'u degistirip build almadan da mesh guncel kalir."""
    prefix = get_package_prefix('igvc_description')
    share = get_package_share_directory('igvc_description')
    script = os.path.join(prefix, 'lib', 'igvc_description', 'generate_chassis_mesh.py')
    cmd = [sys.executable, script, '--if-changed',
           '--macros', os.path.join(share, 'urdf', 'igvc_macros.xacro'),
           os.path.join(share, 'meshes', 'chassis.stl')]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f'[UYARI] chassis.stl guncellenemedi:\n{res.stderr}', file=sys.stderr)
    elif res.stdout:
        print(res.stdout.strip())
    return []


def generate_launch_description():
    pkg_share = get_package_share_directory('igvc_description')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')

    xacro_file = os.path.join(pkg_share, 'urdf', 'igvc.urdf.xacro')
    default_world = os.path.join(pkg_share, 'worlds', 'empty.world')
    default_rviz = os.path.join(pkg_share, 'rviz', 'display.rviz')

    world = LaunchConfiguration('world')
    rviz = LaunchConfiguration('rviz')
    x, y, z, yaw = (LaunchConfiguration(n) for n in ('x', 'y', 'z', 'yaw'))

    def make_rsp(context):
        cad = LaunchConfiguration('cad_visual').perform(context)
        robot_description = ParameterValue(robot_description_xml(xacro_file, cad), value_type=str)
        return [Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
            output='screen',
        )]

    gazebo =IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros_share, 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world, 'verbose': 'false', 'gui': LaunchConfiguration('gui')}.items(),
    )

    rsp = OpaqueFunction(function=make_rsp)

    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'igvc',
                   '-x', x, '-y', y, '-z', z, '-Y', yaw],
        output='screen',
    )

    # ros2_control kontrolculeri: robot (ve gazebo_ros2_control eklentisi) yuklendikten sonra
    spawners = [
        Node(package='controller_manager', executable='spawner', output='screen',
             arguments=[name, '--controller-manager', '/controller_manager',
                        '--controller-manager-timeout', '30'])
        for name in ('joint_state_broadcaster', 'wheel_velocity_controller')
    ]
    start_controllers = RegisterEventHandler(
        OnProcessExit(target_action=spawn, on_exit=spawners))

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(rviz),
    )

    return LaunchDescription([
        OpaqueFunction(function=sync_chassis_mesh),
        DeclareLaunchArgument('world', default_value=default_world),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
        DeclareLaunchArgument('gui', default_value='true', description='Gazebo arayuzu (gzclient)'),
        DeclareLaunchArgument('cad_visual', default_value='true',
                              description='CAD gorsel mesh (true) / primitif (false); fizik ayni'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.05'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        gazebo,
        rsp,
        spawn,
        start_controllers,
        rviz_node,
    ])
