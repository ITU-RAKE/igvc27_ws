"""RViz'de modeli goster: ros2 launch igvc_description display.launch.py"""
import os
import subprocess
import sys

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
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
    xacro_file = os.path.join(pkg_share, 'urdf', 'igvc.urdf.xacro')
    rviz_config = os.path.join(pkg_share, 'rviz', 'display.rviz')

    gui = LaunchConfiguration('gui')

    def make_rsp(context):
        cad = LaunchConfiguration('cad_visual').perform(context)
        robot_description = ParameterValue(robot_description_xml(xacro_file, cad), value_type=str)
        return [Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen',
        )]

    return LaunchDescription([
        OpaqueFunction(function=sync_chassis_mesh),
        DeclareLaunchArgument('gui', default_value='true',
                              description='joint_state_publisher_gui kullan'),
        DeclareLaunchArgument('cad_visual', default_value='true',
                              description='CAD gorsel mesh (true) / primitif (false)'),

        OpaqueFunction(function=make_rsp),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            condition=IfCondition(gui),
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            condition=UnlessCondition(gui),
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
        ),
    ])
