"""4 kamera kus bakisi: ros2 launch igvc_vision surround_view.launch.py

Kamera pozlari ve arac govdesi igvc_macros.xacro'dan hesaplanir (igvc_vision/robot_geometry.py).
Kameralarin ic modeli (K, distorsiyon, isleme cozunurlugu) config/camera_calibration.yaml'dan:
etkin olmayan kamera icin xacro'dan ideal pinhole (simulasyon). Hepsi config/surround_view.yaml'in
uzerine yazilir. URDF'te kamera/govde olcusu degisince burada bir sey yapmaya gerek yok.
  calibration_file:=<yaml>  baska bir kalibrasyon dosyasi (varsayilan: config/camera_calibration.yaml)
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
    calib_file = LaunchConfiguration('calibration_file').perform(context)
    calib = robot_geometry.load_calibration(calib_file or None)
    geometry = robot_geometry.surround_view_params(robot_geometry.read_props(), calib)
    on = [n for n in robot_geometry.CAMERA_LAYOUT if calib.get(n)]
    print(f"[surround_view] kalibrasyon: {calib_file or robot_geometry.default_calibration_path()} | "
          f"etkin: {', '.join(on) if on else 'yok (simulasyon modeli)'} | "
          f"isleme {geometry['image_width']}x{geometry['image_height']}")
    return [Node(
        package='igvc_vision',
        executable='surround_view',
        name='surround_view_node',
        output='screen',
        # sonraki oncekini ezer: geometri (xacro) + ic kalibrasyon her zaman yaml ayarlarinin ustunde
        parameters=[cfg, geometry, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Simulasyon saati (gercek aracta false)'),
        DeclareLaunchArgument('calibration_file', default_value='',
                              description='Kamera ic kalibrasyonu (bos: config/camera_calibration.yaml)'),
        OpaqueFunction(function=launch_node),
    ])
