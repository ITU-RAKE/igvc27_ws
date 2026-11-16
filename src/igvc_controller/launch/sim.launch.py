"""Simulasyon + kontrolcu: ros2 launch igvc_controller sim.launch.py

Gazebo (igvc_course.world, robot + ros2_control), RViz (yeni_config.rviz), skid steer kontrolcu ve
EKF lokalizasyonu (igvc_localization) baslatir.
Argumanlar: world:=<dosya> rviz:=false gui:=false rviz_config:=<dosya> x:= y:= yaw:= ekf:=false
Baska dunya: world:=<tam yol> (orn. .../worlds/parkur.world). x/y/yaw:=auto (varsayilan):
igvc_course*.world'de parkurun dogma noktasi (baslangic cizgisinin onu, kuzeye), diger dunyalarda 0.
Surmek icin ayri terminalde: ros2 run igvc_controller keyboard_controller
"""
import os
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression


def read_course_spawn(desc_share):
    """igvc_course dogma noktasi [x, y, yaw] (course.yaml 'spawn'); yoksa rampasiz varsayilan."""
    path = os.path.join(desc_share, 'models', 'igvc_course', 'course.yaml')
    try:
        with open(path) as f:
            m = re.search(r'^spawn:\s*\[([^\]]+)\]', f.read(), re.M)
        return [float(v) for v in m.group(1).split(',')][:3]
    except (OSError, AttributeError, ValueError):
        return [0.0, 0.0, 1.5708]


def generate_launch_description():
    desc_share = get_package_share_directory('igvc_description')
    desc = os.path.join(desc_share, 'launch')
    ctrl = os.path.join(get_package_share_directory('igvc_controller'), 'launch')
    loc = os.path.join(get_package_share_directory('igvc_localization'), 'launch')
    ekf = LaunchConfiguration('ekf')

    # Varsayilan: IGVC Auto-Nav benzeri parkur (baslangic cizgisi orijinde, kuzeye bakar -> yaw 90 deg)
    course = os.path.join(desc_share, 'worlds', 'igvc_course_2.world')
    default_world = course if os.path.exists(course) else os.path.join(desc_share, 'worlds', 'empty.world')
    notes = [] if os.path.exists(course) else [LogInfo(msg=(
        'igvc_course.world bulunamadi (colcon build alin) -> empty.world'))]

    gz_args = {k: LaunchConfiguration(k) for k in ('rviz', 'rviz_config', 'world', 'gui')}
    # x / y / yaw := auto -> igvc_course.world'de parkurun dogma noktasi (rampanin kuzey ucunun
    # ilerisi, kuzeye bakar; models/igvc_course/course.yaml 'spawn', parkur betigi hesaplar),
    # diger dunyalarda 0. Elle verilen deger her zaman kullanilir.
    course_spawn = read_course_spawn(desc_share)
    world = LaunchConfiguration('world')
    for i, k in enumerate(('x', 'y', 'yaw')):
        v = LaunchConfiguration(k)
        gz_args[k] = PythonExpression([
            # igvc_course.world ve igvc_course_2.world (ayni parkur modeli) -> parkur dogma noktasi
            f"('{course_spawn[i]}' if '", world, "'.split('/')[-1].startswith('igvc_course') else '0.0') if '",
            v, "' == 'auto' else '", v, "'"])
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value=default_world),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config',
                              default_value=os.path.join(desc_share, 'rviz', 'yeni_config.rviz')),
        DeclareLaunchArgument('gui', default_value='true', description='Gazebo arayuzu'),
        DeclareLaunchArgument('x', default_value='auto',
                              description='Aracin baslangic konumu (auto: igvc_course dogma noktasi, diger dunyalar 0)'),
        DeclareLaunchArgument('y', default_value='auto'),
        DeclareLaunchArgument('yaw', default_value='auto',
                              description='auto: igvc_course.world dogma yonu (kuzey), diger dunyalar 0.0'),
        DeclareLaunchArgument('ekf', default_value='true',
                              description='EKF lokalizasyonu (odom TF\'ini EKF yayinlar)'),
        *notes,
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(desc, 'gazebo.launch.py')),
                                 launch_arguments=gz_args.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(ctrl, 'controller.launch.py')),
            launch_arguments={'use_sim_time': 'true',
                              # EKF varsa odom->base_footprint TF'i kontrolcu degil EKF yayinlar
                              'publish_tf': PythonExpression(["'false' if '", ekf, "' == 'true' else 'true'"]),
                              }.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(loc, 'igvc_localization.launch.py')),
            condition=IfCondition(ekf), launch_arguments={'use_sim_time': 'true'}.items()),
    ])
