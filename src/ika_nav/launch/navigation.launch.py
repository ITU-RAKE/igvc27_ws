import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import SetRemap

def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    
    # HATA BURADAYDI: 'kendi_paket_adiniz' yerine 'ika_nav' yazıldı.
    params_file = os.path.join(get_package_share_directory('ika_nav'), 'config', 'nav2_params.yaml')

    return LaunchDescription([
        SetRemap(src='/cmd_vel', dst='/igvc_controllers/cmd_vel_unstamped'),
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')),
            launch_arguments={
                'use_sim_time': 'true',
                'params_file': params_file,
                'autostart': 'true'
            }.items(),
        )
    ])