"""BEV + YOLO (four_cam_test): ros2 launch igvc_vision four_cam.launch.py engine_path:=/yol/best.engine

engine_path verilmezse YOLO sureci modeli yuklemez, yalnizca BEV sureci calisir.
Kamera cozunurlugu ve homografiler igvc_macros.xacro'dan hesaplanir (robot_geometry.py).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from igvc_vision import robot_geometry

BEV_SIZE = 960      # px
BEV_METERS = 12.0   # m (BEV_SIZE px'e karsilik gelen kenar)


def launch_nodes(context):
    v = robot_geometry.read_props()
    w, h, _ = robot_geometry.camera_intrinsics(v)
    common = {
        "cam_width": w,
        "cam_height": h,
        "bev_size": BEV_SIZE,
        "bev_meters": BEV_METERS,
        "max_range_m": 4.9,
        "bev_frame_id": "base_footprint",
        "homographies": robot_geometry.bev_homographies(v, BEV_SIZE, BEV_METERS),
    }
    # Aynı node iki ayrı süreçte: biri sadece BEV, biri sadece YOLO.
    # Böylece YOLO'nun Python ön/son işlemesi BEV thread'ini (GIL) yavaşlatmaz.
    return [
        Node(
            package="igvc_vision",
            executable="four_cam_test",
            name="bev_node",
            output="screen",
            parameters=[{**common, "run_bev": True, "run_yolo": False, "bev_rate_hz": 20.0}],
        ),
        Node(
            package="igvc_vision",
            executable="four_cam_test",
            name="yolo_bev_node",
            output="screen",
            parameters=[{**common, "run_bev": False, "run_yolo": True, "yolo_rate_hz": 10.0,
                         "engine_path": LaunchConfiguration("engine_path")}],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("engine_path", default_value="",
                              description="YOLO TensorRT .engine dosyasi (bos = YOLO kapali)"),
        OpaqueFunction(function=launch_nodes),
    ])
