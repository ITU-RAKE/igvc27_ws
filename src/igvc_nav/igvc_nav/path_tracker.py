#!/usr/bin/env python3
"""
Yol takipcisi (gecen seneki aracin path_tracker'i, igvc27_ws'e uyarlandi): pure pursuit.

Girdi:
  /igvc_nav/planned_path (nav_msgs/Path) path_planner'in yolu. Yerel cercevede (base_footprint)
                         gelirse, gelis anindaki odom pozuyla odom cercevesine cevrilir ve saklanir;
                         boylece arac yol uzerinde ilerlerken yol yerinde kalir.
  /odometry/local        (nav_msgs/Odometry) yerel EKF (igvc_localization); odom_topic ile degisir
Cikti:
  /cmd_vel               (geometry_msgs/Twist) igvc_controller'a

Kontrol: dinamik lookahead'li pure pursuit (ayrintilar igvc_nav/tracker_core.py):
  - Lookahead hizla uzar (lookahead_min + lookahead_time * hiz, en fazla lookahead_max), ilerideki
    yol kivrildikca kisalir (lookahead_curvature_factor / egrilik): dar S'lerde virajin icinden
    kesmez. Planner'in surekli degisen yol uzunluguna uyar (kalan yol kisaysa hedef yolun sonu).
  - angular = kp_angular * yon hatasi (max_angular_speed ile sinirli)
  - linear  = en kucugu: yon hatasi (slowdown_yaw_error), egrilik (max_lateral_accel), donus
    (turn_slowdown: acisal hiz verirken ayrica carpanla yavaslar). Yon hatasi in_place_yaw_error'dan buyukse
    yerinde doner (ileri itmez). Yolun sonuna 0.5 m kala yavaslar, 0.1 m'de durur.
  - Geri vites yalnizca yol geldigi anda tamamen aracin arkasindaysa (planner'in RECOVERY yolu);
    ileri bir yolun hedefi arac donerken arkada kalirsa yerinde doner.

Standart nav modu (standart_mode; varsayilan planner'in standart_mode'unu izler): standart tracker'in
  sade pure pursuit yasasi (sabit standart_lookahead_dist 0.8 m ya da standart_dynamic_lookahead ile
  hiza bagli), tepe hizi standart_max_linear_speed (1.0, standart tracker'daki gibi). kp, min hiz,
  max acisal hiz, slowdown_yaw_error ortak. Geri vites kurali (yalnizca RECOVERY yolu) korunur.

Kaldirilanlar: rake_core (Node/config/sistem durumu), durum makinesi modlari (hizalama, su,
cakil, yan egim) ve /mission_control/is_aligned.
Eklenen guvenlik: path_timeout boyunca yeni yol gelmezse ya da odom yoksa durur.

Not: klavye surusu (keyboard_controller) da /cmd_vel'e yayinlar; ikisini ayni anda calistirma.
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from rcl_interfaces.msg import ParameterEvent

from igvc_nav.tracker_core import compute_command, compute_command_standard, is_recovery_path


class PathTracker(Node):
    def __init__(self):
        super().__init__("path_tracker")
        p = self.declare_parameter
        # ROS parametre adi -> prm anahtari: hepsi calisirken degistirilebilir, ornegin
        #   ros2 param set /path_tracker turn_slowdown 0.7
        self.param_keys = {
            "kp_angular": "kp",
            "max_linear_speed": "max_v",
            "min_linear_speed": "min_v",
            "max_angular_speed": "max_w",
            "slowdown_yaw_error": "slowdown_yaw_error",
            "lookahead_min": "lookahead_min",
            "lookahead_max": "lookahead_max",
            "lookahead_time": "lookahead_time",
            "lookahead_curvature_factor": "lookahead_curvature_factor",
            "curvature_window_m": "curvature_window",
            "max_lateral_accel": "max_lateral_accel",
            "turn_slowdown": "turn_slowdown",
            "in_place_yaw_error": "in_place_yaw_error",
            "closest_search_m": "closest_search",
            "path_smoothing_m": "path_smoothing",
            "clearance_slowdown": "clearance_slowdown",
            "clearance_window_m": "clearance_window",
            "clearance_power": "clearance_power",
        }

        def f(name, default):
            return float(p(name, default).value)

        self.prm = dict(
            kp=f("kp_angular", 1.0),               # 1.0: simulasyonda en iyi (2.5 salinim yapiyordu)
            max_v=f("max_linear_speed", 1.0),      # m/s
            min_v=f("min_linear_speed", 0.0),      # m/s
            max_w=f("max_angular_speed", 2.0),     # rad/s
            slowdown_yaw_error=f("slowdown_yaw_error", 0.8),        # rad: bu hatada min hiz
            lookahead_min=f("lookahead_min", 0.35),                 # m
            lookahead_max=f("lookahead_max", 1.0),                  # m
            lookahead_time=f("lookahead_time", 0.6),                # s: lookahead = min + bu * hiz
            lookahead_curvature_factor=f("lookahead_curvature_factor", 0.5),  # lookahead <= bu / egrilik
            curvature_window=f("curvature_window_m", 1.2),          # m: ileride egrilige bakilan yol
            max_lateral_accel=f("max_lateral_accel", 0.5),          # m/s^2: viraj hiz siniri
            turn_slowdown=f("turn_slowdown", 0.5),                  # max_w'de hiz %50
            in_place_yaw_error=f("in_place_yaw_error", 1.2),        # rad: ustunde yerinde don
            closest_search=f("closest_search_m", 1.0),              # m
            path_smoothing=f("path_smoothing_m", 0.25),             # m: egrilik icin yumusatma
            clearance_slowdown=f("clearance_slowdown", 0.0),        # 0-1: engele yakin yolda yavasla
            clearance_window=f("clearance_window_m", 1.5),          # m: ileride bakilan yol
            clearance_power=f("clearance_power", 1.0),              # >1: yalnizca engele cok yakinken
        )
        # Standart nav modu: standart tracker'in kontrol yasasi (tracker_core.compute_command_standard).
        # follow_planner_mode=true: planner'in standart_mode parametresini izler, tek komut ikisini birden
        # acar/kapatir:  ros2 param set /path_planner standart_mode true
        self.standart_mode = bool(p("standart_mode", False).value)
        self.follow_planner = bool(p("follow_planner_mode", True).value)
        self.planner_node = p("planner_node", "/path_planner").value
        self.std_keys = {
            "standart_max_linear_speed": "max_v",           # standart tracker'in tepe hizi
            "standart_lookahead_dist": "lookahead",         # sabit lookahead (standart)
            "standart_lookahead_min": "lookahead_min",      # dinamik: min + time * hiz, en fazla max
            "standart_lookahead_max": "lookahead_max",
            "standart_lookahead_time": "lookahead_time",
        }
        self.std_prm = dict(
            max_v=f("standart_max_linear_speed", 1.0),
            lookahead=f("standart_lookahead_dist", 0.8),
            dynamic_lookahead=bool(p("standart_dynamic_lookahead", False).value),
            lookahead_min=f("standart_lookahead_min", 0.5),
            lookahead_max=f("standart_lookahead_max", 1.2),
            lookahead_time=f("standart_lookahead_time", 0.5),
        )
        self.path_timeout = float(p("path_timeout", 1.0).value)          # s
        rate = float(p("control_rate", 20.0).value)                      # Hz
        # Yol bu cercevelerden birinde gelirse yerel sayilir (araca bagli)
        self.local_frames = list(p("local_frames", ["base_footprint", "base_link"]).value)
        path_topic = p("path_topic", "/igvc_nav/planned_path").value
        odom_topic = p("odom_topic", "/odometry/local").value   # EKF (igvc_localization)
        cmd_topic = p("cmd_vel_topic", "/cmd_vel").value

        self.robot_x = self.robot_y = self.robot_yaw = self.robot_speed = 0.0
        self.odom_received = False
        self.global_path = []          # odom cercevesinde [(x, y), ...]
        self.path_costs = []           # her noktanin harita maliyeti (pose.z)
        self.reverse = False           # yol RECOVERY yolu mu (geldigi anda tamamen arkada)
        self.last_path_time = None

        self.create_subscription(Path, path_topic, self.path_callback, 10)
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_timer(1.0 / rate, self.control_loop)
        self.add_on_set_parameters_callback(self.on_params)
        self.create_subscription(ParameterEvent, "/parameter_events", self.param_event_callback, 10)
        self.get_logger().info(
            f"Path tracker (pure pursuit): {path_topic} + {odom_topic} -> {cmd_topic} | "
            f"lookahead {self.prm['lookahead_min']}..{self.prm['lookahead_max']} m (dinamik), "
            f"kp {self.prm['kp']}, v {self.prm['min_v']}..{self.prm['max_v']} m/s | "
            f"standart mod {'ACIK' if self.standart_mode else 'KAPALI'}")

    def on_params(self, params):
        """Kontrol parametreleri calisirken degistirilebilir (ros2 param set)."""
        for prm in params:
            if prm.name == "standart_mode":
                self.set_standart(bool(prm.value))
            elif prm.name == "follow_planner_mode":
                self.follow_planner = bool(prm.value)
            elif prm.name == "standart_dynamic_lookahead":
                self.std_prm["dynamic_lookahead"] = bool(prm.value)
                self.get_logger().info(f"standart dinamik lookahead {'ACIK' if prm.value else 'KAPALI'}")
            elif prm.name in self.std_keys:
                self.std_prm[self.std_keys[prm.name]] = float(prm.value)
                self.get_logger().info(f"{prm.name} = {float(prm.value):g}")
            key = self.param_keys.get(prm.name)
            if key is not None:
                self.prm[key] = float(prm.value)
                self.get_logger().info(f"{prm.name} = {float(prm.value):g}")
        return SetParametersResult(successful=True)

    def set_standart(self, on):
        if on != self.standart_mode:
            self.get_logger().info("Standart nav modu " + ("ACIK" if on else "KAPALI"))
        self.standart_mode = on

    def param_event_callback(self, msg):
        """Planner'in standart_mode degisikligini izle (follow_planner_mode)."""
        if not self.follow_planner or msg.node != self.planner_node:
            return
        for prm in list(msg.new_parameters) + list(msg.changed_parameters):
            if prm.name == "standart_mode":
                self.set_standart(bool(prm.value.bool_value))

    # ------------------------------------------------------------------ girdiler
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.robot_speed = msg.twist.twist.linear.x       # dinamik lookahead icin
        self.odom_received = True

    def path_callback(self, msg):
        if not self.odom_received:
            return
        self.path_costs = [ps.pose.position.z for ps in msg.poses]
        self.last_path_time = self.get_clock().now()
        if msg.header.frame_id not in self.local_frames:
            # Zaten odom/map cercevesinde
            self.global_path = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
            c, s = math.cos(self.robot_yaw), math.sin(self.robot_yaw)
            self.reverse = is_recovery_path(
                [((x - self.robot_x) * c + (y - self.robot_y) * s,
                  -(x - self.robot_x) * s + (y - self.robot_y) * c) for x, y in self.global_path])
            return
        # Yerel yol -> odom (gelis anindaki poz)
        self.reverse = is_recovery_path([(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses])
        c, s = math.cos(self.robot_yaw), math.sin(self.robot_yaw)
        self.global_path = [
            (self.robot_x + ps.pose.position.x * c - ps.pose.position.y * s,
             self.robot_y + ps.pose.position.x * s + ps.pose.position.y * c)
            for ps in msg.poses
        ]

    # ------------------------------------------------------------------ kontrol
    def control_loop(self):
        if not self.odom_received or not self.global_path:
            self.stop_robot()
            return
        age = (self.get_clock().now() - self.last_path_time).nanoseconds * 1e-9
        if age > self.path_timeout:
            self.get_logger().warn(f"Yol {age:.1f} s'dir gelmiyor, duruluyor",
                                   throttle_duration_sec=2.0)
            self.global_path = []
            self.stop_robot()
            return

        pose = (self.robot_x, self.robot_y, self.robot_yaw)
        if self.standart_mode:
            prm = dict(kp=self.prm["kp"], min_v=self.prm["min_v"], max_w=self.prm["max_w"],
                       slowdown_yaw_error=self.prm["slowdown_yaw_error"],
                       closest_search=self.prm["closest_search"], **self.std_prm)
            v, w, info = compute_command_standard(self.global_path, pose, self.robot_speed, prm,
                                                  reverse=self.reverse)
        else:
            v, w, info = compute_command(self.global_path, pose, self.robot_speed, self.prm,
                                         reverse=self.reverse, costs=self.path_costs)
        twist = Twist()
        twist.linear.x, twist.angular.z = float(v), float(w)
        self.cmd_pub.publish(twist)
        self.get_logger().debug(
            f"{info['mode']}: v {v:.2f} w {w:.2f} | lookahead {info['lookahead']:.2f} m, "
            f"egrilik {info['curvature']:.2f} 1/m")

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = PathTracker()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C / launch kapanisi
    except Exception:  # noqa: BLE001
        if rclpy.ok():
            raise
    finally:
        if rclpy.ok():
            try:
                node.stop_robot()
            except Exception:  # noqa: BLE001
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
