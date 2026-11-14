#!/usr/bin/env python3
"""
Klavyeden WASD ile surus: /cmd_vel (geometry_msgs/Twist) yayinlar.

Calistirma (kendi terminalinde; launch icinden calismaz, klavye girdisi ister):
  ros2 run igvc_controller keyboard_controller
  ros2 run igvc_controller keyboard_controller --ros-args -p mode:=hold

Tuslar:
  W / S   ileri / geri          A / D   sola / saga don
  Bosluk  dur (hizlari sifirla) Q / E   hiz carpanini azalt / artir
  Ctrl+C  cik (cikarken dur komutu gonderir)

Modlar (terminal tus birakmayi bildirmez ve ayni anda iki tus okuyamaz):
  increment (varsayilan): her basis hizi bir adim degistirir, hiz korunur. W ile hizlanirken
             A/D ile egrilik verilebilir (viraj). Bosluk ile dur.
  hold:      tus basili tutuldukca gider, birakinca (hold_timeout sonra) durur. Tek tus: viraj yok.

Komut 20 Hz surekli yayinlanir; bu dugum kapanirsa controller cmd_timeout sonra durur.
"""
import os
import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

HELP = """
---------------------------------------------
 IGVC klavye surusu ({mode})
   W/S: ileri/geri   A/D: sola/saga
   Bosluk: dur   Q/E: hiz carpani -/+   Ctrl+C: cik
---------------------------------------------"""


class KeyboardController(Node):
    def __init__(self):
        super().__init__('keyboard_controller')
        p = self.declare_parameter
        self.mode = p('mode', 'increment').value             # increment | hold
        self.max_v = float(p('max_linear_velocity', 1.5).value)
        self.max_w = float(p('max_angular_velocity', 2.0).value)
        self.step_v = float(p('linear_step', 0.1).value)      # increment: basis basina
        self.step_w = float(p('angular_step', 0.2).value)
        self.hold_timeout = float(p('hold_timeout', 0.6).value)  # hold: > klavye tekrar gecikmesi
        self.scale = float(p('speed_scale', 0.5).value)       # hold: max hizin carpani
        topic = p('cmd_vel_topic', '/cmd_vel').value
        self.pub = self.create_publisher(Twist, topic, 10)
        self.v = self.w = 0.0
        self.last_key_time = self.get_clock().now()
        self.get_logger().info(f'Klavye -> {topic} (mod: {self.mode})')

    def clamp(self):
        self.v = max(-self.max_v, min(self.max_v, self.v))
        self.w = max(-self.max_w, min(self.max_w, self.w))

    def on_key(self, k):
        self.last_key_time = self.get_clock().now()
        if k == ' ':
            self.v = self.w = 0.0
        elif k in 'qQ':
            self.scale = max(0.1, round(self.scale - 0.1, 2))
        elif k in 'eE':
            self.scale = min(1.0, round(self.scale + 0.1, 2))
        elif self.mode == 'hold':
            vmax, wmax = self.max_v * self.scale, self.max_w * self.scale
            self.v, self.w = {'w': (vmax, 0.0), 's': (-vmax, 0.0),
                              'a': (0.0, wmax), 'd': (0.0, -wmax)}.get(k.lower(), (self.v, self.w))
        else:
            dv, dw = {'w': (self.step_v, 0.0), 's': (-self.step_v, 0.0),
                      'a': (0.0, self.step_w), 'd': (0.0, -self.step_w)}.get(k.lower(), (0.0, 0.0))
            self.v = round(self.v + dv, 3)
            self.w = round(self.w + dw, 3)
        self.clamp()

    def tick(self):
        if self.mode == 'hold':
            idle = (self.get_clock().now() - self.last_key_time).nanoseconds * 1e-9
            if idle > self.hold_timeout:
                self.v = self.w = 0.0
        t = Twist()
        t.linear.x, t.angular.z = self.v, self.w
        self.pub.publish(t)

    def status(self):
        extra = f'  carpan {self.scale:.1f}' if self.mode == 'hold' else ''
        return f'\r  v = {self.v:+.2f} m/s   w = {self.w:+.2f} rad/s{extra}      '

    def send_stop(self):
        self.v = self.w = 0.0
        self.pub.publish(Twist())


def main(args=None):
    if not sys.stdin.isatty():
        print('keyboard_controller bir terminal ister (ros2 run ile kendi terminalinde calistir).',
              file=sys.stderr)
        return 1
    # rclpy'nin SIGINT isleyicisi kapali: Ctrl+C KeyboardInterrupt olarak gelir, context acikken
    # 'dur' komutu gonderilebilir
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = KeyboardController()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print(HELP.format(mode=node.mode))
    try:
        tty.setcbreak(fd)           # satir beklemeden tek tus oku, Ctrl+C yine SIGINT
        period = 0.05               # 20 Hz
        while rclpy.ok():
            ready, _, _ = select.select([sys.stdin], [], [], period)
            if ready:
                k = os.read(fd, 1).decode(errors='ignore')
                if k:
                    node.on_key(k)
            node.tick()
            rclpy.spin_once(node, timeout_sec=0.0)
            sys.stdout.write(node.status())
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()
        if rclpy.ok():
            node.send_stop()
            node.destroy_node()
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
