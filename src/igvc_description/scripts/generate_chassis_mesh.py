#!/usr/bin/env python3
"""
Kesik piramit (frustum) govde icin ASCII STL uretir.
Olculer urdf/igvc_macros.xacro'dan xacro ile okunur (tek kaynak), elle girilmez.
Orijin: taban yuzeyinin merkezi, z yukari, x ileri.

Otomatik calisir:
  - colcon build sirasinda (CMakeLists.txt, macros degistiyse yeniden uretir)
  - display/gazebo launch basinda (--if-changed), boylece --symlink-install ile
    macros'u degistirip yeniden build almadan da mesh guncel kalir.

Elle kullanim:
  python3 generate_chassis_mesh.py [--macros urdf/igvc_macros.xacro] [--if-changed] cikti.stl
"""
import argparse
import os
import sys

import xacro

XACRO_NS = 'http://www.ros.org/wiki/xacro'
PARAMS = ('chassis_bottom_length', 'chassis_bottom_width',
          'chassis_top_length', 'chassis_top_width', 'chassis_height')


def read_dimensions(macros_path):
    """Macros dosyasindaki govde property'lerini xacro ile degerlendir (ifadeler de calisir)."""
    attrs = ' '.join(f'{p}="${{{p}}}"' for p in PARAMS)
    wrapper = (f'<robot xmlns:xacro="{XACRO_NS}">'
               f'<xacro:include filename="{os.path.abspath(macros_path)}"/>'
               f'<dims {attrs}/></robot>')
    doc = xacro.parse(wrapper)
    xacro.process_doc(doc)
    el = doc.getElementsByTagName('dims')[0]
    return {p: float(el.getAttribute(p)) for p in PARAMS}


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]


def normalize(v):
    n = sum(c*c for c in v) ** 0.5
    return [c / n for c in v]


def build_stl(d):
    bl, bw = d['chassis_bottom_length'] / 2, d['chassis_bottom_width'] / 2
    tl, tw = d['chassis_top_length'] / 2, d['chassis_top_width'] / 2
    h = d['chassis_height']
    # 0-3 taban, 4-7 tavan (saat yonu tersine, yukaridan bakinca)
    v = [
        [bl, bw, 0], [-bl, bw, 0], [-bl, -bw, 0], [bl, -bw, 0],
        [tl, tw, h], [-tl, tw, h], [-tl, -tw, h], [tl, -tw, h],
    ]
    # Disa bakan normaller icin CCW sirali yuzler
    quads = [
        (3, 2, 1, 0),  # taban (asagi)
        (4, 5, 6, 7),  # tavan (yukari)
        (0, 1, 5, 4),  # sol (+y)
        (2, 3, 7, 6),  # sag (-y)
        (3, 0, 4, 7),  # on (+x)
        (1, 2, 6, 5),  # arka (-x)
    ]
    out = ['solid chassis']
    for a, b, c, e in quads:
        for tri in ((a, b, c), (a, c, e)):
            p0, p1, p2 = (v[i] for i in tri)
            n = normalize(cross(sub(p1, p0), sub(p2, p0)))
            out.append(f'  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}')
            out.append('    outer loop')
            for p in (p0, p1, p2):
                out.append(f'      vertex {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}')
            out.append('    endloop')
            out.append('  endfacet')
    out.append('endsolid chassis')
    return '\n'.join(out) + '\n'


def main():
    default_macros = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'urdf', 'igvc_macros.xacro')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output', nargs='?', default='chassis.stl')
    ap.add_argument('--macros', default=default_macros, help='igvc_macros.xacro yolu')
    ap.add_argument('--if-changed', action='store_true',
                    help='icerik ayniysa dosyaya dokunma')
    args = ap.parse_args()

    stl = build_stl(read_dimensions(args.macros))

    if args.if_changed and os.path.exists(args.output):
        with open(args.output) as f:
            if f.read() == stl:
                return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(stl)
    print(f'Yazildi: {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
