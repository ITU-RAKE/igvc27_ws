#!/usr/bin/env python3
"""
IGVC Auto-Nav benzeri parkur zemini: asfalt + beyaz serit cizgileri dokusu ve onu tasiyan mesh.

Uretir (models/igvc_course/):
  materials/textures/course.png  asfalt + 3" beyaz cizgiler (+ opsiyonel cukurlar)
  materials/textures/course_2.png, meshes/course_2.dae  ayni parkur, baslangic cizgisi yok (igvc_course_2.world)
  meshes/course.dae              doku bir kez oturtulmus tek yuzlu zemin (dunya olcusunde)
  course.yaml                    cizgilerin dunya koordinatlari (dogrulama / ileride kullanim)

Kullanim:  python3 generate_igvc_course.py            (paket kaynak klasorune yazar)
Parametreleri asagidaki COURSE sozlugunden degistir, yeniden calistir, colcon build al.

Olculer IGVC kurallarindan: alan 120 x 100 ft, serit 10 ft (3.05 m), cizgi 3" (7.6 cm), donus
yaricapi >= 5 ft, cukur 2 ft beyaz daire. Dunya cercevesi ENU (x dogu, y kuzey); baslangic cizgisi
(0, 0)'da, arac kuzeye bakar. Parkur saat yonunde: kuzey (sol duz) -> dogu (ust sikan) -> guney
(sag, No Man's Land + rampa alani, cizgi yok) -> bati (alt sikan) -> baslangic.
Cizgiler duz (yukseklik yok): lidar gormez, kamera gorur (gercekte de bant boyle).
"""
import math
import os
import sys

import cv2
import numpy as np

FT = 0.3048
COURSE = {
    'resolution': 0.02,          # m/piksel (3" cizgi ~ 4 piksel)
    # Zemin dokusunun kapsadigi alan [m]. Seritten en az kamera menzili (~5 m) kadar pay birak:
    # doku kenari kus bakisinda gorunmesin
    'x_range': (-8.0, 41.0),
    'y_range': (-21.0, 21.0),
    'track_width': 10 * FT,      # 3.05 m
    'line_width': 3 * 0.0254,    # 7.6 cm
    # Merkez hattin dikdortgen iskeleti: sol x, sag x, alt y, ust y, kose yaricapi
    'left_x': 0.0, 'right_x': 33.0, 'bottom_y': -13.0, 'top_y': 13.0, 'corner_r': 4.0,
    # Ust ve alt duzluklerde sinus sikan: genlik [m], dalga boyu [m]
    'chicane_amp': 1.3, 'chicane_wavelength': 12.5,
    # No Man's Land: sag tarafta cizgisiz bolum (y araligi) -> waypoint'lerle gecilir
    'no_mans_land_y': (-7.0, 7.0),
    'start_line': True,          # aracin dogdugu yerde seridi kesen cizgi
    'potholes': [],              # [(x, y), ...] 2 ft beyaz daireler, orn. [(10.0, 13.0)]
    # Rampa (kural: egim <= %15). Baslangic noktasinda, sol duzlukte, arac yonunde (y ekseni):
    # cikis rampasi + ust duzluk + inis rampasi. Bos sozluk ({}) -> rampa yok.
    'ramp': {'center': (0.0, 0.0), 'width': 8 * FT, 'height': 0.30, 'grade': 0.15,
             'top_length': 4 * FT, 'color': (0.15, 0.45, 0.20)},
    # Baslangic cizgisi: rampanin kuzey ucundan bu kadar ileride, sol duzlukte.
    'spawn_clearance': 0.8,
    # Aracin dogma noktasi: cizginin bu kadar ilerisinde (base_link), kuzeye bakar, arkasi cizgiye
    # donuk. 0.6 = yarim cizgi 0.04 + govde yarisi 0.475 + ~8 cm pay: cizgi aracin altinda kalmaz.
    # course.yaml'a 'spawn' olarak yazilir; sim.launch.py (yaw/x/y auto) oradan okur.
    'spawn_ahead_of_line': 0.6,
    'pothole_diameter': 2 * FT,
    'asphalt_gray': 45, 'asphalt_noise': 6, 'line_gray': 235,
}


def centerline(c, step=0.05):
    """Saat yonunde kapali merkez hat (sol duzden baslar), (N, 2) [m]."""
    L, R, B, T, r = c['left_x'], c['right_x'], c['bottom_y'], c['top_y'], c['corner_r']
    amp, lam = c['chicane_amp'], c['chicane_wavelength']
    pts = []

    def line(p0, p1):
        n = max(2, int(math.dist(p0, p1) / step))
        for t in np.linspace(0, 1, n, endpoint=False):
            pts.append((p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t))

    def arc(cx, cy, a0, a1):
        n = max(4, int(abs(a1 - a0) * r / step))
        for a in np.linspace(a0, a1, n, endpoint=False):
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))

    def chicane(x0, x1, y):
        # Sinus genligi uclarda sifira iner (kose yayina teget, puruzsuz gecis)
        n = max(2, int(abs(x1 - x0) / step))
        for t in np.linspace(0, 1, n, endpoint=False):
            x = x0 + (x1 - x0) * t
            env = math.sin(math.pi * t) ** 2
            pts.append((x, y + amp * env * math.sin(2 * math.pi * (x - x0) / lam)))

    line((L, 0.0), (L, T - r))                         # sol duz, kuzeye (baslangic y=0)
    arc(L + r, T - r, math.pi, math.pi / 2)            # sol ust kose
    chicane(L + r, R - r, T)                           # ust sikan, doguya
    arc(R - r, T - r, math.pi / 2, 0.0)                # sag ust kose
    line((R, T - r), (R, B + r))                       # sag, guneye (No Man's Land burada)
    arc(R - r, B + r, 0.0, -math.pi / 2)               # sag alt kose
    chicane(R - r, L + r, B)                           # alt sikan, batiya
    arc(L + r, B + r, -math.pi / 2, -math.pi)          # sol alt kose
    line((L, B + r), (L, 0.0))                         # sol duz, baslangica
    return np.array(pts)


def offset(pts, d):
    """Kapali hatti normal yonunde d kadar kaydir (+: sol)."""
    nxt = np.roll(pts, -1, axis=0)
    prv = np.roll(pts, 1, axis=0)
    t = nxt - prv
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return pts + d * np.stack([-t[:, 1], t[:, 0]], axis=1)


def spawn_pose(c):
    """(x, y, yaw): rampa varsa kuzey ucunun spawn_clearance ilerisi, yoksa sol duzlukte y = 0."""
    r = c.get('ramp')
    if r:
        half = r['top_length'] / 2 + r['height'] / r['grade']
        return c['left_x'], r['center'][1] + half + c['spawn_clearance'], math.pi / 2
    return c['left_x'], 0.0, math.pi / 2


def build(c):
    res = c['resolution']
    x0, x1 = c['x_range']
    y0, y1 = c['y_range']
    w, h = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    rng = np.random.default_rng(7)
    img = np.clip(c['asphalt_gray'] + rng.normal(0, c['asphalt_noise'], (h, w)), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    def px(p):  # dunya -> piksel (satir 0 = kuzey kenari)
        return np.stack([(p[:, 0] - x0) / res, (y1 - p[:, 1]) / res], axis=1)

    cl = centerline(c)
    lw = max(1, int(round(c['line_width'] / res)))
    lo, hi = c['no_mans_land_y']
    in_nml = (np.abs(cl[:, 0] - c['right_x']) < 1e-6) & (cl[:, 1] > lo) & (cl[:, 1] < hi)
    lines_world = {}
    for name, side in (('left', 1.0), ('right', -1.0)):
        b = offset(cl, side * c['track_width'] / 2)
        # No Man's Land'de cizgi yok: kapali hatti o bolumde kes
        segs, cur = [], []
        for p, skip in zip(b, in_nml):
            if skip:
                if len(cur) > 1:
                    segs.append(np.array(cur))
                cur = []
            else:
                cur.append(p)
        if len(cur) > 1:
            if segs and not in_nml[0]:
                segs[0] = np.vstack([np.array(cur), segs[0]])   # bastaki parcayla birlestir
            else:
                segs.append(np.array(cur))
        lines_world[name] = segs
        for s in segs:
            q = np.round(px(s) * 16).astype(np.int32)
            cv2.polylines(img, [q], False, c['line_gray'], lw, cv2.LINE_AA, shift=4)
    if c['start_line']:
        sy = spawn_pose(c)[1]
        a = px(np.array([[c['left_x'] - c['track_width'] / 2, sy], [c['left_x'] + c['track_width'] / 2, sy]]))
        cv2.line(img, tuple(np.round(a[0]).astype(int)), tuple(np.round(a[1]).astype(int)),
                 c['line_gray'], lw, cv2.LINE_AA)
    for (x, y) in c['potholes']:
        p = px(np.array([[x, y]]))[0]
        cv2.circle(img, tuple(np.round(p).astype(int)), int(round(c['pothole_diameter'] / 2 / res)),
                   c['line_gray'], -1, cv2.LINE_AA)
    length = float(np.sum(np.linalg.norm(np.diff(np.vstack([cl, cl[:1]]), axis=0), axis=1)))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), lines_world, length


DAE = '''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
  <library_images><image id="tex"><init_from>../materials/textures/course.png</init_from></image></library_images>
  <library_effects><effect id="fx"><profile_COMMON>
    <newparam sid="surf"><surface type="2D"><init_from>tex</init_from></surface></newparam>
    <newparam sid="samp"><sampler2D><source>surf</source></sampler2D></newparam>
    <technique sid="common"><lambert>
      <diffuse><texture texture="samp" texcoord="UV"/></diffuse>
    </lambert></technique></profile_COMMON></effect></library_effects>
  <library_materials><material id="mat"><instance_effect url="#fx"/></material></library_materials>
  <library_geometries><geometry id="quad"><mesh>
    <source id="pos"><float_array id="pa" count="12">{x0} {y0} 0 {x1} {y0} 0 {x1} {y1} 0 {x0} {y1} 0</float_array>
      <technique_common><accessor source="#pa" count="4" stride="3">
        <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
    <source id="nrm"><float_array id="na" count="3">0 0 1</float_array>
      <technique_common><accessor source="#na" count="1" stride="3">
        <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
    <source id="uv"><float_array id="ua" count="8">0 0 1 0 1 1 0 1</float_array>
      <technique_common><accessor source="#ua" count="4" stride="2">
        <param name="S" type="float"/><param name="T" type="float"/></accessor></technique_common></source>
    <vertices id="v"><input semantic="POSITION" source="#pos"/></vertices>
    <triangles material="m" count="2">
      <input semantic="VERTEX" source="#v" offset="0"/><input semantic="NORMAL" source="#nrm" offset="1"/>
      <input semantic="TEXCOORD" source="#uv" offset="2" set="0"/>
      <p>0 0 0 1 0 1 2 0 2 0 0 0 2 0 2 3 0 3</p></triangles>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="scene"><node id="ground">
    <instance_geometry url="#quad"><bind_material><technique_common>
      <instance_material symbol="m" target="#mat"><bind_vertex_input semantic="UV" input_semantic="TEXCOORD" input_set="0"/></instance_material>
    </technique_common></bind_material></instance_geometry></node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>
'''

MODEL_CONFIG = '''<?xml version="1.0"?>
<model><name>igvc_course</name><version>1.0</version><sdf version="1.6">model.sdf</sdf>
<description>IGVC Auto-Nav benzeri parkur zemini (scripts/generate_igvc_course.py ile uretildi)</description></model>
'''

def ramp_dae(r):
    """Kapali kesik prizma: y boyunca yamuk profil (cikis, ust duzluk, inis), x boyunca genislik.
    Orijin rampanin zemindeki merkezi. Yuz basina ayri koseler (duz golgeleme)."""
    h, t = r['height'], r['top_length'] / 2
    a = t + h / r['grade']                      # tabanin yari uzunlugu
    hw = r['width'] / 2
    prof = [(-a, 0.0), (-t, h), (t, h), (a, 0.0)]          # (y, z), guneyden kuzeye
    faces = []
    for (y0, z0), (y1, z1) in zip(prof, prof[1:]):          # ust yuzeyler (3 dortgen)
        faces.append([(-hw, y0, z0), (hw, y0, z0), (hw, y1, z1), (-hw, y1, z1)])
    faces.append([(hw, -a, 0.0), (-hw, -a, 0.0), (-hw, a, 0.0), (hw, a, 0.0)])     # taban
    faces.append([(hw, y, z) for y, z in prof])                                     # dogu yani
    faces.append([(-hw, y, z) for y, z in reversed(prof)])                          # bati yani
    pos, nrm, idx = [], [], []
    for f in faces:
        p = np.array(f, float)
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-12:
            n = np.cross(p[2] - p[0], p[3] - p[0])
        n /= np.linalg.norm(n)
        if np.dot(n, p.mean(axis=0) - np.array([0.0, 0.0, h / 2])) < 0:   # disa baksin
            p, n = p[::-1], -n
        base = len(pos)
        pos.extend(p.tolist())
        ni = len(nrm)
        nrm.append(n.tolist())
        for k in range(1, len(p) - 1):                   # yelpaze ucgenleme (disbukey)
            idx.extend([base, ni, base + k, ni, base + k + 1, ni])
    col = ' '.join(f'{v:.3f}' for v in r['color'])
    return f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
  <library_effects><effect id="fx"><profile_COMMON><technique sid="c"><lambert>
    <diffuse><color>{col} 1</color></diffuse></lambert></technique></profile_COMMON></effect></library_effects>
  <library_materials><material id="mat"><instance_effect url="#fx"/></material></library_materials>
  <library_geometries><geometry id="ramp"><mesh>
    <source id="pos"><float_array id="pa" count="{3 * len(pos)}">{' '.join(f'{v:.5f}' for p in pos for v in p)}</float_array>
      <technique_common><accessor source="#pa" count="{len(pos)}" stride="3">
        <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
    <source id="nrm"><float_array id="na" count="{3 * len(nrm)}">{' '.join(f'{v:.5f}' for n in nrm for v in n)}</float_array>
      <technique_common><accessor source="#na" count="{len(nrm)}" stride="3">
        <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
    <vertices id="v"><input semantic="POSITION" source="#pos"/></vertices>
    <triangles material="m" count="{len(idx) // 6}">
      <input semantic="VERTEX" source="#v" offset="0"/><input semantic="NORMAL" source="#nrm" offset="1"/>
      <p>{' '.join(map(str, idx))}</p></triangles>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="scene"><node id="ramp">
    <instance_geometry url="#ramp"><bind_material><technique_common>
      <instance_material symbol="m" target="#mat"/></technique_common></bind_material></instance_geometry>
  </node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>
''', 2 * a


RAMP_LINK = '''
<link name="ramp"><pose>{x} {y} 0 0 0 0</pose>
  <collision name="collision"><geometry><mesh><uri>model://igvc_course/meshes/ramp.dae</uri></mesh></geometry>
    <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
  <visual name="visual"><geometry><mesh><uri>model://igvc_course/meshes/ramp.dae</uri></mesh></geometry></visual>
</link>'''


MODEL_SDF = '''<?xml version="1.0"?>
<sdf version="1.6"><model name="igvc_course"><static>true</static><link name="ground">
  <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
    <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
  <visual name="asphalt"><cast_shadows>false</cast_shadows><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
    <material><ambient>{g} {g} {g} 1</ambient><diffuse>{g} {g} {g} 1</diffuse></material></visual>
  <visual name="course"><pose>0 0 0.001 0 0 0</pose><cast_shadows>false</cast_shadows>
    <geometry><mesh><uri>model://igvc_course/meshes/course.dae</uri></mesh></geometry></visual>
</link>{ramp}</model></sdf>
'''


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, '..', 'models', 'igvc_course')
    for d in ('materials/textures', 'meshes'):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    img, lines, length = build(COURSE)
    cv2.imwrite(os.path.join(out, 'materials/textures/course.png'), img)
    x0, x1 = COURSE['x_range']
    y0, y1 = COURSE['y_range']
    with open(os.path.join(out, 'meshes/course.dae'), 'w') as f:
        f.write(DAE.format(x0=x0, x1=x1, y0=y0, y1=y1))
    # igvc_course_2.world varyanti: ayni parkur (ayni gurultu tohumu), baslangic cizgisi yok.
    img2, _, _ = build(dict(COURSE, start_line=False))
    cv2.imwrite(os.path.join(out, 'materials/textures/course_2.png'), img2)
    with open(os.path.join(out, 'meshes/course_2.dae'), 'w') as f:
        f.write(DAE.format(x0=x0, x1=x1, y0=y0, y1=y1).replace('textures/course.png', 'textures/course_2.png'))
    with open(os.path.join(out, 'model.config'), 'w') as f:
        f.write(MODEL_CONFIG)
    with open(os.path.join(out, 'model.sdf'), 'w') as f:
        ramp_xml, ramp_info = '', ''
        r = COURSE.get('ramp')
        if r:
            dae, ramp_len = ramp_dae(r)
            with open(os.path.join(out, 'meshes/ramp.dae'), 'w') as fr:
                fr.write(dae)
            ramp_xml = RAMP_LINK.format(x=r['center'][0], y=r['center'][1])
            ramp_info = (f", rampa {ramp_len:.2f} m x {r['width']:.2f} m, yukseklik {r['height']:.2f} m, "
                         f"egim %{r['grade'] * 100:.0f} ({r['center'][0]}, {r['center'][1]})")
        f.write(MODEL_SDF.replace('{g}', f"{COURSE['asphalt_gray'] / 255:.3f}").replace('{ramp}', ramp_xml))
    with open(os.path.join(out, 'course.yaml'), 'w') as f:
        f.write(f'# generate_igvc_course.py ciktisi. Merkez hat uzunlugu: {length:.1f} m ({length / FT:.0f} ft)\n')
        sx, sy, syaw = spawn_pose(COURSE)        # baslangic cizgisinin konumu
        sy += COURSE['spawn_ahead_of_line']      # arac cizginin onunde
        f.write(f'spawn: [{sx:.3f}, {sy:.3f}, {syaw:.4f}]   # x, y, yaw -> sim.launch.py (auto)\n')
        for k, v in COURSE.items():
            f.write(f'{k}: {list(v) if isinstance(v, tuple) else v}\n')
    print(f'Yazildi: {os.path.normpath(out)} | doku {img.shape[1]}x{img.shape[0]} px, '
          f'merkez hat {length:.1f} m ({length / FT:.0f} ft){ramp_info}')


if __name__ == '__main__':
    main()
