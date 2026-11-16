#!/usr/bin/env python3
"""Proyecta un scan del LiDAR sobre las tres cámaras leyendo el rosbag
directamente (sqlite), sin ROS. Sirve para validar la calibración y generar
figuras para la memoria.

    python3 tools/project_frame.py --bag ../1710255615_2024-03-12-15-00-16 \
        --calib semantic_mapping_bringup/config/calibration --scan 600 --out proj.jpg

--color depth   puntos coloreados por distancia (validación de la calibración)
--color image   puntos con el color del píxel (paso 1: nube coloreada)
"""
import argparse
import glob
import os
import sqlite3
import struct
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'semantic_mapping'))
from semantic_mapping.projection import load_cameras, sample_image  # noqa: E402

CAMS = ['cam1', 'cam0', 'cam2']   # izquierda, frontal, derecha
IMG_TOPIC = '/alphasense_driver_ros/{}/debayered/image/compressed'
LIDAR_TOPIC = '/hesai/pandar'


def _u32(b, o):
    o = (o + 3) // 4 * 4
    return struct.unpack_from('<I', b, o)[0], o + 4


def _str(b, o):
    n, o = _u32(b, o)
    return b[o:o + n - 1].decode(), o + n


def _header(b):
    sec, ns = struct.unpack_from('<iI', b, 0)
    fid, o = _str(b, 8)
    return sec + ns * 1e-9, o


def decode_cloud(raw):
    b = raw[4:]                      # cabecera de encapsulación CDR
    _, o = _header(b)
    _, o = _u32(b, o)                # height
    _, o = _u32(b, o)                # width
    nf, o = _u32(b, o)
    offs = {}
    for _ in range(nf):
        name, o = _str(b, o)
        off, o = _u32(b, o)
        o += 1                       # datatype
        _, o = _u32(b, o)            # count
        offs[name] = off
    o += 1                           # is_bigendian
    ps, o = _u32(b, o)
    _, o = _u32(b, o)                # row_step
    ln, o = _u32(b, o)
    arr = np.frombuffer(b[o:o + ln], np.uint8).reshape(-1, ps)
    xyz = np.stack([arr[:, offs[c]:offs[c] + 4].copy().view(np.float32)[:, 0] for c in 'xyz'], 1)
    return xyz[np.isfinite(xyz).all(1)]


def decode_image(raw):
    b = raw[4:]
    _, o = _header(b)
    _, o = _str(b, o)                # format
    ln, o = _u32(b, o)
    return cv2.imdecode(np.frombuffer(b[o:o + ln], np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--scan', type=int, default=600, help='índice del scan LiDAR')
    ap.add_argument('--color', choices=['depth', 'image'], default='depth')
    ap.add_argument('--max-depth', type=float, default=30.0)
    ap.add_argument('--out', default='proj.jpg')
    a = ap.parse_args()

    db = glob.glob(os.path.join(a.bag, '*.db3'))[0]
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    tid = dict(con.execute('select name, id from topics').fetchall())
    ts, raw = con.execute('select timestamp, data from messages where topic_id=? '
                          'order by timestamp limit 1 offset ?', (tid[LIDAR_TOPIC], a.scan)).fetchone()
    xyz = decode_cloud(raw)
    cams = load_cameras(a.calib, CAMS)

    tiles = []
    for name in CAMS:
        ti, raw = con.execute('select timestamp, data from messages where topic_id=? '
                              'order by abs(timestamp - ?) limit 1',
                              (tid[IMG_TOPIC.format(name)], ts)).fetchone()
        img = decode_image(raw)
        u, v, valid, _ = cams[name].project(xyz)
        pts = xyz[valid]
        if a.color == 'depth':
            d = np.linalg.norm(pts, axis=1)
            g = np.clip(255 - d / a.max_depth * 255, 0, 255).astype(np.uint8)
            col = cv2.applyColorMap(g.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
            canvas = img.copy()
        else:
            col = sample_image(img, u[valid], v[valid])
            canvas = np.zeros_like(img)
        for (x, y), c in zip(np.stack([u[valid], v[valid]], 1).astype(int), col):
            cv2.circle(canvas, (int(x), int(y)), 2 if a.color == 'depth' else 5, tuple(int(k) for k in c), -1)
        print(f'{name}: dt imagen-LiDAR = {(ti - ts) * 1e-9:+.3f} s, {valid.sum()} puntos proyectados')
        tiles.append(cv2.resize(canvas, (720, 540)))
    cv2.imwrite(a.out, np.hstack(tiles))
    print('guardado', a.out)


if __name__ == '__main__':
    main()
