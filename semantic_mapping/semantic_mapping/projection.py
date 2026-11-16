"""Proyección de puntos LiDAR sobre cámaras ojo de pez (modelo equidistante de Kalibr).

Este módulo no depende de ROS: solo numpy y yaml, para poder probarlo aislado.

Convenio (README de calibración del Oxford Spires Dataset):
    T_cam_lidar convierte puntos del frame del LiDAR al frame de la cámara:
        p_cam = R @ p_lidar + t
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class FisheyeCamera:
    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray          # [k1, k2, k3, k4] modelo equidistante
    width: int
    height: int
    T_cam_lidar: np.ndarray   # 4x4
    topic: str = ''

    @classmethod
    def from_files(cls, calib_dir: str, name: str) -> 'FisheyeCamera':
        with open(os.path.join(calib_dir, f'{name}.yaml')) as f:
            intr = yaml.safe_load(f)
        with open(os.path.join(calib_dir, 'cam-lidar-imu.yaml')) as f:
            extr = yaml.safe_load(f)[name]
        K = np.array(intr['camera_matrix']['data'], dtype=float).reshape(3, 3)
        if intr.get('distortion_model', 'equidistant') != 'equidistant':
            raise ValueError(f'{name}: solo se soporta el modelo equidistante')
        return cls(
            name=name,
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
            dist=np.array(intr['distortion_coefficients']['data'], dtype=float),
            width=int(intr['image_width']), height=int(intr['image_height']),
            T_cam_lidar=np.array(extr['T_cam_lidar'], dtype=float),
            topic=extr.get('rostopic', ''),
        )

    def project(self, pts_lidar: np.ndarray, min_depth: float = 0.5,
                max_angle_deg: float = 85.0):
        """Proyecta puntos (N,3) del frame LiDAR a píxeles.

        Devuelve (u, v, valid, theta):
          u, v   : coordenadas de píxel (float, N)
          valid  : máscara de puntos delante de la cámara y dentro de la imagen
          theta  : ángulo respecto al eje óptico (rad), útil para elegir la
                   cámara que ve cada punto con menos distorsión.
        """
        R = self.T_cam_lidar[:3, :3]
        t = self.T_cam_lidar[:3, 3]
        pc = pts_lidar @ R.T + t
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

        r = np.hypot(x, y)
        theta = np.arctan2(r, z)
        k1, k2, k3, k4 = self.dist
        th2 = theta * theta
        theta_d = theta * (1 + k1 * th2 + k2 * th2**2 + k3 * th2**3 + k4 * th2**4)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where(r > 1e-9, theta_d / r, 1.0 / np.maximum(z, 1e-9))
        u = self.fx * scale * x + self.cx
        v = self.fy * scale * y + self.cy

        valid = (z > min_depth) & (theta < np.deg2rad(max_angle_deg))
        valid &= (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return u, v, valid, theta


def load_cameras(calib_dir: str, names) -> dict:
    return {n: FisheyeCamera.from_files(calib_dir, n) for n in names}


def sample_image(image: np.ndarray, u: np.ndarray, v: np.ndarray, src_size=None):
    """Lee los píxeles (vecino más cercano). Si la imagen se ha reescalado
    respecto a la calibración, src_size=(w, h) de la calibración."""
    h, w = image.shape[:2]
    if src_size is not None and (w, h) != tuple(src_size):
        u = u * (w / src_size[0])
        v = v * (h / src_size[1])
    ui = np.clip(u.astype(np.int32), 0, w - 1)
    vi = np.clip(v.astype(np.int32), 0, h - 1)
    return image[vi, ui]


def fuse_cameras(cameras: dict, pts: np.ndarray, images: dict, labels: dict | None = None,
                 min_depth: float = 0.5, ignore_label: int = 255):
    """Asigna a cada punto el color (y etiqueta) de la cámara que lo ve con menor
    ángulo respecto a su eje óptico.

    images : {cam: imagen BGR}  (cámaras sin imagen disponible se omiten)
    labels : {cam: mapa de etiquetas uint8} o None
    Devuelve (seen, bgr (N,3) uint8, label (N,) uint8, cam_idx (N,) int8)
    """
    n = len(pts)
    best = np.full(n, np.inf)
    bgr = np.zeros((n, 3), np.uint8)
    lab = np.full(n, ignore_label, np.uint8)
    cam_idx = np.full(n, -1, np.int8)
    for i, (name, cam) in enumerate(cameras.items()):
        img = images.get(name)
        if img is None:
            continue
        u, v, valid, theta = cam.project(pts, min_depth=min_depth)
        better = valid & (theta < best)
        if not better.any():
            continue
        idx = np.nonzero(better)[0]
        best[idx] = theta[idx]
        cam_idx[idx] = i
        bgr[idx] = sample_image(img, u[idx], v[idx], (cam.width, cam.height))
        if labels is not None:
            lm = labels.get(name)
            if lm is not None:
                lab[idx] = sample_image(lm, u[idx], v[idx], (cam.width, cam.height))
            else:
                lab[idx] = ignore_label
    return cam_idx >= 0, bgr, lab, cam_idx
