"""Tests sin ROS de la proyección y del mapa voxelizado.
Ejecutar:  python3 -m pytest semantic_mapping/test"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from semantic_mapping.labels import IGNORE  # noqa: E402
from semantic_mapping.projection import FisheyeCamera, fuse_cameras  # noqa: E402
from semantic_mapping.voxel_map import SemanticVoxelMap  # noqa: E402

CALIB = os.path.join(os.path.dirname(__file__), '..', '..', 'semantic_mapping_bringup',
                     'config', 'calibration')


def _cam(T=np.eye(4), dist=(0, 0, 0, 0)):
    return FisheyeCamera('c', 700., 700., 720., 540., np.array(dist, float), 1440, 1080, T)


def test_principal_point():
    u, v, valid, theta = _cam().project(np.array([[0., 0., 10.]]))
    assert valid[0] and abs(u[0] - 720) < 1e-6 and abs(v[0] - 540) < 1e-6 and theta[0] == 0


def test_behind_camera_is_invalid():
    _, _, valid, _ = _cam().project(np.array([[0., 0., -5.]]))
    assert not valid[0]


def test_matches_opencv_fisheye():
    cv2 = __import__('cv2')
    dist = np.array([-0.041, -0.0024, 0.0034, -0.0018])
    cam = _cam(dist=dist)
    rng = np.random.default_rng(0)
    pts = rng.uniform([-8, -6, 1], [8, 6, 20], size=(500, 3))
    u, v, _, _ = cam.project(pts)
    K = np.array([[700, 0, 720], [0, 700, 540], [0, 0, 1.]])
    ref, _ = cv2.fisheye.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    ref = ref.reshape(-1, 2)
    assert np.abs(ref[:, 0] - u).max() < 1e-3 and np.abs(ref[:, 1] - v).max() < 1e-3


def test_real_calibration_loads():
    if not os.path.isdir(CALIB):
        return
    cam = FisheyeCamera.from_files(CALIB, 'cam0')
    assert cam.width == 1440 and cam.T_cam_lidar.shape == (4, 4)
    # El frame del Hesai está girado 180° en z respecto al frame 'base'
    # (T_base_lidar del README): lo que está delante del vehículo es -x del LiDAR.
    _, _, valid, _ = cam.project(np.array([[-10., 0., 0.], [10., 0., 0.]]))
    assert valid[0] and not valid[1]


def test_fuse_prefers_central_camera():
    T2 = np.eye(4)
    T2[:3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]   # cámara girada 90°
    cams = {'a': _cam(), 'b': _cam(T2)}
    pts = np.array([[0.2, 0., 10.]])
    imgs = {'a': np.full((1080, 1440, 3), 10, np.uint8), 'b': np.full((1080, 1440, 3), 200, np.uint8)}
    labs = {'a': np.full((1080, 1440), 2, np.uint8), 'b': np.full((1080, 1440), 8, np.uint8)}
    seen, bgr, lab, idx = fuse_cameras(cams, pts, imgs, labs)
    assert seen[0] and idx[0] == 0 and lab[0] == 2 and bgr[0, 0] == 10


def test_voxel_majority_vote():
    m = SemanticVoxelMap(voxel_size=1.0, num_classes=19)
    xyz = np.array([[0.1, 0.1, 0.1]] * 5 + [[5.5, 0, 0]], np.float32)
    rgb = np.full((6, 3), 100, np.uint8)
    m.insert(xyz, rgb, np.array([2, 2, 8, 2, IGNORE, IGNORE], np.uint8))
    m.insert(xyz[:1], rgb[:1], np.array([8], np.uint8))
    centers, _, label, conf = m.export()
    assert len(m) == 2
    i = int(np.argmin(np.linalg.norm(centers - 0.5, axis=1)))
    assert label[i] == 2 and abs(conf[i] - 3 / 5) < 1e-6
    assert label[1 - i] == IGNORE


def test_voxel_growth():
    m = SemanticVoxelMap(voxel_size=0.1, capacity=4)
    pts = np.random.default_rng(1).uniform(-50, 50, (20000, 3)).astype(np.float32)
    m.insert(pts, np.zeros((len(pts), 3), np.uint8), np.zeros(len(pts), np.uint8))
    assert len(m) == len(np.unique(np.floor(pts / 0.1).astype(int), axis=0))
