"""Nodo cloud_projector.

Proyecta cada scan del LiDAR sobre las tres cámaras y asigna a cada punto
  * el color del píxel (modo 'color', paso 1 de la práctica), y
  * la etiqueta semántica del píxel (modo 'semantic', usando los mapas de
    etiquetas que publica segmentation_node).

Entrada : /hesai/pandar (PointCloud2), /alphasense_driver_ros/camX/... (CompressedImage),
          /semantic_mapping/camX/labels (Image mono8, solo modo semantic)
Salida  : /semantic_mapping/cloud (PointCloud2 x y z rgb label, frame del LiDAR)
"""
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, PointCloud2

from .labels import IGNORE
from .projection import fuse_cameras, load_cameras
from .ros_utils import (cloud_to_xyz, compressed_to_bgr, image_to_numpy,
                        make_cloud, stamp_to_sec)


def _nearest(buf, t, tol):
    best, best_dt = None, tol
    for ts, msg in buf:
        dt = abs(ts - t)
        if dt <= best_dt:
            best, best_dt = msg, dt
    return best


class CloudProjector(Node):
    def __init__(self):
        super().__init__('cloud_projector')
        self.declare_parameter('calibration_dir', '')
        self.declare_parameter('cameras', ['cam0', 'cam1', 'cam2'])
        self.declare_parameter('lidar_topic', '/hesai/pandar')
        self.declare_parameter('image_topic_suffix', '/debayered/image/compressed')
        self.declare_parameter('label_topic_template', '/semantic_mapping/{cam}/labels')
        self.declare_parameter('use_labels', True)
        self.declare_parameter('max_time_diff', 0.06)
        self.declare_parameter('label_wait', 0.8)   # s máx. esperando etiquetas de la GPU
        self.declare_parameter('min_depth', 0.5)
        self.declare_parameter('max_range', 60.0)
        self.declare_parameter('drop_unseen', True)
        self.declare_parameter('output_topic', '/semantic_mapping/cloud')
        # Decodificar los JPEG a 1/2 de resolución: era el mayor coste por scan
        self.declare_parameter('image_decode_reduce', 2)

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        calib_dir = p('calibration_dir')
        if not calib_dir:
            from ament_index_python.packages import get_package_share_directory
            calib_dir = get_package_share_directory('semantic_mapping_bringup') + '/config/calibration'
        self.cameras = load_cameras(calib_dir, p('cameras'))
        self.use_labels = p('use_labels')
        self.tol = p('max_time_diff')
        self.min_depth = p('min_depth')
        self.max_range = p('max_range')
        self.drop_unseen = p('drop_unseen')
        self.label_wait = p('label_wait')
        self.reduce = p('image_decode_reduce')

        self.img_buf = {c: deque(maxlen=40) for c in self.cameras}
        self.lab_buf = {c: deque(maxlen=40) for c in self.cameras}
        for name, cam in self.cameras.items():
            self.create_subscription(
                CompressedImage, cam.topic + p('image_topic_suffix'),
                lambda m, c=name: self.img_buf[c].append((stamp_to_sec(m.header.stamp), m)),
                qos_profile_sensor_data)
            if self.use_labels:
                self.create_subscription(
                    Image, p('label_topic_template').format(cam=name),
                    lambda m, c=name: self.lab_buf[c].append((stamp_to_sec(m.header.stamp), m)),
                    10)
        self.pending = deque(maxlen=50)
        self.create_subscription(PointCloud2, p('lidar_topic'),
                                 lambda m: self.pending.append(m), qos_profile_sensor_data)
        self.create_timer(0.02, self.process_pending)
        self.pub = self.create_publisher(PointCloud2, p('output_topic'), 10)

        self.n_scans = 0
        self.t_acc = 0.0
        self.n_skipped = 0
        self.get_logger().info(
            f'cloud_projector listo: cámaras={list(self.cameras)}, '
            f'modo={"semantic" if self.use_labels else "color"}, calib={calib_dir}')

    def _labels_ready(self, t):
        """True si todas las cámaras con imagen cercana tienen ya su etiqueta."""
        for c in self.cameras:
            if _nearest(self.img_buf[c], t, self.tol) is not None and \
                    _nearest(self.lab_buf[c], t, self.tol) is None:
                return False
        return True

    def process_pending(self):
        while self.pending:
            msg = self.pending[0]
            t = stamp_to_sec(msg.header.stamp)
            if self.use_labels:
                newest = max((b[-1][0] for b in self.img_buf.values() if b), default=t)
                if not self._labels_ready(t) and newest - t < self.label_wait:
                    return  # aún esperando a la segmentación
            self.pending.popleft()
            self.on_cloud(msg)

    def on_cloud(self, msg: PointCloud2):
        t0 = time.perf_counter()
        t = stamp_to_sec(msg.header.stamp)
        images, labels = {}, ({} if self.use_labels else None)
        for c in self.cameras:
            im = _nearest(self.img_buf[c], t, self.tol)
            if im is None:
                continue
            if self.use_labels:
                lm = _nearest(self.lab_buf[c], t, self.tol)
                if lm is None:
                    continue  # sin etiqueta para esta cámara: no la usamos
                labels[c] = image_to_numpy(lm)
            images[c] = compressed_to_bgr(im, self.reduce)
        if not images:
            self.n_skipped += 1
            if self.n_skipped % 20 == 1:
                self.get_logger().warn('Scan sin imágenes sincronizadas (¿segmentación aún cargando?)')
            return

        xyz = cloud_to_xyz(msg)
        rng = np.linalg.norm(xyz, axis=1)
        xyz = xyz[rng < self.max_range]
        seen, bgr, lab, _ = fuse_cameras(self.cameras, xyz, images, labels,
                                         min_depth=self.min_depth, ignore_label=IGNORE)
        rgb = bgr[:, ::-1]
        if self.drop_unseen:
            xyz, rgb, lab = xyz[seen], rgb[seen], lab[seen]
        else:
            rgb[~seen] = 128
        self.pub.publish(make_cloud(msg.header, xyz, rgb, lab))

        self.n_scans += 1
        self.t_acc += time.perf_counter() - t0
        if self.n_scans % 50 == 0:
            self.get_logger().info(
                f'{self.n_scans} scans | {1000 * self.t_acc / self.n_scans:.1f} ms/scan | '
                f'{len(xyz)} pts coloreados en el último')


def main(args=None):
    rclpy.init(args=args)
    node = CloudProjector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
