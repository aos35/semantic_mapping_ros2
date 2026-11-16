"""Nodo semantic_mapper.

Acumula las nubes etiquetadas en el frame global usando la pose de rko_lio
(TF odom -> pandar) y las fusiona en un mapa voxelizado (voxel_map.py).

Entrada : /semantic_mapping/cloud (PointCloud2 x y z rgb label, frame del LiDAR)
          TF odom -> pandar publicado por rko_lio
Salida  : /semantic_mapping/map      (PointCloud2 en 'odom', publicado periódicamente)
          ~/save (std_srvs/Trigger)  guarda mapa y estadísticas en output_dir
Al cerrar el nodo se guardan automáticamente:
          semantic_map.ply, rgb_map.ply, trajectory.csv, stats.json
"""
import json
import os
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from .labels import CLASSES, IGNORE, label_to_rgb
from .ros_utils import make_cloud, quat_to_matrix, semantic_cloud_to_arrays, stamp_to_sec
from .voxel_map import SemanticVoxelMap, save_ply


class SemanticMapper(Node):
    def __init__(self):
        super().__init__('semantic_mapper')
        self.declare_parameter('input_topic', '/semantic_mapping/cloud')
        self.declare_parameter('map_topic', '/semantic_mapping/map')
        self.declare_parameter('map_frame', 'odom')
        self.declare_parameter('voxel_size', 0.15)
        # sky (10), person (11), rider (12): el cielo no puede tener retornos
        # LiDAR (son errores de borde) y personas/ciclistas son dinámicos.
        self.declare_parameter('exclude_labels', [10, 11, 12])
        self.declare_parameter('color_mode', 'semantic')   # 'semantic' | 'rgb'
        self.declare_parameter('publish_period', 3.0)
        self.declare_parameter('min_hits', 2)
        self.declare_parameter('output_dir', os.path.expanduser('~/output'))

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.map_frame = p('map_frame')
        self.exclude = np.array(p('exclude_labels'), dtype=np.uint8)
        self.color_mode = p('color_mode')
        self.min_hits = p('min_hits')
        self.output_dir = p('output_dir')
        self.vmap = SemanticVoxelMap(p('voxel_size'))

        self.tf_buffer = Buffer(cache_time=Duration(seconds=30))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pending = deque(maxlen=50)
        self.create_subscription(PointCloud2, p('input_topic'),
                                 lambda m: self.pending.append(m), 10)
        self.pub = self.create_publisher(PointCloud2, p('map_topic'), 1)
        self.create_timer(0.05, self.process_pending)
        self.create_timer(p('publish_period'), self.publish_map)
        self.create_service(Trigger, '~/save', self.on_save)

        self.trajectory = []
        self.n_scans = 0
        self.n_tf_fail = 0
        self.t_acc = 0.0
        self.get_logger().info(f'semantic_mapper listo (voxel={p("voxel_size")} m, '
                               f'frame={self.map_frame}, salida={self.output_dir})')

    # ------------------------------------------------------------------ fusión
    def process_pending(self):
        while self.pending:
            msg = self.pending[0]
            try:
                tf = self.tf_buffer.lookup_transform(self.map_frame, msg.header.frame_id,
                                                     Time.from_msg(msg.header.stamp))
            except TransformException:
                # rko_lio publica la pose tras procesar el scan: esperamos un poco
                age = stamp_to_sec(self.get_clock().now().to_msg()) - stamp_to_sec(msg.header.stamp)
                if age < 2.0:
                    return
                self.pending.popleft()
                self.n_tf_fail += 1
                if self.n_tf_fail % 10 == 1:
                    self.get_logger().warn(f'Sin TF {self.map_frame}<-{msg.header.frame_id} '
                                           f'({self.n_tf_fail} scans descartados)')
                continue
            self.pending.popleft()
            self.integrate(msg, tf)

    def integrate(self, msg, tf):
        t0 = time.perf_counter()
        xyz, rgb, lab = semantic_cloud_to_arrays(msg)
        keep = ~np.isin(lab, self.exclude)
        xyz, rgb, lab = xyz[keep], rgb[keep], lab[keep]
        tr = tf.transform.translation
        R = quat_to_matrix(tf.transform.rotation)
        t = np.array([tr.x, tr.y, tr.z])
        self.vmap.insert(xyz @ R.T + t, rgb, lab)
        self.trajectory.append((stamp_to_sec(msg.header.stamp), tr.x, tr.y, tr.z))
        self.n_scans += 1
        self.t_acc += time.perf_counter() - t0
        if self.n_scans % 50 == 0:
            self.get_logger().info(f'{self.n_scans} scans | {len(self.vmap)} voxels | '
                                   f'{1000 * self.t_acc / self.n_scans:.1f} ms/scan')

    # ------------------------------------------------------------------ salida
    def publish_map(self):
        if len(self.vmap) == 0 or self.pub.get_subscription_count() == 0:
            return
        xyz, rgb, lab, _ = self.vmap.export(self.min_hits)
        if self.color_mode == 'semantic':
            rgb = label_to_rgb(lab)
        h = Header(frame_id=self.map_frame, stamp=self.get_clock().now().to_msg())
        self.pub.publish(make_cloud(h, xyz, rgb, lab))

    def save(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        xyz, rgb, lab, conf = self.vmap.export(self.min_hits)
        save_ply(os.path.join(self.output_dir, 'semantic_map.ply'), xyz, label_to_rgb(lab), lab, conf)
        save_ply(os.path.join(self.output_dir, 'rgb_map.ply'), xyz, rgb, lab, conf)
        np.savetxt(os.path.join(self.output_dir, 'trajectory.csv'), np.array(self.trajectory),
                   delimiter=',', header='t,x,y,z', comments='', fmt='%.6f')
        hist = {}
        vals, cnt = np.unique(lab, return_counts=True)
        for v, c in zip(vals, cnt):
            hist['unlabeled' if v == IGNORE else CLASSES[v]] = int(c)
        labeled = lab != IGNORE
        stats = {
            'scans_integrated': self.n_scans,
            'scans_without_tf': self.n_tf_fail,
            'voxels': int(len(xyz)),
            'voxel_size': self.vmap.voxel_size,
            'labeled_ratio': float(labeled.mean()) if len(lab) else 0.0,
            'mean_confidence_labeled': float(conf[labeled].mean()) if labeled.any() else 0.0,
            'mapper_ms_per_scan': 1000 * self.t_acc / max(self.n_scans, 1),
            'voxels_per_class': hist,
        }
        with open(os.path.join(self.output_dir, 'stats.json'), 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        return f'{len(xyz)} voxels guardados en {self.output_dir}'

    def on_save(self, _req, res):
        try:
            res.message = self.save()
            res.success = True
        except Exception as e:  # noqa: BLE001
            res.success, res.message = False, str(e)
        return res


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapper()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if len(node.vmap):
            print('[semantic_mapper] ' + node.save(), flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
