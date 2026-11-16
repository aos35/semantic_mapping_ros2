"""Nodo segmentation_node.

Segmenta las imágenes de las cámaras con un modelo preentrenado y publica,
con el mismo header (stamp) que la imagen original:
  /semantic_mapping/camX/labels      Image mono8 (id de clase Cityscapes, 255 = nada)
  /semantic_mapping/camX/labels_viz  Image bgr8  (superposición, solo si hay suscriptores)

Rendimiento: los callbacks solo guardan la última imagen de cada cámara; un
temporizador las procesa juntas en un único lote de GPU. Las imágenes se
decodifican directamente a resolución reducida y las etiquetas se publican a
esa resolución (cloud_projector las reescala a la de la calibración).
"""
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from .labels import label_to_rgb
from .ros_utils import compressed_to_bgr, numpy_to_image, stamp_to_sec
from .segmenters import build_segmenter


class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')
        self.declare_parameter('cameras', ['cam0', 'cam1', 'cam2'])
        self.declare_parameter('image_topic_template',
                               '/alphasense_driver_ros/{cam}/debayered/image/compressed')
        self.declare_parameter('backend', 'segformer')
        self.declare_parameter('model', '')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('decode_reduce', 2)     # 1440x1080 -> 720x540 al decodificar
        self.declare_parameter('scale', 1.0)           # escala extra (solo segformer)
        self.declare_parameter('max_rate_hz', 10.0)    # por cámara; el LiDAR va a 10 Hz
        self.declare_parameter('publish_viz', True)

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        backend = p('backend')
        kw = {'device': p('device')}
        if p('model'):
            kw['model'] = p('model')
        if backend == 'segformer':
            kw['scale'] = p('scale')
        self.get_logger().info(f'Cargando modelo {backend} {kw} ...')
        self.seg = build_segmenter(backend, **kw)
        self.get_logger().info('Modelo cargado')

        self.reduce = p('decode_reduce')
        self.min_dt = 1.0 / p('max_rate_hz') if p('max_rate_hz') > 0 else 0.0
        self.publish_viz = p('publish_viz')
        self.cameras = list(p('cameras'))
        self.latest = {}          # cam -> último mensaje sin procesar
        self.last_done = {}       # cam -> stamp de la última imagen procesada
        self.lock = threading.Lock()
        self.pub, self.pub_viz = {}, {}
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        sub_qos = QoSProfile(depth=2, reliability=qos_profile_sensor_data.reliability)
        rx_group = MutuallyExclusiveCallbackGroup()
        for cam in self.cameras:
            self.pub[cam] = self.create_publisher(Image, f'/semantic_mapping/{cam}/labels', qos)
            if self.publish_viz:
                self.pub_viz[cam] = self.create_publisher(
                    Image, f'/semantic_mapping/{cam}/labels_viz', qos)
            self.create_subscription(CompressedImage, p('image_topic_template').format(cam=cam),
                                     lambda m, c=cam: self.on_image(c, m), sub_qos,
                                     callback_group=rx_group)
        # Temporizador de pared (no de simulación): procesa en cuanto hay imágenes
        self.create_timer(0.005, self.process, callback_group=MutuallyExclusiveCallbackGroup(),
                          clock=rclpy.clock.Clock())
        self.n = 0
        self.n_batches = 0
        self.t_acc = 0.0

    def on_image(self, cam, msg):
        t = stamp_to_sec(msg.header.stamp)
        if t - self.last_done.get(cam, -1e9) < self.min_dt:
            return
        with self.lock:
            self.latest[cam] = msg

    def process(self):
        with self.lock:
            if not self.latest:
                return
            batch, self.latest = self.latest, {}
        t0 = time.perf_counter()
        cams = list(batch)
        imgs = [compressed_to_bgr(batch[c], self.reduce) for c in cams]
        if hasattr(self.seg, 'batch'):
            labs = self.seg.batch(imgs)
        else:
            labs = [self.seg(im) for im in imgs]
        for c, im, lab in zip(cams, imgs, labs):
            msg = batch[c]
            self.last_done[c] = stamp_to_sec(msg.header.stamp)
            self.pub[c].publish(numpy_to_image(np.ascontiguousarray(lab), msg.header, 'mono8'))
            if self.publish_viz and self.pub_viz[c].get_subscription_count() > 0:
                color = label_to_rgb(lab)[..., ::-1]
                viz = cv2.addWeighted(im, 0.45, color, 0.55, 0)
                self.pub_viz[c].publish(numpy_to_image(np.ascontiguousarray(viz), msg.header, 'bgr8'))
        self.n += len(cams)
        self.n_batches += 1
        self.t_acc += time.perf_counter() - t0
        if self.n_batches % 100 == 0:
            self.get_logger().info(
                f'{self.n} imágenes en {self.n_batches} lotes | '
                f'{1000 * self.t_acc / self.n_batches:.1f} ms/lote | '
                f'{self.n / max(self.t_acc, 1e-9):.1f} img/s de GPU')


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
