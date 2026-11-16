"""Espacio de etiquetas común: las 19 clases de Cityscapes.

Se eligen porque cubren justo las clases que pide el guion (edificios,
vegetación, vehículos, señales) y existen modelos preentrenados para ellas.
Las detecciones de YOLO (COCO) se traducen a este mismo espacio.
"""
import numpy as np

IGNORE = 255

CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle',
]
NUM_CLASSES = len(CLASSES)

# Paleta de Cityscapes (RGB) con un único cambio: 'building' pasa del gris
# oscuro oficial (70,70,70) a ocre, porque sobre el fondo oscuro de RViz no se
# distinguía de los puntos sin etiqueta.
PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [214, 140, 60], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
    [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)
UNLABELED_RGB = np.array([128, 128, 128], np.uint8)

# Índices de clase COCO (ultralytics) -> Cityscapes
COCO_TO_CITYSCAPES = {
    0: 11,   # person
    1: 18,   # bicycle
    2: 13,   # car
    3: 17,   # motorcycle
    5: 15,   # bus
    6: 16,   # train
    7: 14,   # truck
    9: 6,    # traffic light
    11: 7,   # stop sign
}


def label_to_rgb(labels: np.ndarray) -> np.ndarray:
    """(N,) o (H,W) uint8 -> (...,3) uint8 RGB; IGNORE en gris."""
    lut = np.tile(UNLABELED_RGB, (256, 1))
    lut[:NUM_CLASSES] = PALETTE
    return lut[labels]


def label_id(name: str) -> int:
    return CLASSES.index(name)
