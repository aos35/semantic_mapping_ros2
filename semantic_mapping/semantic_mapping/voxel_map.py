"""Mapa semántico voxelizado con fusión probabilística de etiquetas.

Cada voxel guarda un histograma de votos por clase. Con un modelo de sensor
simétrico (probabilidad p de acertar la clase, repartida uniformemente si falla)
la actualización bayesiana recursiva de la distribución por voxel es monótona en
el número de votos, así que la clase MAP coincide con la clase más votada.
Esto resuelve de forma natural oclusiones parciales y errores puntuales de la
segmentación 2D cuando un mismo voxel se observa desde varias posiciones.

Sin dependencias de ROS.
"""
from __future__ import annotations

import numpy as np

from .labels import IGNORE, NUM_CLASSES, label_to_rgb

_OFF = 1 << 20          # 21 bits por eje -> ±1M voxels
_MASK = (1 << 21) - 1


def _encode(k: np.ndarray) -> np.ndarray:
    k = k.astype(np.int64) + _OFF
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


class SemanticVoxelMap:
    def __init__(self, voxel_size: float = 0.15, num_classes: int = NUM_CLASSES,
                 capacity: int = 1 << 16):
        self.voxel_size = float(voxel_size)
        self.num_classes = num_classes
        self._index: dict[int, int] = {}
        self._n = 0
        self._alloc(capacity)

    def _alloc(self, cap):
        self.keys = np.zeros((cap, 3), np.int32)
        self.counts = np.zeros((cap, self.num_classes), np.uint32)
        self.rgb_sum = np.zeros((cap, 3), np.float32)
        self.hits = np.zeros(cap, np.uint32)

    def _grow(self, need):
        cap = len(self.hits)
        if need <= cap:
            return
        new = max(need, cap * 2)
        for name in ('keys', 'counts', 'rgb_sum', 'hits'):
            old = getattr(self, name)
            arr = np.zeros((new,) + old.shape[1:], old.dtype)
            arr[:cap] = old
            setattr(self, name, arr)

    def __len__(self):
        return self._n

    def insert(self, xyz: np.ndarray, rgb: np.ndarray, labels: np.ndarray):
        """xyz (N,3) en el frame del mapa, rgb (N,3) uint8, labels (N,) uint8."""
        if len(xyz) == 0:
            return
        k = np.floor(xyz / self.voxel_size).astype(np.int32)
        codes = _encode(k)
        ucodes, first, inv = np.unique(codes, return_index=True, return_inverse=True)

        idx = np.empty(len(ucodes), np.int64)
        index = self._index
        new_pos = []
        for j, c in enumerate(ucodes.tolist()):
            i = index.get(c)
            if i is None:
                i = self._n + len(new_pos)
                index[c] = i
                new_pos.append(j)
            idx[j] = i
        if new_pos:
            self._grow(self._n + len(new_pos))
            new_pos = np.asarray(new_pos)
            self.keys[idx[new_pos]] = k[first[new_pos]]
            self._n += len(new_pos)

        vox = idx[inv.ravel()]
        np.add.at(self.hits, vox, 1)
        np.add.at(self.rgb_sum, vox, rgb.astype(np.float32))
        m = labels < self.num_classes
        if m.any():
            np.add.at(self.counts, (vox[m], labels[m].astype(np.int64)), 1)

    def export(self, min_hits: int = 1):
        """Devuelve centros (M,3), rgb medio (M,3) uint8, etiqueta MAP (M,),
        confianza (M,) = votos de la clase ganadora / votos totales."""
        n = self._n
        sel = self.hits[:n] >= min_hits
        centers = (self.keys[:n][sel].astype(np.float32) + 0.5) * self.voxel_size
        rgb = (self.rgb_sum[:n][sel] / self.hits[:n][sel, None]).astype(np.uint8)
        counts = self.counts[:n][sel]
        total = counts.sum(1)
        label = counts.argmax(1).astype(np.uint8)
        label[total == 0] = IGNORE
        with np.errstate(divide='ignore', invalid='ignore'):
            conf = np.where(total > 0, counts.max(1) / np.maximum(total, 1), 0.0)
        return centers, rgb, label, conf.astype(np.float32)

    def class_histogram(self, min_hits: int = 1) -> dict:
        _, _, label, _ = self.export(min_hits)
        vals, cnt = np.unique(label, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, cnt)}


def save_ply(path: str, xyz: np.ndarray, rgb: np.ndarray, label: np.ndarray | None = None,
             conf: np.ndarray | None = None):
    """PLY binario con x y z red green blue [label] [confidence]."""
    n = len(xyz)
    fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if label is not None:
        fields.append(('label', 'u1'))
    if conf is not None:
        fields.append(('confidence', '<f4'))
    arr = np.empty(n, dtype=fields)
    arr['x'], arr['y'], arr['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr['red'], arr['green'], arr['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if label is not None:
        arr['label'] = label
    if conf is not None:
        arr['confidence'] = conf
    ply_t = {'<f4': 'float', 'u1': 'uchar'}
    header = ['ply', 'format binary_little_endian 1.0', f'element vertex {n}']
    header += [f'property {ply_t[t]} {name}' for name, t in fields]
    header += ['end_header']
    with open(path, 'wb') as f:
        f.write(('\n'.join(header) + '\n').encode())
        f.write(arr.tobytes())


def semantic_colors(label: np.ndarray) -> np.ndarray:
    return label_to_rgb(label)
