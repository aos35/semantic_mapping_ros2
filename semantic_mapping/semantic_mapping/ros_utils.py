"""Conversión entre mensajes ROS 2 y arrays numpy (sin cv_bridge, para evitar
conflictos de versiones de numpy dentro del contenedor)."""
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField

_PF_TO_NP = {
    PointField.INT8: 'i1', PointField.UINT8: 'u1', PointField.INT16: '<i2',
    PointField.UINT16: '<u2', PointField.INT32: '<i4', PointField.UINT32: '<u4',
    PointField.FLOAT32: '<f4', PointField.FLOAT64: '<f8',
}


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    """PointCloud2 -> (N,3) float32 sin NaN. Funciona con campos mixtos
    (el Hesai trae float32 + float64 timestamp + uint16 ring)."""
    fields = {f.name: f for f in msg.fields}
    dtype = np.dtype({
        'names': ['x', 'y', 'z'],
        'formats': [_PF_TO_NP[fields[c].datatype] for c in 'xyz'],
        'offsets': [fields[c].offset for c in 'xyz'],
        'itemsize': msg.point_step,
    })
    arr = np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float32)
    return xyz[np.isfinite(xyz).all(1)]


def pack_rgb(rgb: np.ndarray) -> np.ndarray:
    """(N,3) uint8 RGB -> (N,) float32 con el empaquetado 0x00RRGGBB de PCL/RViz."""
    rgb = rgb.astype(np.uint32)
    packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    return packed.view(np.float32)


_CLOUD_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                         ('rgb', '<f4'), ('label', 'u1'), ('_pad', 'u1', 3)])
_CLOUD_FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='label', offset=16, datatype=PointField.UINT8, count=1),
]


def make_cloud(header, xyz: np.ndarray, rgb: np.ndarray, label: np.ndarray) -> PointCloud2:
    arr = np.zeros(len(xyz), dtype=_CLOUD_DTYPE)
    arr['x'], arr['y'], arr['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr['rgb'] = pack_rgb(rgb)
    arr['label'] = label
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(arr)
    msg.fields = _CLOUD_FIELDS
    msg.is_bigendian = False
    msg.point_step = _CLOUD_DTYPE.itemsize
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = arr.tobytes()
    return msg


def semantic_cloud_to_arrays(msg: PointCloud2):
    """Inversa de make_cloud: devuelve xyz, rgb (N,3) uint8, label (N,)."""
    arr = np.frombuffer(bytes(msg.data), dtype=_CLOUD_DTYPE, count=msg.width * msg.height)
    xyz = np.stack([arr['x'], arr['y'], arr['z']], 1)
    packed = arr['rgb'].view(np.uint32)
    rgb = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], 1).astype(np.uint8)
    return xyz, rgb, arr['label'].copy()


_REDUCED = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
            4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}


def compressed_to_bgr(msg: CompressedImage, reduce: int = 1) -> np.ndarray:
    """Decodifica el JPEG. reduce=2/4/8 decodifica directamente a menor
    resolución (mucho más rápido que decodificar y reescalar)."""
    return cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), _REDUCED[reduce])


def image_to_numpy(msg: Image) -> np.ndarray:
    ch = {'mono8': 1, 'bgr8': 3, 'rgb8': 3}[msg.encoding]
    arr = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    arr = arr[:, :msg.width * ch]
    return arr.reshape(msg.height, msg.width, ch) if ch > 1 else arr


def numpy_to_image(arr: np.ndarray, header, encoding: str) -> Image:
    msg = Image()
    msg.header = header
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = msg.width * (arr.shape[2] if arr.ndim == 3 else 1)
    msg.data = np.ascontiguousarray(arr).tobytes()
    return msg


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
