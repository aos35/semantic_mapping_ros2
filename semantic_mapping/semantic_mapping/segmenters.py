"""Back-ends de segmentación 2D. Todos devuelven un mapa (H,W) uint8 en el
espacio de etiquetas de labels.py (Cityscapes, 255 = sin etiqueta).

* segformer : segmentación semántica densa (Cityscapes). Cubre edificios,
              vegetación, calzada, señales, vehículos... Recomendado por defecto.
* yolo      : YOLO-seg (COCO). Solo objetos discretos (personas, vehículos,
              semáforos); el resto queda sin etiqueta.
"""
import os

import cv2
import numpy as np

from .labels import COCO_TO_CITYSCAPES, IGNORE


class SegFormerSegmenter:
    def __init__(self, model='nvidia/segformer-b2-finetuned-cityscapes-1024-1024',
                 device='cuda', scale=1.0, fp16=True):
        import torch
        from transformers import SegformerForSemanticSegmentation
        self.torch = torch
        self.device = device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu'
        self.model = SegformerForSemanticSegmentation.from_pretrained(model).to(self.device).eval()
        self.fp16 = fp16 and self.device.startswith('cuda')
        if self.fp16:
            self.model = self.model.half()
        self.scale = scale
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        if self.device.startswith('cuda'):
            torch.backends.cudnn.benchmark = True

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        return self.batch([bgr])[0]

    def batch(self, images):
        """Segmenta una lista de imágenes del mismo tamaño en un único paso de GPU.
        Devuelve mapas de etiquetas del tamaño de la imagen de entrada."""
        torch = self.torch
        h, w = images[0].shape[:2]
        if self.scale != 1.0:
            images = [cv2.resize(im, None, fx=self.scale, fy=self.scale,
                                 interpolation=cv2.INTER_AREA) for im in images]
        x = torch.from_numpy(np.ascontiguousarray(np.stack(images)[..., ::-1])).to(self.device)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std
        if self.fp16:
            x = x.half()
        with torch.inference_mode():
            logits = self.model(pixel_values=x).logits          # (B, C, H/4, W/4)
            logits = torch.nn.functional.interpolate(logits, size=(h, w), mode='bilinear',
                                                     align_corners=False)
            lab = logits.argmax(1).to(torch.uint8).cpu().numpy()
        return list(lab)


class YoloSegmenter:
    def __init__(self, model=os.environ.get('YOLO_MODEL', 'yolo11m-seg.pt'), device='cuda', conf=0.35, imgsz=960):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        out = np.full((h, w), IGNORE, np.uint8)
        res = self.model(bgr, verbose=False, conf=self.conf, imgsz=self.imgsz,
                         device=self.device, retina_masks=True)[0]
        if res.masks is None:
            return out
        masks = res.masks.data.cpu().numpy() > 0.5
        cls = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        for i in np.argsort(confs):          # las más seguras se pintan al final
            lab = COCO_TO_CITYSCAPES.get(int(cls[i]))
            if lab is None:
                continue
            m = masks[i]
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            out[m] = lab
        return out


def build_segmenter(backend: str, **kw):
    if backend == 'segformer':
        return SegFormerSegmenter(**kw)
    if backend == 'yolo':
        return YoloSegmenter(**kw)
    raise ValueError(f'backend desconocido: {backend}')
