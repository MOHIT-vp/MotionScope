"""Moving-object detection from foreground masks.

Lucas–Kanade gives *point* motion, not *objects*.  The original script grouped
points into "vehicles" with a 20-px radius rule and re-numbered them every
frame, so IDs and speeds were not persistent.  Here object hypotheses come
from a background model (MOG2) fused with the motion-saliency mask, cleaned
with morphology and turned into bounding boxes.

Blob detections are class-agnostic *moving objects*.  When the optional
YOLOX model is installed, vehicle boxes and class labels come from it instead
and the motion mask is used to keep only vehicles that actually move.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

Box = Tuple[float, float, float, float]  # x, y, w, h


class ForegroundDetector:
    def __init__(self, frame_shape, min_area_frac: float):
        h, w = frame_shape[:2]
        self.min_area = max(40.0, min_area_frac * w * h)
        self.bg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=32, detectShadows=True)
        k = max(3, int(round(w / 160)) | 1)
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
        self.mask = np.zeros((h, w), np.uint8)

    def warm_up(self, frame: np.ndarray) -> None:
        self.bg.apply(cv2.GaussianBlur(frame, (5, 5), 0), learningRate=1.0)

    def detect(self, frame: np.ndarray, saliency_binary: np.ndarray, camera_moving: bool) -> List[Box]:
        fg = self.bg.apply(cv2.GaussianBlur(frame, (5, 5), 0))
        fg = np.where(fg == 255, 255, 0).astype(np.uint8)          # drop shadows (127)
        if camera_moving:
            # A shaking camera corrupts the background model; rely on the
            # (motion-compensated) saliency mask instead for this frame.
            fg = np.zeros_like(fg)
        sal = cv2.dilate(saliency_binary, self.k_open, iterations=2)
        m = cv2.bitwise_or(fg, sal)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k_open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.k_close, iterations=2)
        self.mask = m
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Box] = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 6 or h < 6:
                continue
            boxes.append((float(x), float(y), float(w), float(h)))
        return merge_boxes(boxes)


def iou(a: Box, b: Box) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def containment(a: Box, b: Box) -> float:
    """Fraction of the smaller box covered by the intersection."""
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    small = min(a[2] * a[3], b[2] * b[3])
    return iw * ih / small if small > 0 else 0.0


def merge_boxes(boxes: List[Box], thresh: float = 0.6) -> List[Box]:
    boxes = sorted(boxes, key=lambda b: -b[2] * b[3])
    out: List[Box] = []
    for b in boxes:
        for i, o in enumerate(out):
            if containment(b, o) > thresh:
                x1, y1 = min(b[0], o[0]), min(b[1], o[1])
                x2, y2 = max(b[0] + b[2], o[0] + o[2]), max(b[1] + b[3], o[1] + o[3])
                out[i] = (x1, y1, x2 - x1, y2 - y1)
                break
        else:
            out.append(b)
    return out


# ---------------------------------------------------------------------------
# Learned vehicle detector (optional)
# ---------------------------------------------------------------------------
# Motion blobs merge vehicles that drive close together and miss dark cars on
# dark asphalt.  When the YOLOX-S ONNX model from the official OpenCV model
# zoo is present (scripts/fetch_model.py), it supplies object extents and a
# class label; Lucas–Kanade still supplies all motion measurements and
# carries boxes between detector frames.

COCO_VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
YOLOX_FILE = MODEL_DIR / "object_detection_yolox_2022nov.onnx"


def yolox_available() -> bool:
    return YOLOX_FILE.is_file() and YOLOX_FILE.stat().st_size > 1_000_000


class YoloxDetector:
    size = 640

    def __init__(self, conf: float = 0.4, nms: float = 0.45):
        self.net = cv2.dnn.readNet(str(YOLOX_FILE))
        self.conf = conf
        self.nms = nms
        grids, strides = [], []
        for s in (8, 16, 32):
            n = self.size // s
            yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            g = np.stack((xv, yv), 2).reshape(-1, 2)
            grids.append(g)
            strides.append(np.full((len(g), 1), s))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.stride = np.concatenate(strides).astype(np.float32)

    def detect(self, frame: np.ndarray) -> List[Tuple[Box, str, float]]:
        h, w = frame.shape[:2]
        r = min(self.size / h, self.size / w)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        rs = cv2.resize(frame, (int(w * r), int(h * r)))
        canvas[: rs.shape[0], : rs.shape[1]] = rs
        self.net.setInput(cv2.dnn.blobFromImage(canvas, 1.0, (self.size, self.size), swapRB=False))
        out = self.net.forward()[0]
        xy = (out[:, :2] + self.grid) * self.stride
        wh = np.exp(out[:, 2:4]) * self.stride
        scores = out[:, 4:5] * out[:, 5:]
        cls = scores.argmax(1)
        conf = scores.max(1)
        keep = (conf >= self.conf) & np.isin(cls, list(COCO_VEHICLES))
        if not keep.any():
            return []
        boxes = (np.concatenate([xy - wh / 2, wh], 1)[keep] / r)
        conf, cls = conf[keep], cls[keep]
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), self.conf, self.nms)
        res = []
        for i in np.array(idx).reshape(-1):
            x, y, bw, bh = boxes[i]
            x, y = max(0.0, x), max(0.0, y)
            bw, bh = min(bw, w - x), min(bh, h - y)
            if bw > 4 and bh > 4:
                res.append(((float(x), float(y), float(bw), float(bh)), COCO_VEHICLES[int(cls[i])], float(conf[i])))
        return res


def motion_fraction(mask: np.ndarray, box: Box) -> float:
    x, y, w, h = [int(round(v)) for v in box]
    roi = mask[max(0, y): y + h, max(0, x): x + w]
    return float(np.count_nonzero(roi)) / roi.size if roi.size else 0.0
