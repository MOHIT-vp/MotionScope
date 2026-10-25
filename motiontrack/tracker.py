"""Persistent multi-object tracking (SORT-style, guided by Lucas–Kanade).

The original project report cites SORT (Bewley et al., 2016).  This tracker follows the
same predict → associate → update loop, but the *prediction* step uses the
median Lucas–Kanade displacement of the corners inside each box, so optical
flow drives the motion model and detections correct the box geometry.
When too few corners are inside a box it falls back to constant velocity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .detection import Box, iou
from .flow import FlowSample

Det = Tuple[Box, str, Optional[float]]   # box, class label, detector confidence


@dataclass
class Track:
    tid: int
    box: Box
    velocity: Tuple[float, float] = (0.0, 0.0)   # px / frame, camera-compensated
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    first_frame: int = 0
    # per-frame history: (frame, t_s, cx, cy, bottom_x, bottom_y, w, h, vx, vy, n_flow, matched)
    history: List[tuple] = field(default_factory=list)
    flow_support: int = 0
    labels: Counter = field(default_factory=Counter)
    confs: List[float] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.labels.most_common(1)[0][0] if self.labels else "object"

    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2, y + h / 2

    @property
    def ground_point(self) -> Tuple[float, float]:
        """Bottom-centre of the box: the best single-point proxy for road contact."""
        x, y, w, h = self.box
        return x + w / 2, y + h


class MultiObjectTracker:
    def __init__(self, frame_shape, iou_threshold: float = 0.2, max_age: int = 12, min_hits: int = 4):
        self.h, self.w = frame_shape[:2]
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self.finished: List[Track] = []
        self._next = 1

    def step(self, detections: Optional[Sequence[Det]], flow: Sequence[FlowSample],
             frame_idx: int, t: float) -> List[Track]:
        """Advance one frame.  ``detections=None`` means the detector was not run
        on this frame: tracks are propagated by Lucas–Kanade only and are not
        penalised for the missing detection."""
        detector_ran = detections is not None
        detections = list(detections or [])
        # 1. predict with LK
        pts_old = np.array([s.old for s in flow], np.float32).reshape(-1, 2)
        motion = np.array([s.motion for s in flow], np.float32).reshape(-1, 2)
        new_pos = np.array([s.new for s in flow], np.float32).reshape(-1, 2)
        for tr in self.tracks:
            x, y, w, h = tr.box
            if len(pts_old):
                # shrink box a little so road/background corners at the edge are excluded
                inside = ((pts_old[:, 0] > x + 0.1 * w) & (pts_old[:, 0] < x + 0.9 * w) &
                          (pts_old[:, 1] > y + 0.1 * h) & (pts_old[:, 1] < y + 0.9 * h))
            else:
                inside = np.zeros(0, bool)
            n = int(inside.sum())
            tr.flow_support = n
            if n >= 3:
                d_img = np.median(new_pos[inside] - pts_old[inside], axis=0)
                d_obj = np.median(motion[inside], axis=0)
                tr.velocity = (0.6 * float(d_obj[0]) + 0.4 * tr.velocity[0],
                               0.6 * float(d_obj[1]) + 0.4 * tr.velocity[1])
                dx, dy = float(d_img[0]), float(d_img[1])
            else:
                dx, dy = tr.velocity
            tr.box = (x + dx, y + dy, w, h)

        # 2. associate (greedy on IoU, highest first)
        pairs = []
        for i, tr in enumerate(self.tracks):
            for j, det in enumerate(detections):
                v = iou(tr.box, det[0])
                if v >= self.iou_threshold:
                    pairs.append((v, i, j))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, i, j in pairs:
            if i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            tr = self.tracks[i]
            px, py, pw, ph = tr.box
            (dx_, dy_, dw, dh), lab, conf = detections[j]
            tr.labels[lab] += 1
            if conf is not None:
                tr.confs.append(conf)
            # Blend: detection fixes size/position drift, LK prediction smooths jitter.
            a = 0.65
            tr.box = (a * dx_ + (1 - a) * px, a * dy_ + (1 - a) * py,
                      a * dw + (1 - a) * pw, a * dh + (1 - a) * ph)
            tr.hits += 1
            tr.misses = 0
            if tr.hits >= self.min_hits:
                tr.confirmed = True

        # 3. unmatched tracks coast; drop stale / out-of-frame ones
        alive: List[Track] = []
        for i, tr in enumerate(self.tracks):
            if i not in used_t and detector_ran:
                tr.misses += 1
            cx, cy = tr.center
            gone = not (0 <= cx < self.w and 0 <= cy < self.h)
            if tr.misses > self.max_age or gone:
                if tr.confirmed:
                    self.finished.append(tr)
                continue
            alive.append(tr)
        self.tracks = alive

        # 4. new tracks for unmatched detections not overlapping a live track
        for j, (box, lab, conf) in enumerate(detections):
            if j in used_d:
                continue
            if any(iou(box, tr.box) > 0.1 for tr in self.tracks):
                continue
            tr = Track(self._next, box, first_frame=frame_idx)
            tr.labels[lab] += 1
            if conf is not None:
                tr.confs.append(conf)
            self.tracks.append(tr)
            self._next += 1

        # 5. record history (only for tracks updated or still coasting)
        for tr in self.tracks:
            # Light exponential smoothing: blob outlines flicker frame to frame,
            # which would otherwise show up as zig-zag trajectories.
            cx, cy = tr.center
            gx, gy = tr.ground_point
            if tr.history:
                p = tr.history[-1]
                cx, cy = 0.5 * cx + 0.5 * (p[2] + tr.velocity[0]), 0.5 * cy + 0.5 * (p[3] + tr.velocity[1])
                gx, gy = 0.5 * gx + 0.5 * (p[4] + tr.velocity[0]), 0.5 * gy + 0.5 * (p[5] + tr.velocity[1])
            tr.history.append((frame_idx, t, cx, cy, gx, gy, tr.box[2], tr.box[3],
                               tr.velocity[0], tr.velocity[1], tr.flow_support, tr.misses == 0))
        return [tr for tr in self.tracks if tr.confirmed and tr.misses == 0]

    def close(self) -> List[Track]:
        self.finished.extend(tr for tr in self.tracks if tr.confirmed)
        self.tracks = []
        return sorted(self.finished, key=lambda t: t.tid)

    def find(self, tid: int) -> Optional[Track]:
        for tr in self.tracks:
            if tr.tid == tid:
                return tr
        return None
