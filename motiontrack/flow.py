"""Sparse Lucas–Kanade feature tracking.

Improvements over the original speed/saliency loops, which never advanced
``p0`` to the tracked positions (``p0 = good_new`` only exists in the plain
keypoint script) and re-detected corners whenever the saliency-change ratio
exceeded 0.1, so no multi-frame trajectory per point existed:

* corners persist across frames and keep a stable ID, so real trajectories
  can be drawn and measured;
* a forward–backward consistency check rejects unreliable LK solutions;
* new corners are only sampled inside the motion-saliency mask (original
  idea) and away from already-tracked corners;
* corners that stay still for a while (background) are retired at each
  ``recalc_interval``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np


@dataclass
class FlowSample:
    fid: int
    old: np.ndarray          # (2,) position in previous frame
    new: np.ndarray          # (2,) position in current frame
    motion: np.ndarray       # (2,) displacement with camera motion removed


class FeatureTracker:
    def __init__(self, cfg, frame_shape):
        self.cfg = cfg
        self.h, self.w = frame_shape[:2]
        self.feature_params = dict(maxCorners=cfg.max_corners, qualityLevel=cfg.quality_level,
                                   minDistance=cfg.min_distance, blockSize=cfg.block_size)
        self.lk_params = dict(
            winSize=(cfg.lk_win_size, cfg.lk_win_size), maxLevel=cfg.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.lk_iters, cfg.lk_eps),
        )
        self.pts = np.empty((0, 2), np.float32)
        self.ids = np.empty((0,), np.int64)
        self.still = np.empty((0,), np.int32)
        self.trails: Dict[int, Deque[tuple]] = {}
        self.next_id = 0
        self.total_ids = 0

    # ------------------------------------------------------------------
    def step(self, prev_gray: np.ndarray, gray: np.ndarray, saliency_mask: np.ndarray,
             frame_idx: int, cam: Optional[np.ndarray] = None) -> List[FlowSample]:
        samples: List[FlowSample] = []
        if len(self.pts):
            p0 = self.pts.reshape(-1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **self.lk_params)
            p0r, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **self.lk_params)
            fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
            p1 = p1.reshape(-1, 2)
            ok = (st.reshape(-1) == 1) & (st_b.reshape(-1) == 1) & (fb < self.cfg.fb_threshold)
            ok &= (p1[:, 0] >= 0) & (p1[:, 0] < self.w) & (p1[:, 1] >= 0) & (p1[:, 1] < self.h)

            old = self.pts[ok]
            new = p1[ok]
            ids = self.ids[ok]
            still = self.still[ok]
            if cam is not None:
                expected = old @ cam[:, :2].T + cam[:, 2]
            else:
                expected = old
            motion = new - expected
            mag = np.linalg.norm(motion, axis=1)
            still = np.where(mag < 0.25, still + 1, 0)

            for fid, o, n, m in zip(ids, old, new, motion):
                samples.append(FlowSample(int(fid), o, n, m))
                self.trails.setdefault(int(fid), deque(maxlen=self.cfg.trail_length)).append(
                    (float(n[0]), float(n[1])))

            for fid in set(self.ids[~ok].tolist()):
                self.trails.pop(int(fid), None)
            self.pts, self.ids, self.still = new.astype(np.float32), ids, still

        # Retire background corners periodically (they carry no motion).
        if frame_idx % self.cfg.recalc_interval == 0 and len(self.pts):
            keep = self.still < 8
            for fid in self.ids[~keep]:
                self.trails.pop(int(fid), None)
            self.pts, self.ids, self.still = self.pts[keep], self.ids[keep], self.still[keep]

        # Top up with fresh corners inside the salient (moving) region.
        if frame_idx % 3 == 0 and len(self.pts) < self.cfg.max_corners:
            self._replenish(gray, saliency_mask)
        return samples

    def _replenish(self, gray: np.ndarray, saliency_mask: np.ndarray) -> None:
        mask = cv2.dilate(saliency_mask, np.ones((7, 7), np.uint8))
        for x, y in self.pts:
            cv2.circle(mask, (int(x), int(y)), max(self.cfg.min_distance, 4), 0, -1)
        params = dict(self.feature_params)
        params["maxCorners"] = int(self.cfg.max_corners - len(self.pts))
        if params["maxCorners"] <= 0 or not mask.any():
            return
        new = cv2.goodFeaturesToTrack(gray, mask=mask, **params)
        if new is None:
            return
        new = new.reshape(-1, 2).astype(np.float32)
        n = len(new)
        ids = np.arange(self.next_id, self.next_id + n, dtype=np.int64)
        self.next_id += n
        self.total_ids += n
        self.pts = np.vstack([self.pts, new])
        self.ids = np.concatenate([self.ids, ids])
        self.still = np.concatenate([self.still, np.zeros(n, np.int32)])
        for fid, p in zip(ids, new):
            self.trails[int(fid)] = deque([(float(p[0]), float(p[1]))], maxlen=self.cfg.trail_length)

    @property
    def active(self) -> int:
        return len(self.pts)
