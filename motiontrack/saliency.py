"""Motion saliency and camera-motion compensation.

Motion saliency follows the original formulation exactly:

    B_t = 1[ |I_t - I_{t-1}| > τ ]            (τ = 30)
    S_t = α · S_{t-1} + (1 - α) · B_t         (α = 0.9)

New: before differencing, the previous frame can be warped onto the current
one with a similarity transform estimated from background features (RANSAC).
This removes small camera shake, which the original report lists as an open
challenge ("distinguishing vehicle motion from camera jitter").
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class MotionSaliency:
    def __init__(self, shape: Tuple[int, int], threshold: int = 30, alpha: float = 0.9):
        self.threshold = threshold
        self.alpha = alpha
        self.accumulated = np.zeros(shape, dtype=np.float32)
        self.binary = np.zeros(shape, dtype=np.uint8)

    def update(self, prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(prev_gray, gray)
        _, self.binary = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        self.accumulated = cv2.addWeighted(
            self.accumulated, self.alpha, self.binary.astype(np.float32), 1.0 - self.alpha, 0
        )
        return self.binary

    def coverage(self) -> float:
        """Fraction of pixels flagged as moving in the current frame."""
        return float(np.count_nonzero(self.binary)) / self.binary.size


class CameraMotion:
    """Estimate global (camera) motion between consecutive frames."""

    def __init__(self, lk_params: dict, grid_corners: int = 300):
        self.lk_params = lk_params
        self.grid_corners = grid_corners

    def estimate(self, prev_gray: np.ndarray, gray: np.ndarray,
                 exclude_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """Return (2×3 transform prev→cur, translation magnitude in px).

        Points inside ``exclude_mask`` (current foreground) are ignored so
        that moving vehicles do not dominate the estimate.
        """
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        mask = None
        if exclude_mask is not None:
            mask = cv2.bitwise_not(cv2.dilate(exclude_mask, np.ones((15, 15), np.uint8)))
        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=self.grid_corners,
                                      qualityLevel=0.01, minDistance=12, mask=mask)
        if pts is None or len(pts) < 12:
            return identity, 0.0
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **self.lk_params)
        good = st.reshape(-1) == 1
        if good.sum() < 12:
            return identity, 0.0
        M, inliers = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                                 ransacReprojThreshold=1.5)
        if M is None or inliers is None or inliers.mean() < 0.5:
            return identity, 0.0
        shift = float(np.hypot(M[0, 2], M[1, 2]))
        if shift < 0.3:  # sub-pixel: treat camera as static, avoid resampling blur
            return identity, shift
        return M.astype(np.float32), shift


def warp(gray: np.ndarray, M: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
