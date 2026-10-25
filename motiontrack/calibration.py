"""Ground-plane calibration for converting image motion into metres.

The original script used a single constant ``scaling_factor = 0.1`` (m/px).
A constant scale ignores perspective: on a road filmed at an angle one pixel
near the horizon covers many more metres than one pixel near the camera, so
the same car "slows down" as it approaches.  Here the user marks four image
points that lie on the road surface and form a rectangle of known size in the
real world (e.g. one lane width by one lane-marking cycle).  A homography H
then maps any image point on the road plane to metric ground coordinates.

Point order (normalised 0..1 image coordinates):

    A ──width── B
    │           │
  length      length
    │           │
    D ──width── C

Speed is only reported when such a calibration is supplied.  Without it the
system reports measured pixel displacement and pixel velocity only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


class CalibrationError(ValueError):
    pass


@dataclass
class GroundCalibration:
    points: List[Tuple[float, float]]   # normalised (x, y), order A, B, C, D
    width_m: float                      # real distance A→B (and D→C)
    length_m: float                     # real distance B→C (and A→D)
    label: str = ""                     # free text, e.g. "lane width 3.65 m × dash cycle 9 m"

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ io
    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": [list(p) for p in self.points],
            "width_m": self.width_m,
            "length_m": self.length_m,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GroundCalibration":
        try:
            pts = [(float(p[0]), float(p[1])) for p in d["points"]]
            return cls(pts, float(d["width_m"]), float(d["length_m"]), str(d.get("label", "")))
        except (KeyError, TypeError, IndexError) as exc:
            raise CalibrationError(f"Malformed calibration: {exc}") from exc

    # ------------------------------------------------------------ checks
    def validate(self) -> None:
        if len(self.points) != 4:
            raise CalibrationError("Calibration needs exactly 4 points (A, B, C, D).")
        for x, y in self.points:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise CalibrationError("Calibration points must be normalised to 0..1.")
        if not (0.1 <= self.width_m <= 500 and 0.1 <= self.length_m <= 500):
            raise CalibrationError("Reference distances must be between 0.1 m and 500 m.")
        quad = np.array(self.points, dtype=np.float64)
        if not _is_convex(quad):
            raise CalibrationError("Calibration points must form a convex quadrilateral in order A→B→C→D.")
        if abs(_polygon_area(quad)) < 1e-4:
            raise CalibrationError("Calibration quadrilateral is too small.")

    # --------------------------------------------------------- geometry
    def homography(self, frame_w: int, frame_h: int) -> np.ndarray:
        src = np.array([(x * frame_w, y * frame_h) for x, y in self.points], dtype=np.float32)
        w, l = self.width_m, self.length_m
        dst = np.array([(0, 0), (w, 0), (w, l), (0, l)], dtype=np.float32)
        return cv2.getPerspectiveTransform(src, dst)

    def pixel_quad(self, frame_w: int, frame_h: int) -> np.ndarray:
        return np.array([(x * frame_w, y * frame_h) for x, y in self.points], dtype=np.int32)


def to_ground(H: np.ndarray, pts: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Project image points (px) onto the calibrated ground plane (metres)."""
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, H).reshape(-1, 2)


def fit_speed(times: Sequence[float], ground_pts: np.ndarray) -> Optional[Tuple[float, float]]:
    """Least-squares velocity over a window of ground positions.

    Returns (speed m/s, residual std in metres) or None if ill-posed.
    A linear fit over several frames is far less noisy than the original
    frame-to-frame difference, which amplified 1-px jitter into km/h spikes.
    """
    t = np.asarray(times, dtype=np.float64)
    if len(t) < 3 or (t[-1] - t[0]) <= 0:
        return None
    A = np.vstack([t - t[0], np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, ground_pts.astype(np.float64), rcond=None)
    vx, vy = coef[0]
    resid = ground_pts - A @ coef
    return float(np.hypot(vx, vy)), float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))


def _polygon_area(q: np.ndarray) -> float:
    x, y = q[:, 0], q[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _is_convex(q: np.ndarray) -> bool:
    signs = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < 1e-12:
            return False
        signs.append(cross > 0)
    return all(signs) or not any(signs)
