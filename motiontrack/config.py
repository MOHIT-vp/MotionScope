"""Pipeline configuration.

Defaults for feature detection, Lucas–Kanade and saliency are taken verbatim
from the original implementation so results stay comparable:

    feature_params = dict(maxCorners=200, qualityLevel=0.2, minDistance=5, blockSize=5)
    lk_params      = dict(winSize=(15, 15), maxLevel=2, criteria=(EPS|COUNT, 10, 0.03))
    saliency       : |I_t - I_{t-1}| > 30,  S_t = 0.9 * S_{t-1} + 0.1 * B_t
    recalc_interval = 30
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from .calibration import GroundCalibration


@dataclass
class Layers:
    """Which overlays are drawn into the annotated output video."""

    flow_vectors: bool = True      # per-frame LK displacement arrows
    feature_trails: bool = True    # fading trajectories of tracked corners
    saliency: bool = False         # accumulated motion-saliency heat overlay
    objects: bool = True           # bounding boxes, IDs, object trajectories
    hud: bool = True               # frame counter / statistics panel
    calibration: bool = True       # ground-plane quad (only if calibrated)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Layers":
        if not d:
            return cls()
        known = {k: bool(v) for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class PipelineConfig:
    # --- preprocessing -----------------------------------------------------
    process_size: int = 960           # long side in px (original resized to width 512)
    max_frames: Optional[int] = None  # stop early (used for demos / previews)
    start_frame: int = 0
    fps_override: Optional[float] = None

    # --- Shi–Tomasi corners (original values) --------------------------------
    max_corners: int = 200
    quality_level: float = 0.2
    min_distance: int = 5
    block_size: int = 5
    recalc_interval: int = 30

    # --- Lucas–Kanade (original values) --------------------------------------
    lk_win_size: int = 15
    lk_max_level: int = 2
    lk_iters: int = 10
    lk_eps: float = 0.03
    fb_threshold: float = 1.0         # forward–backward error limit in px (new)

    # --- motion saliency (original values) -----------------------------------
    diff_threshold: int = 30
    saliency_alpha: float = 0.9

    # --- camera motion compensation (new) -------------------------------------
    compensate_camera: bool = True

    # --- object detection / tracking (new) ------------------------------------
    detector: str = "auto"            # "auto" | "yolox" | "motion"
    detect_every: int = 2             # run YOLOX every N frames; LK carries boxes between
    det_confidence: float = 0.4
    min_motion_fraction: float = 0.04 # share of a box that must be moving (static cars ignored)
    min_object_area: float = 0.0012   # fraction of frame area
    max_track_age: int = 12           # frames a track may coast without a detection
    min_track_hits: int = 4           # detections before a track is confirmed
    iou_threshold: float = 0.2
    trail_length: int = 24            # frames of feature trail drawn

    # --- speed estimation -------------------------------------------------------
    calibration: Optional[GroundCalibration] = None
    speed_window_s: float = 0.6       # sliding window for speed regression

    layers: Layers = field(default_factory=Layers)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["calibration"] = self.calibration.to_dict() if self.calibration else None
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "PipelineConfig":
        d = dict(d or {})
        cal = d.pop("calibration", None)
        layers = d.pop("layers", None)
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg.layers = Layers.from_dict(layers)
        cfg.calibration = GroundCalibration.from_dict(cal) if cal else None
        return cfg
