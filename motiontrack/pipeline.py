"""End-to-end video analysis: read → saliency → LK → detect → track → measure → render.

Outputs written to ``out_dir``:

    annotated.mp4     H.264 video with the selected overlays
    report.json       full analysis (metadata, summary, per-track, time series)
    tracks.csv        one row per confirmed moving object
    heatmap.png       cumulative motion-saliency energy over the background
    trajectories.png  every confirmed object path over the background
    thumbnail.jpg     first processed frame
"""

from __future__ import annotations

import csv
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from . import render
from .calibration import fit_speed, to_ground
from .config import PipelineConfig
from .detection import ForegroundDetector, YoloxDetector, motion_fraction, yolox_available
from .flow import FeatureTracker
from .saliency import CameraMotion, MotionSaliency, warp
from .tracker import MultiObjectTracker, Track
from .video_io import H264Writer, VideoError, iter_frames, probe, resize_long_side

ProgressFn = Callable[[int, int, float], None]


class PipelineCancelled(Exception):
    pass


class JobControl:
    """Cooperative pause / resume / cancel used by the job manager."""

    def __init__(self) -> None:
        self._cancel = threading.Event()
        self._run = threading.Event()
        self._run.set()
        self.paused_s = 0.0          # wall time spent paused (excluded from timings)

    def cancel(self) -> None:
        self._cancel.set()
        self._run.set()

    def pause(self) -> None:
        self._run.clear()

    def resume(self) -> None:
        self._run.set()

    @property
    def paused(self) -> bool:
        return not self._run.is_set()

    def checkpoint(self) -> None:
        if not self._run.is_set():
            t0 = time.time()
            while not self._run.wait(timeout=0.25):
                if self._cancel.is_set():
                    break
            self.paused_s += time.time() - t0
        if self._cancel.is_set():
            raise PipelineCancelled()


DIRECTIONS = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
ARROWS = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]


def direction_of(dx: float, dy: float) -> Dict[str, object]:
    """Angle in degrees, counter-clockwise from image-right (image y axis flipped)."""
    ang = math.degrees(math.atan2(-dy, dx)) % 360
    idx = int(((ang + 22.5) % 360) // 45)
    return {"angle_deg": round(ang, 1) % 360, "label": DIRECTIONS[idx], "arrow": ARROWS[idx]}


def run_pipeline(input_path: Path, out_dir: Path, cfg: Optional[PipelineConfig] = None,
                 progress: Optional[ProgressFn] = None, control: Optional[JobControl] = None) -> dict:
    cfg = cfg or PipelineConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()

    info = probe(Path(input_path))
    fps = float(cfg.fps_override) if cfg.fps_override else info.fps
    fps_source = "user override" if cfg.fps_override else "container metadata"
    if not (1.0 <= fps <= 240.0):
        raise VideoError("Frame rate must be between 1 and 240 fps.")
    dt = 1.0 / fps

    total = max(0, info.frame_count - cfg.start_frame)
    if cfg.max_frames:
        total = min(total, cfg.max_frames)

    frames = iter_frames(Path(input_path), cfg.start_frame)
    first = next(frames, None)
    if first is None:
        raise VideoError("The video contains no readable frames.")
    first = resize_long_side(first, cfg.process_size)
    H_, W_ = first.shape[:2]
    sx, sy = info.width / W_, info.height / H_      # process px → source px

    lk_params = dict(winSize=(cfg.lk_win_size, cfg.lk_win_size), maxLevel=cfg.lk_max_level,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.lk_iters, cfg.lk_eps))
    saliency = MotionSaliency((H_, W_), cfg.diff_threshold, cfg.saliency_alpha)
    camera = CameraMotion(lk_params)
    features = FeatureTracker(cfg, first.shape)
    detector = ForegroundDetector(first.shape, cfg.min_object_area)
    tracker = MultiObjectTracker(first.shape, cfg.iou_threshold, cfg.max_track_age, cfg.min_track_hits)
    detector.warm_up(first)
    if cfg.detector not in ("auto", "yolox", "motion"):
        raise ValueError("detector must be auto, yolox or motion")
    if cfg.detector == "yolox" and not yolox_available():
        raise VideoError("YOLOX model not installed; run scripts/fetch_model.py or choose the motion detector.")
    yolo = YoloxDetector(cfg.det_confidence) if cfg.detector != "motion" and yolox_available() else None
    detector_name = "yolox" if yolo else "motion"

    H = cfg.calibration.homography(W_, H_) if cfg.calibration else None
    quad = cfg.calibration.pixel_quad(W_, H_) if cfg.calibration else None

    writer = H264Writer(out_dir / "annotated.mp4", W_, H_, fps)
    cv2.imwrite(str(out_dir / "thumbnail.jpg"), first, [cv2.IMWRITE_JPEG_QUALITY, 85])

    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    energy = np.zeros((H_, W_), np.float32)
    display_ids: Dict[int, int] = {}
    live_speed: Dict[int, List[float]] = {}     # tid → windowed speed samples (m/s)
    live_resid: Dict[int, List[float]] = {}
    ts: Dict[str, list] = {k: [] for k in ("t", "active_features", "active_objects",
                                           "mean_flow_px_s", "saliency_coverage", "camera_shift_px")}
    dir_hist = np.zeros(16, np.float64)
    win = max(3, int(round(cfg.speed_window_s * fps)))
    processed = 1
    t_last_progress = 0.0

    try:
        writer.write(first)   # frame 0 has no predecessor, so it carries no motion overlay
        for idx, frame in enumerate(frames, start=1):
            if cfg.max_frames and processed >= cfg.max_frames:
                break
            if control:
                control.checkpoint()
            frame = resize_long_side(frame, cfg.process_size)
            if frame.shape[:2] != (H_, W_):
                frame = cv2.resize(frame, (W_, H_))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t = idx * dt

            # --- camera motion + saliency -----------------------------------
            M, shift = (camera.estimate(prev_gray, gray, detector.mask)
                        if cfg.compensate_camera else (None, 0.0))
            cam_moving = shift >= 0.3
            ref = warp(prev_gray, M) if (M is not None and cam_moving) else prev_gray
            binary = saliency.update(ref, gray)
            energy += binary.astype(np.float32) / 255.0

            # --- Lucas–Kanade ------------------------------------------------
            flow = features.step(prev_gray, gray, binary, idx, M if cam_moving else None)

            # --- objects -----------------------------------------------------
            blobs = detector.detect(frame, binary, cam_moving)
            if yolo is None:
                dets = [(b, "moving object", None) for b in blobs]
            elif idx % max(1, cfg.detect_every) == 0:
                dets = []
                for box, lab, conf in yolo.detect(frame):
                    # keep only vehicles that actually move (parked cars, signs → ignored)
                    if motion_fraction(detector.mask, box) >= cfg.min_motion_fraction or \
                            _flow_inside(flow, box) >= 0.5:
                        dets.append((box, lab, conf))
            else:
                dets = None
            active = tracker.step(dets, flow, idx, t)

            # --- measurements -----------------------------------------------
            mags = [float(np.hypot(*s.motion)) for s in flow]
            moving = [s for s, m in zip(flow, mags) if m >= 0.4]
            for s in moving:
                ang = math.atan2(-float(s.motion[1]), float(s.motion[0])) % (2 * math.pi)
                dir_hist[int(ang / (2 * math.pi) * 16) % 16] += float(np.hypot(*s.motion))
            mean_flow = float(np.mean([np.hypot(*s.motion) for s in moving])) if moving else 0.0

            objects_draw = []
            for tr in active:
                if tr.tid not in display_ids:
                    display_ids[tr.tid] = len(display_ids) + 1
                did = display_ids[tr.tid]
                label = f"#{did}" + (f" {tr.label}" if yolo else "")
                if H is not None:
                    hist = [h for h in tr.history if h[11]][-win:]
                    if any(_at_border(h, W_, H_) for h in hist):
                        hist = []      # box clipped by the frame: ground point unreliable
                    g = to_ground(H, [(h[4], h[5]) for h in hist]) if len(hist) >= max(3, win // 2) else None
                    if g is not None and _in_zone(g, cfg.calibration):
                        fit = fit_speed([h[1] for h in hist], g)
                        if fit:
                            live_speed.setdefault(tr.tid, []).append(fit[0])
                            live_resid.setdefault(tr.tid, []).append(fit[1])
                            label += f" ~{fit[0] * 3.6:.0f} km/h"
                else:
                    v = math.hypot(*tr.velocity) * fps * sx
                    label += f" {v:.0f} px/s"
                trail = [(h[2], h[3]) for h in tr.history[-50:]]
                objects_draw.append({"box": tr.box, "label": label, "trail": trail})

            ts["t"].append(round(t, 3))
            ts["active_features"].append(features.active)
            ts["active_objects"].append(len(active))
            ts["mean_flow_px_s"].append(round(mean_flow * fps * sx, 2))
            ts["saliency_coverage"].append(round(saliency.coverage() * 100, 3))
            ts["camera_shift_px"].append(round(shift * sx, 3))

            # --- render ------------------------------------------------------
            out = frame
            L = cfg.layers
            if L.saliency:
                out = render.draw_saliency(out, saliency.accumulated)
            else:
                out = out.copy()
            if L.calibration and quad is not None:
                render.draw_calibration(out, quad)
            if L.feature_trails:
                render.draw_trails(out, features.trails)
            if L.flow_vectors:
                render.draw_flow(out, flow)
            if L.objects:
                render.draw_objects(out, objects_draw)
            if L.hud:
                render.draw_hud(out, [
                    "LUCAS-KANADE / MOTION SALIENCY",
                    f"t {t:6.2f}s  frame {cfg.start_frame + idx}",
                    f"features {features.active:4d}  objects {len(active):2d}",
                    f"flow {mean_flow * fps * sx:6.1f} px/s  cam {shift * sx:4.1f}px",
                    "speed: calibrated" if H is not None else "speed: not calibrated",
                ])
            writer.write(out)
            processed += 1
            prev_gray = gray

            if progress and (time.time() - t_last_progress > 0.2):
                t_last_progress = time.time()
                el = time.time() - started - (control.paused_s if control else 0.0)
                progress(processed, total, processed / el if el > 0 else 0.0)
        writer.close()
    except BaseException:
        writer.abort()
        raise

    elapsed = time.time() - started - (control.paused_s if control else 0.0)
    finished = tracker.close()
    background = detector.bg.getBackgroundImage()
    if background is None or background.shape[:2] != (H_, W_):
        background = first

    tracks = [_summarise_track(tr, display_ids, live_speed, live_resid, fps, sx, sy, W_, H_, H, yolo is not None)
              for tr in finished if tr.tid in display_ids]
    tracks = [t for t in tracks if t is not None]
    tracks.sort(key=lambda d: d["id"])

    cv2.imwrite(str(out_dir / "heatmap.png"), render.heatmap_image(background, energy))
    cv2.imwrite(str(out_dir / "trajectories.png"),
                render.trajectories_image(background, tracks, sx, sy))

    report = _build_report(info, cfg, fps, fps_source, processed, elapsed, started_iso,
                           features.total_ids, ts, dir_hist, tracks, W_, H_, detector_name)
    for t in tracks:
        t.pop("_trail_px", None)
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=1)
    _write_csv(out_dir / "tracks.csv", tracks, cfg.calibration is not None)
    if progress:
        progress(processed, processed, processed / elapsed if elapsed else 0.0)
    return report


# ---------------------------------------------------------------------------
def _flow_inside(flow, box) -> float:
    x, y, w, h = box
    m = [float(np.hypot(*s.motion)) for s in flow
         if x <= s.old[0] <= x + w and y <= s.old[1] <= y + h]
    return float(np.median(m)) if len(m) >= 3 else 0.0


def _at_border(h, W_, H_, m: float = 3.0) -> bool:
    _, _, cx, cy, _, gy, w, hh = h[:8]
    return gy >= H_ - m or cy - hh / 2 <= m or cx - w / 2 <= m or cx + w / 2 >= W_ - m


def _in_zone(g: np.ndarray, cal) -> bool:
    """Homography error grows quickly away from the reference rectangle
    (especially in depth), so speed is only estimated while the object's
    ground point stays within one rectangle length before/after it, and
    within one rectangle length to either side."""
    W, L = cal.width_m, cal.length_m
    x, y = g[:, 0], g[:, 1]
    return bool(np.all((x > -L) & (x < W + L) & (y > -L) & (y < 2 * L)))


def _summarise_track(tr: Track, display_ids, live_speed, live_resid, fps, sx, sy, W_, H_, H,
                     classified: bool) -> Optional[dict]:
    hist = [h for h in tr.history if h[11]]
    if len(hist) < 3:
        return None
    t0, t1 = hist[0][1], hist[-1][1]
    c0 = np.array([hist[0][2] * sx, hist[0][3] * sy])
    c1 = np.array([hist[-1][2] * sx, hist[-1][3] * sy])
    pts = np.array([(h[2] * sx, h[3] * sy) for h in hist])
    path = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    net = c1 - c0
    flow_v = [math.hypot(h[8] * sx, h[9] * sy) * fps for h in hist if h[10] >= 3]
    duration = t1 - t0
    # Drop objects that never really moved (e.g. a parked car or a sign that
    # briefly passed the motion gate): < 1 % of the frame diagonal overall.
    if np.linalg.norm(net) < 0.01 * math.hypot(W_ * sx, H_ * sy) and path < 0.03 * math.hypot(W_ * sx, H_ * sy):
        return None
    d = {
        "id": display_ids[tr.tid],
        "class": tr.label if classified else "moving object",
        "detector_confidence": round(float(np.mean(tr.confs)), 2) if tr.confs else None,
        "first_frame": int(hist[0][0]),
        "last_frame": int(hist[-1][0]),
        "start_s": round(t0, 3),
        "end_s": round(t1, 3),
        "duration_s": round(duration, 3),
        "frames_observed": len(hist),
        "net_displacement_px": round(float(np.linalg.norm(net)), 1),
        "path_length_px": round(path, 1),
        "displacement_vector_px": [round(float(net[0]), 1), round(float(net[1]), 1)],
        "direction": direction_of(float(net[0]), float(net[1])),
        "mean_box_px": [round(float(np.mean([h[6] for h in hist])) * sx, 1),
                        round(float(np.mean([h[7] for h in hist])) * sy, 1)],
        "lk_velocity_px_s": round(float(np.median(flow_v)), 1) if flow_v else None,
        "track_velocity_px_s": round(float(np.linalg.norm(net)) / duration, 1) if duration > 0 else None,
        "lk_support_ratio": round(len(flow_v) / len(hist), 2),
        "trajectory_norm": [[round(x / (W_ * sx), 4), round(y / (H_ * sy), 4)]
                            for x, y in pts[:: max(1, len(pts) // 120)]],
        "_trail_px": [(float(x), float(y)) for x, y in pts],
        "speed": None,
    }
    if H is not None:
        samples = live_speed.get(tr.tid, [])
        if len(samples) >= 3 and duration >= 0.5:
            arr = np.array(samples) * 3.6
            p25, p50, p75 = np.percentile(arr, [25, 50, 75])
            g = to_ground(H, [(h[4], h[5]) for h in hist])
            ground_dist = float(np.linalg.norm(g[-1] - g[0]))
            d["speed"] = {
                "kind": "estimate",
                "median_kmh": round(float(p50), 1),
                "iqr_kmh": [round(float(p25), 1), round(float(p75), 1)],
                "average_kmh": round(ground_dist / duration * 3.6, 1),
                "ground_distance_m": round(ground_dist, 2),
                "fit_residual_m": round(float(np.median(live_resid.get(tr.tid, [0]))), 3),
                "samples": len(samples),
            }
        else:
            d["speed"] = {"kind": "unavailable",
                          "reason": "not enough observations inside the calibrated zone"}
    return d


def _build_report(info, cfg, fps, fps_source, processed, elapsed, started_iso, total_features,
                  ts, dir_hist, tracks, W_, H_, detector_name) -> dict:
    calibrated = cfg.calibration is not None
    speeds = [t["speed"]["median_kmh"] for t in tracks
              if t.get("speed") and t["speed"].get("kind") == "estimate"]
    moving_ratio = [v for v in ts["mean_flow_px_s"] if v > 0]
    peak = int(np.argmax(dir_hist)) if dir_hist.sum() > 0 else None

    # Down-sample long series for the dashboard (≤ 600 points).
    step = max(1, len(ts["t"]) // 600)
    series = {k: v[::step] for k, v in ts.items()}

    summary = {
        "frames_processed": processed,
        "processing_time_s": round(elapsed, 2),
        "throughput_fps": round(processed / elapsed, 1) if elapsed else None,
        "unique_features_tracked": int(total_features),
        "mean_active_features": round(float(np.mean(ts["active_features"])), 1) if ts["active_features"] else 0,
        "objects_tracked": len(tracks),
        "peak_concurrent_objects": int(max(ts["active_objects"])) if ts["active_objects"] else 0,
        "mean_flow_px_s": round(float(np.mean(moving_ratio)), 1) if moving_ratio else 0.0,
        "dominant_direction": (direction_of(math.cos((peak + 0.5) * math.pi / 8),
                                            -math.sin((peak + 0.5) * math.pi / 8)) if peak is not None else None),
        "mean_camera_shift_px": round(float(np.mean(ts["camera_shift_px"])), 3) if ts["camera_shift_px"] else 0.0,
        "speed": {
            "calibrated": calibrated,
            "objects_with_estimate": len(speeds),
            "median_kmh": round(float(np.median(speeds)), 1) if speeds else None,
            "range_kmh": [round(min(speeds), 1), round(max(speeds), 1)] if speeds else None,
        },
    }
    notes = [
        "Pixel quantities (displacement, px/s, flow) are measured directly from the video.",
        ("Vehicle boxes and classes come from YOLOX-S (COCO, OpenCV model zoo), gated by motion; "
         "motion and trajectories come from Lucas–Kanade." if detector_name == "yolox" else
         "Objects are class-agnostic motion blobs (background subtraction + saliency); "
         "vehicles driving close together may merge into one object."),
        "Pixel values are expressed in source-video pixels.",
    ]
    if calibrated:
        notes.append("Speeds are estimates from a 4-point ground-plane homography supplied by the user; "
                     "accuracy depends on how well the reference distances and points match reality, "
                     "on the frame rate, and on the bottom of each box touching the road.")
        if cfg.calibration.label:
            notes.append(f"Calibration reference: {cfg.calibration.label}")
    else:
        notes.append("No calibration was supplied, so no real-world speed is reported. "
                     "Pixel speed is not convertible to km/h without scene geometry.")
    return {
        "schema": "motiontrack.report/1",
        "generated_at": started_iso,
        "video": {**info.to_dict(), "fps_used": round(fps, 3), "fps_source": fps_source},
        "processing": {"width": W_, "height": H_, "detector": detector_name, "config": cfg.to_dict()},
        "summary": summary,
        "direction_histogram": {
            "bins": 16,
            "weights": [round(float(v), 2) for v in (dir_hist / dir_hist.sum() if dir_hist.sum() else dir_hist)],
        },
        "timeseries": series,
        "tracks": tracks,
        "notes": notes,
    }


def _write_csv(path: Path, tracks: List[dict], calibrated: bool) -> None:
    cols = ["id", "start_s", "end_s", "duration_s", "frames_observed", "net_displacement_px",
            "path_length_px", "direction_deg", "direction", "lk_velocity_px_s", "track_velocity_px_s"]
    if calibrated:
        cols += ["speed_median_kmh_estimate", "speed_iqr_low_kmh", "speed_iqr_high_kmh", "ground_distance_m"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in tracks:
            row = [t["id"], t["start_s"], t["end_s"], t["duration_s"], t["frames_observed"],
                   t["net_displacement_px"], t["path_length_px"], t["direction"]["angle_deg"],
                   t["direction"]["label"], t["lk_velocity_px_s"], t["track_velocity_px_s"]]
            if calibrated:
                s = t.get("speed") or {}
                if s.get("kind") == "estimate":
                    row += [s["median_kmh"], s["iqr_kmh"][0], s["iqr_kmh"][1], s["ground_distance_m"]]
                else:
                    row += ["", "", "", ""]
            w.writerow(row)
