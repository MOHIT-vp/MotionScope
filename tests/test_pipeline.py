"""End-to-end pipeline checks against synthetic ground truth."""

import csv
import json

import cv2
import pytest

from motiontrack import GroundCalibration, PipelineConfig, run_pipeline
from motiontrack.pipeline import JobControl, PipelineCancelled, direction_of
from motiontrack.video_io import VideoError, probe

from conftest import FPS, H, N_FRAMES, VX, W

TRUE_PX_S = VX * FPS                     # 100 px/s
SQUARE = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]


def _main_track(report):
    assert report["tracks"], "no track was confirmed"
    return max(report["tracks"], key=lambda t: t["frames_observed"])


def test_outputs_and_measurements(synthetic_video, tmp_path):
    cfg = PipelineConfig(process_size=640, detector="motion")
    report = run_pipeline(synthetic_video, tmp_path, cfg)

    for name in ("annotated.mp4", "report.json", "tracks.csv", "heatmap.png", "trajectories.png"):
        assert (tmp_path / name).stat().st_size > 0, name

    out = probe(tmp_path / "annotated.mp4")
    assert out.codec == "h264" and (out.width, out.height) == (W, H)
    assert report["summary"]["frames_processed"] == N_FRAMES

    tr = _main_track(report)
    assert tr["direction"]["label"] == "right"
    assert abs(tr["direction"]["angle_deg"] - 0) < 8 or abs(tr["direction"]["angle_deg"] - 360) < 8
    # Lucas–Kanade velocity is measured directly: expect it within 5 % of truth.
    assert tr["lk_velocity_px_s"] == pytest.approx(TRUE_PX_S, rel=0.05)
    assert report["summary"]["speed"]["calibrated"] is False
    assert tr["speed"] is None

    with open(tmp_path / "tracks.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(report["tracks"])
    assert json.loads((tmp_path / "report.json").read_text())["schema"] == "motiontrack.report/1"


def test_calibrated_speed_matches_ground_truth(synthetic_video, tmp_path):
    # Top-down synthetic camera: 320 px = 16 m → 20 px/m.  100 px/s = 5 m/s = 18 km/h.
    cal = GroundCalibration(SQUARE, width_m=16, length_m=12)
    cfg = PipelineConfig(process_size=640, detector="motion", calibration=cal)
    report = run_pipeline(synthetic_video, tmp_path, cfg)
    tr = _main_track(report)
    assert tr["speed"]["kind"] == "estimate"
    assert tr["speed"]["median_kmh"] == pytest.approx(18.0, rel=0.06)
    assert report["summary"]["speed"]["calibrated"] is True


def test_fps_override_changes_speed(synthetic_video, tmp_path):
    cal = GroundCalibration(SQUARE, width_m=16, length_m=12)
    cfg = PipelineConfig(process_size=640, detector="motion", calibration=cal, fps_override=50)
    tr = _main_track(run_pipeline(synthetic_video, tmp_path, cfg))
    assert tr["speed"]["median_kmh"] == pytest.approx(36.0, rel=0.06)


def test_camera_shake_is_measured_and_compensated(shaky_video, tmp_path):
    cfg = PipelineConfig(process_size=640, detector="motion")
    report = run_pipeline(shaky_video, tmp_path, cfg)
    assert report["summary"]["mean_camera_shift_px"] > 0.5
    tr = _main_track(report)
    assert tr["direction"]["label"] == "right"
    assert tr["lk_velocity_px_s"] == pytest.approx(TRUE_PX_S, rel=0.12)


def test_max_frames_and_layers(synthetic_video, tmp_path):
    cfg = PipelineConfig.from_dict({"process_size": 320, "max_frames": 20, "detector": "motion",
                                    "layers": {"saliency": True, "hud": False}})
    report = run_pipeline(synthetic_video, tmp_path, cfg)
    assert report["summary"]["frames_processed"] == 20
    assert report["processing"]["width"] == 320
    cap = cv2.VideoCapture(str(tmp_path / "annotated.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20


def test_cancel_raises_and_cleans_encoder(synthetic_video, tmp_path):
    ctrl = JobControl()
    ctrl.cancel()
    with pytest.raises(PipelineCancelled):
        run_pipeline(synthetic_video, tmp_path, PipelineConfig(detector="motion"), control=ctrl)
    assert not (tmp_path / "annotated.mp4").exists()


def test_unreadable_file(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 2048)
    with pytest.raises(VideoError):
        run_pipeline(bad, tmp_path / "out", PipelineConfig())


@pytest.mark.parametrize("dx,dy,label", [(1, 0, "right"), (0, -1, "up"), (-1, 0, "left"), (0, 1, "down"), (1, 1, "down-right")])
def test_direction_labels(dx, dy, label):
    assert direction_of(dx, dy)["label"] == label
