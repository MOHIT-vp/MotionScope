"""Generate the landing-page demonstrations by running the real pipeline.

    python scripts/make_demos.py            # all demos
    python scripts/make_demos.py tracking   # one demo

Every demo video is produced by ``motiontrack.run_pipeline`` on the sample
footage in the project root; statistics in demos/manifest.json are copied
from the generated report.json, not typed by hand.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from motiontrack import GroundCalibration, PipelineConfig, run_pipeline  # noqa: E402
from motiontrack.video_io import ffmpeg_exe, probe  # noqa: E402

DEMO_DIR = ROOT / "demos"

DEMOS = [
    {
        "slug": "optical-flow",
        "title": "Sparse optical flow & feature trajectories",
        "source": "ex-1.mp4",
        "start_s": 0.0,
        "seconds": 15,
        "summary": "Shi–Tomasi corners sampled inside the motion-saliency mask are followed frame to frame "
                   "with pyramidal Lucas–Kanade. Arrows show per-frame displacement, coloured by direction; "
                   "fading lines are each corner's trajectory.",
        "shows": ["Lucas–Kanade", "Feature trails", "Direction colour wheel"],
        "layers": {"flow_vectors": True, "feature_trails": True, "saliency": False,
                   "objects": False, "hud": True, "calibration": False},
        "calibration": None,
    },
    {
        "slug": "vehicle-tracking",
        "title": "Vehicle tracking with calibrated speed estimates",
        "source": "ex-2.mp4",
        "start_s": 0.0,
        "seconds": 15,
        "summary": "Vehicles are detected (YOLOX), kept only if they move, and carried between detections "
                   "by their median Lucas–Kanade flow. Speeds come from a 4-point ground-plane homography "
                   "built from lane markings with assumed standard dimensions, so they are estimates.",
        "shows": ["Persistent IDs", "Trajectories", "km/h estimate (assumed calibration)"],
        "layers": {"flow_vectors": False, "feature_trails": True, "saliency": False,
                   "objects": True, "hud": True, "calibration": True},
        "calibration": "samples/ex-2.calibration.json",
    },
    {
        "slug": "motion-saliency",
        "title": "Motion saliency on a hand-held night shot",
        "source": "ex-3.mp4",
        "start_s": 0.0,
        "seconds": 12,
        "summary": "Accumulated motion saliency (S = 0.9·S + 0.1·B) rendered as a heat overlay. The phone "
                   "camera is hand-held, so global motion is estimated with RANSAC and removed before "
                   "frame differencing; remaining heat belongs to moving traffic.",
        "shows": ["Saliency heat map", "Camera-shake compensation", "Wet-road reflections"],
        "layers": {"flow_vectors": True, "feature_trails": False, "saliency": True,
                   "objects": True, "hud": True, "calibration": False},
        "calibration": None,
    },
]


def make(demo: dict) -> dict:
    src = ROOT / demo["source"]
    out = DEMO_DIR / demo["slug"]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    info = probe(src)
    start = int(round(demo["start_s"] * info.fps))
    n = int(round(demo["seconds"] * info.fps))
    cfg = PipelineConfig.from_dict({"start_frame": start, "max_frames": n, "layers": demo["layers"]})
    if demo["calibration"]:
        cfg.calibration = GroundCalibration.from_dict(json.loads((ROOT / demo["calibration"]).read_text()))

    def prog(done, total, fps):
        sys.stderr.write(f"\r  {demo['slug']}: {done}/{total} frames  {fps:4.1f} fps")

    report = run_pipeline(src, out, cfg, progress=prog)
    sys.stderr.write("\n")

    # Matching original clip for the before/after comparison.
    w, h = report["processing"]["width"], report["processing"]["height"]
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{demo['start_s']}", "-i", str(src), "-frames:v", str(report["summary"]["frames_processed"]),
                    "-vf", f"scale={w}:{h}", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / "original.mp4")], check=True)

    cap = cv2.VideoCapture(str(out / "annotated.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(report["summary"]["frames_processed"] * 0.55))
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(out / "poster.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    (out / "thumbnail.jpg").unlink(missing_ok=True)

    s = report["summary"]
    return {
        "slug": demo["slug"],
        "title": demo["title"],
        "summary": demo["summary"],
        "shows": demo["shows"],
        "source": demo["source"],
        "clip": {"start_s": demo["start_s"], "seconds": round(s["frames_processed"] / report["video"]["fps_used"], 2)},
        "orientation": "portrait" if h > w else "landscape",
        "files": {k: f"/demos/{demo['slug']}/{k}" for k in
                  ("annotated.mp4", "original.mp4", "poster.jpg", "report.json", "tracks.csv",
                   "heatmap.png", "trajectories.png")},
        "stats": {
            "frames": s["frames_processed"],
            "resolution": f"{report['video']['width']}×{report['video']['height']}",
            "fps": report["video"]["fps_used"],
            "unique_features": s["unique_features_tracked"],
            "objects": s["objects_tracked"],
            "mean_flow_px_s": s["mean_flow_px_s"],
            "dominant_direction": s["dominant_direction"],
            "camera_shift_px": s["mean_camera_shift_px"],
            "speed": s["speed"],
            "detector": report["processing"]["detector"],
        },
        "calibration_note": (cfg.calibration.label if cfg.calibration else None),
    }


def main(argv) -> None:
    wanted = set(argv)
    DEMO_DIR.mkdir(exist_ok=True)
    manifest_path = DEMO_DIR / "manifest.json"
    existing = {}
    if manifest_path.is_file():
        existing = {d["slug"]: d for d in json.loads(manifest_path.read_text()).get("demos", [])}
    for demo in DEMOS:
        if wanted and demo["slug"] not in wanted and not any(w in demo["slug"] for w in wanted):
            continue
        existing[demo["slug"]] = make(demo)
    ordered = [existing[d["slug"]] for d in DEMOS if d["slug"] in existing]
    manifest_path.write_text(json.dumps({"demos": ordered}, indent=1, ensure_ascii=False))
    print(f"wrote {manifest_path} ({len(ordered)} demos)")


if __name__ == "__main__":
    main(sys.argv[1:])
