"""Command-line entry point.

    python -m motiontrack.cli input.mp4 -o out_dir [--width 960] [--frames 300]
        [--calibration calib.json] [--saliency] [--fps 25]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .calibration import GroundCalibration
from .config import PipelineConfig
from .pipeline import run_pipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Lucas–Kanade motion & vehicle tracking")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("output"))
    ap.add_argument("--size", type=int, default=960, help="processing size (long side, px)")
    ap.add_argument("--detector", choices=["auto", "yolox", "motion"], default="auto")
    ap.add_argument("--start", type=int, default=0, help="first frame index")
    ap.add_argument("--frames", type=int, default=None, help="max frames to process")
    ap.add_argument("--fps", type=float, default=None, help="override frame rate")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="JSON with points (normalised A,B,C,D), width_m, length_m")
    ap.add_argument("--saliency", action="store_true", help="draw saliency heat overlay")
    ap.add_argument("--no-flow", action="store_true", help="hide flow arrows")
    ap.add_argument("--no-trails", action="store_true", help="hide feature trails")
    ap.add_argument("--no-objects", action="store_true", help="hide object boxes")
    args = ap.parse_args(argv)

    cfg = PipelineConfig(process_size=args.size, detector=args.detector, start_frame=args.start,
                         max_frames=args.frames, fps_override=args.fps)
    cfg.layers.saliency = args.saliency
    cfg.layers.flow_vectors = not args.no_flow
    cfg.layers.feature_trails = not args.no_trails
    cfg.layers.objects = not args.no_objects
    if args.calibration:
        cfg.calibration = GroundCalibration.from_dict(json.loads(args.calibration.read_text()))

    def prog(done, total, fps):
        pct = 100 * done / total if total else 0
        sys.stderr.write(f"\r{done}/{total} frames  {pct:5.1f}%  {fps:5.1f} fps")

    report = run_pipeline(args.input, args.out, cfg, progress=prog)
    sys.stderr.write("\n")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
