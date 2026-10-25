"""Runtime settings (override with environment variables)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MT_DATA_DIR", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
JOB_DIR = DATA_DIR / "jobs"
FRONTEND_DIR = ROOT / "frontend"
DEMO_DIR = ROOT / "demos"

MAX_UPLOAD_MB = int(os.environ.get("MT_MAX_UPLOAD_MB", "1024"))
MAX_DURATION_S = float(os.environ.get("MT_MAX_DURATION_S", "900"))
MIN_DIMENSION = 64
RETENTION_HOURS = float(os.environ.get("MT_RETENTION_HOURS", "72"))

# Processing caps. A hosted demo sets these low so one visitor cannot occupy
# the single worker for minutes; local use keeps the full range.
MAX_PROCESS_SIZE = int(os.environ.get("MT_MAX_PROCESS_SIZE", "1920"))
MAX_FRAMES = int(os.environ.get("MT_MAX_FRAMES", "0")) or None   # 0 = no cap
# Shown on the page when set, e.g. "Demo: uploads are processed and deleted after 1 hour."
DEMO_NOTE = os.environ.get("MT_DEMO_NOTE", "").strip()

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv"}
# Containers + codec combinations browsers play natively; anything else gets an H.264 preview.
BROWSER_PLAYABLE = {(".mp4", "h264"), (".m4v", "h264"), (".mov", "h264"), (".webm", "vp8"), (".webm", "vp9")}

RESULT_FILES = {
    "annotated.mp4": "video/mp4",
    "report.json": "application/json",
    "tracks.csv": "text/csv",
    "heatmap.png": "image/png",
    "trajectories.png": "image/png",
    "thumbnail.jpg": "image/jpeg",
}

for d in (UPLOAD_DIR, JOB_DIR):
    d.mkdir(parents=True, exist_ok=True)
