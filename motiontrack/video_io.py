"""Video probing, frame reading and browser-compatible H.264 writing.

OpenCV's ``mp4v`` writer (used by the original script) produces MPEG-4 Part 2
files that browsers cannot play.  Output is therefore piped into the ffmpeg
binary shipped with ``imageio-ffmpeg`` and encoded as H.264 / yuv420p with the
``moov`` atom at the front so it streams in a <video> element.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


class VideoError(RuntimeError):
    pass


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str
    has_audio: bool

    def to_dict(self):
        return asdict(self)


def probe(path: Path) -> VideoInfo:
    """Read basic properties; raises VideoError if the file is not decodable."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("The file could not be opened as a video.")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError("No decodable video frames were found.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        h, w = frame.shape[:2]
    finally:
        cap.release()

    codec, has_audio, duration = _ffmpeg_header(path)
    if not (1.0 <= fps <= 240.0):
        # Variable / missing frame-rate metadata: fall back to 30 and let the
        # user override it explicitly (speed depends on it).
        fps = 30.0
    if count <= 0 and duration:
        count = int(round(duration * fps))
    if duration is None:
        duration = count / fps if fps else 0.0
    return VideoInfo(w, h, round(fps, 3), count, round(duration, 3), codec, has_audio)


def _ffmpeg_header(path: Path) -> Tuple[str, bool, Optional[float]]:
    try:
        out = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
            capture_output=True, text=True, timeout=30,
        ).stderr
    except Exception:  # pragma: no cover - ffmpeg missing is reported elsewhere
        return "unknown", False, None
    m = re.search(r"Video: (\w+)", out)
    d = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", out)
    dur = None
    if d:
        dur = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3))
    return (m.group(1) if m else "unknown"), ("Audio:" in out), dur


def iter_frames(path: Path, start: int = 0) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("The file could not be opened as a video.")
    try:
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def resize_long_side(frame: np.ndarray, size: int) -> np.ndarray:
    """Generalisation of the original ``ResizeWithAspectRatio(width=512)``.

    Scales so the *longer* side is at most ``size`` (portrait phone videos
    would otherwise stay huge) and forces even dimensions for H.264.
    """
    h, w = frame.shape[:2]
    r = min(1.0, size / max(h, w))
    width = int(round(w * r))
    width -= width % 2
    height = int(round(h * r))
    height -= height % 2
    if (width, height) == (w, h):
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


class H264Writer:
    """Write BGR frames to a browser-playable MP4 through an ffmpeg pipe."""

    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int = 23):
        self.path = Path(path)
        self.size = (width, height)
        cmd = [
            ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", f"{fps:.6f}", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            raise VideoError("Frame size changed mid-stream.")
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise VideoError(f"Video encoder stopped: {self._stderr()}") from exc
        self.frames += 1

    def close(self) -> None:
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        code = self._proc.wait(timeout=300)
        if code != 0:
            raise VideoError(f"Video encoder failed ({code}): {self._stderr()}")

    def abort(self) -> None:
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        finally:
            self._proc.kill()
            self._proc.wait()
            self.path.unlink(missing_ok=True)

    def _stderr(self) -> str:
        try:
            return (self._proc.stderr.read() or b"").decode(errors="replace")[-400:]
        except Exception:
            return ""


def transcode_preview(src: Path, dst: Path, max_width: int = 1280) -> None:
    """Make a browser-playable H.264 copy of an upload (used for non-H.264 sources)."""
    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
        "-vf", f"scale='min({max_width},iw)':-2", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(dst),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if res.returncode != 0:
        dst.unlink(missing_ok=True)
        raise VideoError(f"Preview transcode failed: {res.stderr[-400:]}")
