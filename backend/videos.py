"""Upload validation and storage."""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
from fastapi import UploadFile

from motiontrack.video_io import VideoError, probe, resize_long_side, transcode_preview

from . import settings


class UploadError(ValueError):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def _safe_id(value: str) -> str:
    try:
        return uuid.UUID(value).hex
    except (ValueError, AttributeError, TypeError):
        raise UploadError("Unknown video id.", 404)


def video_dir(video_id: str) -> Path:
    return settings.UPLOAD_DIR / _safe_id(video_id)


async def save_upload(file: UploadFile) -> dict:
    name = Path(file.filename or "").name
    ext = Path(name).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(e.lstrip(".") for e in settings.ALLOWED_EXTENSIONS))
        raise UploadError(f"Unsupported file type '{ext or 'none'}'. Allowed: {allowed}.", 415)

    vid = uuid.uuid4().hex
    vdir = settings.UPLOAD_DIR / vid
    vdir.mkdir(parents=True)
    src = vdir / f"source{ext}"
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    try:
        with open(src, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise UploadError(f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit.", 413)
                out.write(chunk)
        if size == 0:
            raise UploadError("The uploaded file is empty.")

        try:
            info = probe(src)
        except VideoError as exc:
            raise UploadError(f"Not a readable video: {exc}") from exc
        if min(info.width, info.height) < settings.MIN_DIMENSION:
            raise UploadError("Video resolution is too small to analyse.")
        if info.duration_s > settings.MAX_DURATION_S:
            raise UploadError(f"Video is longer than the {int(settings.MAX_DURATION_S)} s limit.")
        if info.frame_count < 2:
            raise UploadError("Video needs at least two frames for motion analysis.")

        _write_poster(src, vdir / "poster.jpg")
        playable = (ext, info.codec) in settings.BROWSER_PLAYABLE
        meta = {
            "id": vid,
            "filename": name,
            "size_bytes": size,
            "uploaded_at": time.time(),
            "info": info.to_dict(),
            "source_file": src.name,
            "preview": "ready" if playable else "pending",
        }
        _write_meta(vdir, meta)
        if not playable:
            threading.Thread(target=_make_preview, args=(vdir,), daemon=True).start()
        return meta
    except BaseException:
        shutil.rmtree(vdir, ignore_errors=True)
        raise


def _write_poster(src: Path, dst: Path) -> None:
    cap = cv2.VideoCapture(str(src))
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(dst), resize_long_side(frame, 1280), [cv2.IMWRITE_JPEG_QUALITY, 88])


def _make_preview(vdir: Path) -> None:
    meta = load_meta(vdir.name)
    try:
        transcode_preview(vdir / meta["source_file"], vdir / "preview.mp4")
        meta["preview"] = "ready"
    except Exception:
        meta["preview"] = "failed"
    _write_meta(vdir, meta)


def _write_meta(vdir: Path, meta: dict) -> None:
    tmp = vdir / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(vdir / "meta.json")


def load_meta(video_id: str) -> dict:
    p = video_dir(video_id) / "meta.json"
    if not p.is_file():
        raise UploadError("Unknown video id.", 404)
    return json.loads(p.read_text())


def source_path(video_id: str) -> Path:
    meta = load_meta(video_id)
    return video_dir(video_id) / meta["source_file"]


def playable_path(video_id: str) -> Optional[Path]:
    meta = load_meta(video_id)
    vdir = video_dir(video_id)
    if meta.get("preview") != "ready":
        return None
    preview = vdir / "preview.mp4"
    return preview if preview.is_file() else vdir / meta["source_file"]


def delete_video(video_id: str) -> None:
    shutil.rmtree(video_dir(video_id), ignore_errors=True)


def cleanup_expired(max_age_hours: float) -> int:
    """Remove uploads and job folders older than the retention window."""
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for base in (settings.UPLOAD_DIR, settings.JOB_DIR):
        for d in base.iterdir():
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
    return removed
