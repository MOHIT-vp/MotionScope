"""FastAPI application: upload → job → results, plus the static frontend.

Run:  uvicorn backend.main:app --port 8000
"""

from __future__ import annotations

import json
import threading
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import motiontrack
from motiontrack.calibration import CalibrationError, GroundCalibration
from motiontrack.detection import yolox_available

from . import settings, videos
from .jobs import JobManager, job_dir

app = FastAPI(title="Motion Tracking Studio", version=motiontrack.__version__,
              description="Lucas–Kanade optical flow, motion saliency and vehicle tracking API")
manager = JobManager()


@app.on_event("startup")
def _startup() -> None:
    videos.cleanup_expired(settings.RETENTION_HOURS)
    # With a short retention window (hosted demo) expired files must also go
    # while the server keeps running, not only at the next restart.
    interval = min(3600.0, max(300.0, settings.RETENTION_HOURS * 3600 / 4))
    stop = threading.Event()

    def sweep() -> None:
        while not stop.wait(interval):
            try:
                videos.cleanup_expired(settings.RETENTION_HOURS)
            except Exception:
                pass

    threading.Thread(target=sweep, daemon=True, name="retention-sweep").start()


@app.exception_handler(videos.UploadError)
def _upload_error(_: Request, exc: videos.UploadError):
    return JSONResponse({"detail": str(exc)}, status_code=exc.status)


@app.exception_handler(RequestValidationError)
def _validation_error(_: Request, exc: RequestValidationError):
    msgs = []
    for e in exc.errors():
        loc = ".".join(str(p) for p in e.get("loc", ()) if p != "body")
        msg = str(e.get("msg", "invalid value")).replace("Value error, ", "")
        msgs.append(f"{loc}: {msg}" if loc else msg)
    return JSONResponse({"detail": "; ".join(msgs) or "Invalid request."}, status_code=422)


# ----------------------------------------------------------------- models
class CalibrationIn(BaseModel):
    points: List[List[float]] = Field(..., min_length=4, max_length=4)
    width_m: float = Field(..., gt=0)
    length_m: float = Field(..., gt=0)
    label: str = Field("", max_length=300)


class LayersIn(BaseModel):
    flow_vectors: bool = True
    feature_trails: bool = True
    saliency: bool = False
    objects: bool = True
    hud: bool = True
    calibration: bool = True


class JobIn(BaseModel):
    video_id: str
    process_size: int = Field(960, ge=320, le=1920)
    detector: str = Field("auto", pattern="^(auto|yolox|motion)$")
    detect_every: int = Field(2, ge=1, le=10)
    max_frames: Optional[int] = Field(None, ge=2)
    start_frame: int = Field(0, ge=0)
    fps_override: Optional[float] = Field(None, ge=1, le=240)
    compensate_camera: bool = True
    max_corners: int = Field(200, ge=20, le=1000)
    diff_threshold: int = Field(30, ge=5, le=120)
    layers: LayersIn = LayersIn()
    calibration: Optional[CalibrationIn] = None

    @field_validator("calibration")
    @classmethod
    def _check_cal(cls, v):
        if v is not None:
            try:
                GroundCalibration.from_dict(v.model_dump())
            except CalibrationError as exc:
                raise ValueError(str(exc))
        return v


# ------------------------------------------------------------------ meta
@app.get("/api/health")
def health():
    from motiontrack.video_io import ffmpeg_exe
    try:
        ffmpeg = bool(ffmpeg_exe())
    except Exception:
        ffmpeg = False
    return {"status": "ok", "version": motiontrack.__version__, "ffmpeg": ffmpeg,
            "yolox_model": yolox_available(), "max_upload_mb": settings.MAX_UPLOAD_MB,
            "max_duration_s": settings.MAX_DURATION_S,
            "max_process_size": settings.MAX_PROCESS_SIZE, "max_frames": settings.MAX_FRAMES,
            "retention_hours": settings.RETENTION_HOURS, "demo_note": settings.DEMO_NOTE or None,
            "allowed_extensions": sorted(settings.ALLOWED_EXTENSIONS)}


@app.get("/api/demos")
def demos():
    manifest = settings.DEMO_DIR / "manifest.json"
    if not manifest.is_file():
        return {"demos": []}
    return json.loads(manifest.read_text())


# ---------------------------------------------------------------- videos
@app.post("/api/videos", status_code=201)
async def upload_video(file: UploadFile = File(...)):
    return await videos.save_upload(file)


@app.get("/api/videos/{video_id}")
def get_video(video_id: str):
    return videos.load_meta(video_id)


@app.get("/api/videos/{video_id}/stream")
def stream_video(video_id: str):
    path = videos.playable_path(video_id)
    if path is None:
        raise HTTPException(409, "Browser preview is still being prepared.")
    return FileResponse(path, media_type="video/mp4" if path.suffix == ".mp4" else None)


@app.get("/api/videos/{video_id}/poster")
def poster(video_id: str):
    p = videos.video_dir(video_id) / "poster.jpg"
    if not p.is_file():
        raise HTTPException(404, "No poster frame.")
    return FileResponse(p, media_type="image/jpeg")


@app.delete("/api/videos/{video_id}", status_code=204)
def delete_video(video_id: str):
    videos.load_meta(video_id)
    videos.delete_video(video_id)


# ------------------------------------------------------------------ jobs
@app.post("/api/jobs", status_code=202)
def create_job(body: JobIn):
    if body.process_size > settings.MAX_PROCESS_SIZE:
        raise HTTPException(422, f"process_size: this server processes at most {settings.MAX_PROCESS_SIZE} px.")
    cfg = body.model_dump(exclude={"video_id"})
    if settings.MAX_FRAMES:
        cfg["max_frames"] = min(cfg["max_frames"] or settings.MAX_FRAMES, settings.MAX_FRAMES)
    try:
        job = manager.submit(body.video_id, cfg)
    except CalibrationError as exc:
        raise HTTPException(422, str(exc))
    return job.public()


def _job(job_id: str):
    try:
        return manager.get(job_id)
    except KeyError:
        raise HTTPException(404, "Unknown job id.")


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _job(job_id).public()


@app.post("/api/jobs/{job_id}/{action}")
def job_action(job_id: str, action: str):
    _job(job_id)
    if action == "pause":
        return manager.pause(job_id).public()
    if action == "resume":
        return manager.resume(job_id).public()
    if action == "cancel":
        return manager.cancel(job_id).public()
    raise HTTPException(404, "Unknown action.")


@app.get("/api/jobs/{job_id}/report")
def job_report(job_id: str):
    job = _job(job_id)
    p = job_dir(job.id) / "report.json"
    if job.status != "completed" or not p.is_file():
        raise HTTPException(409, f"Job is {job.status}; no report yet.")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str, download: bool = False):
    job = _job(job_id)
    if name not in settings.RESULT_FILES:
        raise HTTPException(404, "Unknown file.")
    p = job_dir(job.id) / name
    if job.status != "completed" or not p.is_file():
        raise HTTPException(409, f"Job is {job.status}; file not available.")
    fname = f"motiontrack_{job.id[:8]}_{name}" if download else None
    return FileResponse(p, media_type=settings.RESULT_FILES[name], filename=fname)


# --------------------------------------------------------------- static
if settings.DEMO_DIR.is_dir():
    app.mount("/demos", StaticFiles(directory=settings.DEMO_DIR), name="demos")
app.mount("/", StaticFiles(directory=settings.FRONTEND_DIR, html=True), name="frontend")
