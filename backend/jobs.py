"""Background processing jobs with progress, pause / resume and cancel.

Jobs run one at a time on a single worker thread: the pipeline is CPU-bound
(OpenCV already parallelises internally), so running several in parallel only
slows each one down.  Further jobs wait in a FIFO queue.
"""

from __future__ import annotations

import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

from motiontrack import PipelineConfig
from motiontrack.calibration import CalibrationError
from motiontrack.pipeline import JobControl, PipelineCancelled, run_pipeline
from motiontrack.video_io import VideoError

from . import settings, videos

TERMINAL = {"completed", "failed", "cancelled"}


@dataclass
class Job:
    id: str
    video_id: str
    config: dict
    status: str = "queued"          # queued | running | paused | completed | failed | cancelled
    frames_done: int = 0
    frames_total: int = 0
    throughput_fps: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None

    def public(self) -> dict:
        d = asdict(self)
        d["progress"] = round(self.frames_done / self.frames_total, 4) if self.frames_total else 0.0
        if self.status == "running" and self.throughput_fps > 0 and self.frames_total:
            d["eta_s"] = round((self.frames_total - self.frames_done) / self.throughput_fps, 1)
        else:
            d["eta_s"] = None
        if self.status == "completed":
            d["files"] = {name: f"/api/jobs/{self.id}/files/{name}"
                          for name in settings.RESULT_FILES if (job_dir(self.id) / name).is_file()}
        return d


def job_dir(job_id: str) -> Path:
    try:
        return settings.JOB_DIR / uuid.UUID(job_id).hex
    except (ValueError, AttributeError, TypeError):
        raise KeyError(job_id)


class JobManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, Job] = {}
        self.controls: Dict[str, JobControl] = {}
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.lock = threading.Lock()
        self._load_finished()
        self.worker = threading.Thread(target=self._work, daemon=True, name="motiontrack-worker")
        self.worker.start()

    # ------------------------------------------------------------ public API
    def submit(self, video_id: str, config: dict) -> Job:
        src = videos.source_path(video_id)          # raises UploadError(404) if unknown
        cfg = PipelineConfig.from_dict(config)       # raises CalibrationError / TypeError
        meta = videos.load_meta(video_id)
        total = max(0, meta["info"]["frame_count"] - cfg.start_frame)
        if cfg.max_frames:
            total = min(total, cfg.max_frames)
        job = Job(uuid.uuid4().hex, video_id, cfg.to_dict(), frames_total=total)
        job_dir(job.id).mkdir(parents=True)
        with self.lock:
            self.jobs[job.id] = job
            self.controls[job.id] = JobControl()
        self._persist(job)
        self.queue.put(job.id)
        assert src.exists()
        return job

    def get(self, job_id: str) -> Job:
        job_dir(job_id)  # validates format
        job = self.jobs.get(uuid.UUID(job_id).hex)
        if job is None:
            raise KeyError(job_id)
        return job

    def pause(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status == "running":
            self.controls[job.id].pause()
            job.status = "paused"
        return job

    def resume(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status == "paused":
            self.controls[job.id].resume()
            job.status = "running"
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status in TERMINAL:
            return job
        ctrl = self.controls.get(job.id)
        if ctrl:
            ctrl.cancel()
        if job.status == "queued":
            self._finish(job, "cancelled")
        return job

    # -------------------------------------------------------------- worker
    def _work(self) -> None:
        while True:
            job_id = self.queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status != "queued":
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        ctrl = self.controls[job.id]
        out = job_dir(job.id)
        job.status = "running"
        job.started_at = time.time()

        def progress(done: int, total: int, fps: float) -> None:
            job.frames_done, job.throughput_fps = done, round(fps, 1)
            if total:
                job.frames_total = total

        try:
            cfg = PipelineConfig.from_dict(job.config)
            run_pipeline(videos.source_path(job.video_id), out, cfg, progress, ctrl)
            self._finish(job, "completed")
        except PipelineCancelled:
            self._clear_outputs(out)
            self._finish(job, "cancelled")
        except (VideoError, CalibrationError, ValueError) as exc:
            self._clear_outputs(out)
            self._finish(job, "failed", str(exc))
        except Exception as exc:  # unexpected: keep a trace for debugging
            traceback.print_exc()
            self._clear_outputs(out)
            self._finish(job, "failed", f"Processing error: {exc.__class__.__name__}: {exc}")

    def _finish(self, job: Job, status: str, error: Optional[str] = None) -> None:
        job.status = status
        job.error = error
        job.finished_at = time.time()
        self.controls.pop(job.id, None)
        self._persist(job)

    @staticmethod
    def _clear_outputs(out: Path) -> None:
        for name in settings.RESULT_FILES:
            (out / name).unlink(missing_ok=True)

    def _persist(self, job: Job) -> None:
        d = job_dir(job.id)
        if d.is_dir():
            (d / "job.json").write_text(json.dumps(asdict(job), indent=1))

    def _load_finished(self) -> None:
        """Restore finished jobs after a restart so result links keep working."""
        for p in settings.JOB_DIR.glob("*/job.json"):
            try:
                job = Job(**json.loads(p.read_text()))
            except Exception:
                continue
            if job.status not in TERMINAL:
                job.status, job.error = "failed", "Server restarted while this job was running."
            self.jobs[job.id] = job

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        self.cancel(job_id)
        if job.status in TERMINAL:
            shutil.rmtree(job_dir(job.id), ignore_errors=True)
            self.jobs.pop(job.id, None)
