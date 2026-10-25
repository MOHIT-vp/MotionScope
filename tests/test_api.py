"""HTTP API: validation, job lifecycle, cancellation and file access."""

import os
import tempfile
import time

import pytest

os.environ["MT_DATA_DIR"] = tempfile.mkdtemp(prefix="mt-test-")

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402

client = TestClient(app)


def wait(job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError("job timed out")


@pytest.fixture(scope="module")
def video_id(synthetic_video):
    with open(synthetic_video, "rb") as f:
        r = client.post("/api/videos", files={"file": ("block.mp4", f, "video/mp4")})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["info"]["width"] == 640 and body["preview"] == "ready"
    return body["id"]


def test_health():
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["ffmpeg"] is True


@pytest.mark.parametrize("name,data,status", [
    ("notes.txt", b"hello", 415),
    ("fake.mp4", b"\x00" * 4096, 422),
    ("empty.mov", b"", 422),
])
def test_rejects_bad_uploads(name, data, status):
    r = client.post("/api/videos", files={"file": (name, data)})
    assert r.status_code == status
    assert isinstance(r.json()["detail"], str)


def test_failed_upload_leaves_no_files():
    from backend import settings
    before = set(settings.UPLOAD_DIR.iterdir())
    client.post("/api/videos", files={"file": ("fake.mp4", b"\x01" * 1000)})
    assert set(settings.UPLOAD_DIR.iterdir()) == before


def test_video_endpoints(video_id):
    assert client.get(f"/api/videos/{video_id}").status_code == 200
    assert client.get(f"/api/videos/{video_id}/poster").headers["content-type"] == "image/jpeg"
    r = client.get(f"/api/videos/{video_id}/stream", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100


@pytest.mark.parametrize("path", ["/api/videos/not-a-uuid", "/api/videos/" + "0" * 32, "/api/jobs/" + "f" * 32])
def test_unknown_ids(path):
    assert client.get(path).status_code == 404


def test_job_validation(video_id):
    bad_cal = {"points": [[0, 0], [1, 1], [0, 1], [1, 0]], "width_m": 3, "length_m": 3}
    r = client.post("/api/jobs", json={"video_id": video_id, "calibration": bad_cal})
    assert r.status_code == 422 and "convex" in r.json()["detail"]
    assert client.post("/api/jobs", json={"video_id": video_id, "process_size": 5000}).status_code == 422
    assert client.post("/api/jobs", json={"video_id": video_id, "detector": "magic"}).status_code == 422
    assert client.post("/api/jobs", json={"video_id": "0" * 32}).status_code == 404


def test_job_lifecycle_and_files(video_id):
    cal = {"points": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]], "width_m": 16, "length_m": 12}
    r = client.post("/api/jobs", json={"video_id": video_id, "process_size": 640, "detector": "motion",
                                       "calibration": cal})
    assert r.status_code == 202
    j = wait(r.json()["id"])
    assert j["status"] == "completed", j.get("error")
    assert j["progress"] == 1.0
    assert set(j["files"]) >= {"annotated.mp4", "report.json", "tracks.csv", "heatmap.png"}
    rep = client.get(f"/api/jobs/{j['id']}/report").json()
    assert rep["summary"]["speed"]["calibrated"] is True
    d = client.get(f"/api/jobs/{j['id']}/files/tracks.csv?download=true")
    assert d.status_code == 200 and "attachment" in d.headers["content-disposition"]
    # only whitelisted result names are served
    assert client.get(f"/api/jobs/{j['id']}/files/job.json").status_code == 404
    assert client.get(f"/api/jobs/{j['id']}/files/..%2Fjob.json").status_code == 404


def test_pause_resume_cancel(video_id):
    r = client.post("/api/jobs", json={"video_id": video_id, "process_size": 640, "detector": "motion"})
    jid = r.json()["id"]
    # wait until it is running, then pause
    for _ in range(100):
        if client.get(f"/api/jobs/{jid}").json()["status"] == "running":
            break
        time.sleep(0.05)
    p = client.post(f"/api/jobs/{jid}/pause").json()
    if p["status"] == "paused":
        f1 = p["frames_done"]
        time.sleep(0.8)
        assert client.get(f"/api/jobs/{jid}").json()["frames_done"] <= f1 + 2
        assert client.post(f"/api/jobs/{jid}/resume").json()["status"] in ("running", "completed")
    c = client.post(f"/api/jobs/{jid}/cancel").json()
    final = wait(jid)
    if c["status"] != "completed" and final["status"] == "cancelled":
        assert client.get(f"/api/jobs/{jid}/files/annotated.mp4").status_code == 409
    assert client.post(f"/api/jobs/{jid}/explode").status_code == 404


def test_demos_manifest():
    r = client.get("/api/demos")
    assert r.status_code == 200 and "demos" in r.json()
