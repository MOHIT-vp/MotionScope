# MotionScope — Technical Report

**Project:** Vehicle Speed Estimation and Motion Saliency Tracking (reconstruction & enhancement)
**Document status:** This report separates **historical findings** (what the original project contained, §2) from **new results** produced by the reconstructed system (§9). No result in this document is invented. Every number in §9 can be reproduced with the commands given.

---

## 1. Overview and objectives

Traffic monitoring needs to know *what moves, where, how fast*. Radar and loop sensors are expensive and cover single points; a camera covers a whole scene. The project's objective, as stated in the original report, is “a robust system for real-time vehicle speed estimation and motion analysis in surveillance video sequences” using Lucas–Kanade optical flow and motion saliency.

The reconstruction keeps that objective and the original algorithmic core, and adds what is needed to make it a usable, honest tool:

1. a browser application (upload → process → inspect → download);
2. persistent per-vehicle tracking instead of per-frame grouping;
3. speed estimation that accounts for perspective, and is shown only with calibration;
4. measured / estimated separation and uncertainty in every output;
5. tests with synthetic ground truth.

### Problem statement (from the original report)

- Track vehicles despite appearance changes and occlusions.
- Distinguish vehicle motion from camera jitter.
- Estimate real-world speed accurately from pixel displacement.

## 2. Recovered materials (historical evidence)

| File | What it is | What it tells us |
|---|---|---|
| `legacy/untitled0.py` | Colab export (Jan 2026) with three pipelines: (1) LK + saliency + speed, (2) saliency overlay, (3) plain LK keypoints | The implemented algorithm and all parameters |
| `legacy/Vehicle Speed Estimation Report.docx` | Original project report: introduction, literature, problem statement, method, equations, two output screenshots | Objectives, methodology and the original visual results |
| `main.pdf` | University LaTeX report **template** (placeholder Lorem-ipsum text, no project content) | Nothing about the project; kept locally, excluded from the repository |
| `ex-1.mp4` (originally `VIDEO-2026-01-06-10-57-47.mp4`) | 60 s, 1280×720, 25 fps static-camera dual-carriageway footage | The clip used in the report's screenshots |
| `ex-2.mp4`, `ex-3.mp4` | Additional samples added during reconstruction (1080p motorway; 4K portrait hand-held night scene; sample footage is not distributed with the repository) | Used for demonstrations and robustness checks |
| Reference repo `ShreyaPI/Lucas-Kanade-with-Motion-Saliency` | `LucasKanadeWithSaliency.py`, `SaliencyMap.py`, `lucaskanadewithKeypoints.py`, README, 2 screenshots | Same three scripts as the Colab file; the Colab notebook is a port of this repository |

### 2.1 Original method (verified from code)

- Resize frames to width 512, convert to grayscale.
- Motion saliency: `B_t = |I_t − I_{t−1}| > 30`; accumulated `S_t = 0.9·S_{t−1} + 0.1·B_t`.
- Corners: `goodFeaturesToTrack(maxCorners=200, qualityLevel=0.2, minDistance=5, blockSize=5)` inside the saliency mask, re-detected every 30 frames **or** when the saliency change ratio exceeds 0.1.
- Optical flow: `calcOpticalFlowPyrLK(winSize=15×15, maxLevel=2, criteria=(EPS|COUNT, 10, 0.03))`.
- Speed: `v = displacement_px × 0.1 m/px ÷ (1/fps) × 3.6`, per corner, per frame; points faster than 5 km/h are grouped into “vehicles” within a 20 px radius; average of the last 30 values is printed beside each group.

### 2.2 Issues found in the original implementation

These were established by reading the code and by running a faithful port (`scripts/legacy_baseline.py`) on the sample footage:

| Issue | Evidence |
|---|---|
| Corners are re-detected on **every** frame | `saliency_change > 0.1` held on 100 % of frames (500/500 on `ex-1`, 375/375 on `ex-2`) |
| Tracked points are not propagated in pipelines 1 and 2 (`p0 = good_new` exists only in pipeline 3) | Code inspection |
| “Vehicle” IDs are re-numbered every frame (`new_id = len(new_vehicle_positions)`), so speeds are averaged over unrelated points | Code inspection; 14.7 speed labels per frame on `ex-1` for a scene with a handful of cars |
| Fixed `scaling_factor = 0.1 m/px` ignores perspective and is not calibrated | On `ex-1` the original prints a median of **21.5 km/h** (p5–p95: 6.1–39.7) for traffic on a dual carriageway; the report's Fig. 1 shows the same 6–40 km/h range with overlapping labels |
| Output codec `mp4v` is not playable in browsers; Colab-only I/O | Code inspection |

## 3. System architecture

```
frontend/ (HTML/CSS/JS)  ──HTTP──▶  backend/ (FastAPI)
                                       ├── videos.py   upload validation, storage, previews
                                       ├── jobs.py     single-worker queue, pause/resume/cancel
                                       └── main.py     REST API + static hosting
                                                │
                                                ▼
                                   motiontrack/ (pure Python + OpenCV)
     video_io → saliency → flow → detection → tracker → calibration → render → pipeline
```

- **Separation:** `motiontrack` has no web dependencies and can be used from the CLI or tests; the backend only orchestrates; the frontend only talks HTTP.
- **Jobs** run one at a time on a background thread (the pipeline is CPU-bound and OpenCV already parallelises internally). Pause/resume/cancel use cooperative checkpoints between frames; cancelled or failed jobs delete partial outputs; paused time is excluded from processing-time statistics.
- **Storage:** `data/uploads/<uuid>/` (source, poster, optional preview) and `data/jobs/<uuid>/` (results + `job.json`). Finished jobs survive a server restart. Folders older than `MT_RETENTION_HOURS` (default 72) are removed at start-up.
- **Video output:** frames are piped to the ffmpeg binary bundled with `imageio-ffmpeg` and encoded as H.264/yuv420p with `+faststart`, so results stream in any browser. Non-H.264 uploads get an H.264 preview automatically.

## 4. Lucas–Kanade optical flow

Brightness constancy, `I(x+u, y+v, t+1) = I(x, y, t)`, linearised gives one equation per pixel, `I_x u + I_y v + I_t = 0`. LK assumes the flow is constant in a window W and solves

```
[ Σ I_x²     Σ I_x I_y ] [u]     [ Σ I_x I_t ]
[ Σ I_x I_y  Σ I_y²    ] [v] = − [ Σ I_y I_t ]
```

The 2×2 structure tensor is invertible and well conditioned only where the image has gradient in two directions, so points come from the Shi–Tomasi detector, which selects exactly where its smaller eigenvalue is large. A pyramid (3 levels) lets the method follow displacements larger than the 15×15 window.

**What changed (`motiontrack/flow.py`):**
- Points persist across frames with stable IDs, which gives real per-point trajectories.
- **Forward–backward check:** each point is tracked t→t+1 and back; if it does not return within 1 px, it is discarded.
- New corners are sampled only in the (dilated) saliency mask and away from existing points (the original idea, applied as top-up every 3 frames).
- Corners that have not moved for 8 frames are retired every `recalc_interval` (30).
- Camera motion is subtracted from each displacement, so flow vectors and statistics describe object motion.

All original parameter values are kept as defaults (`motiontrack/config.py`).

## 5. Motion saliency, detection and tracking

### 5.1 Saliency and camera-motion compensation
The original saliency formula is kept unchanged (`motiontrack/saliency.py`). Before differencing, corners outside the current foreground are tracked between frames and a similarity transform is fitted with RANSAC (`estimateAffinePartial2D`). If the shift exceeds 0.3 px, the previous frame is warped onto the current one. On a synthetic hand-held video, this changes the result from 48 spurious tracks to the single correct one (§9.1).

### 5.2 Why a detector is needed in addition to LK
Optical flow describes **points**, not **objects**. Grouping points by distance (the original approach) cannot keep identities. Background subtraction (MOG2) plus saliency gives object blobs, but on `ex-1` adjacent cars merge into one blob and dark cars on dark asphalt fragment (observed while developing; see `detection.py` docstring). The system therefore supports two detectors:

| Mode | How | Pros / cons |
|---|---|---|
| `yolox` (default when the model is present) | YOLOX-S ONNX from the official OpenCV model zoo, run with `cv2.dnn` every 2nd frame; COCO classes bicycle/car/motorcycle/bus/truck; confidence ≥ 0.4 | Separates adjacent vehicles, gives a class; boxes are kept only if ≥ 4 % of their area is moving or the median LK flow inside is ≥ 0.5 px, so parked cars and signs are ignored |
| `motion` (fallback) | MOG2 foreground ∪ saliency, morphology, contours | No model needed; class-agnostic; merges close vehicles |

### 5.3 Tracker
`motiontrack/tracker.py` follows SORT's predict → associate → update loop (Bewley et al. 2016, cited in the original report), with one change: **prediction uses the median LK displacement of the corners inside each box**, so optical flow is the motion model and detections correct the geometry. Between detector frames, boxes are carried by LK alone. Association is greedy IoU (≥ 0.2); tracks are confirmed after 4 matched detections and dropped after 12 missed detector frames or on leaving the image. Trajectories are lightly smoothed (EMA 0.5) to suppress box-outline flicker.

### 5.4 Per-track measurements (all measured, in source-video pixels)
Start/end time, duration, frames observed, net displacement and path length, direction (degrees counter-clockwise from image-right, plus an 8-way label), median LK velocity (px/s, camera-compensated), box-centre velocity, and a normalised trajectory polyline.

## 6. Speed estimation and calibration

### 6.1 Why pixels are not km/h
With perspective, the metres represented by one pixel depend on depth. A car covering 10 m spans many pixels near the camera and few near the horizon, so the original constant factor makes the same car appear to change speed and gives wrong absolute values (§2.2).

### 6.2 Method
The user marks four image points **on the road surface** forming a real rectangle, A→B (width) and B→C (length), and enters both distances. `cv2.getPerspectiveTransform` gives a homography H from the image to road-plane metres. For each confirmed vehicle:

1. the ground contact point is approximated by the **bottom-centre** of its box;
2. over a sliding window of 0.6 s (15 frames at 25 fps) the ground positions are fitted with a least-squares line in time, and the slope is the speed (much less noisy than frame-to-frame differences);
3. a window is used only if every sample lies within one rectangle length before or after the calibrated area (and within one length sideways), and no box touches the image border (a clipped box has a wrong bottom edge);
4. the vehicle's speed is the **median** of its window speeds, reported with the **interquartile range**, the number of windows, the fit residual, and the straight-line ground distance ÷ duration as a cross-check.

### 6.3 Requirements and sources of error
- **Reference distances.** Measure them on site, or use documented road-marking standards, and state which. Errors scale speed proportionally.
- **Point placement.** Speed is very sensitive to it. In the browser test (§9.3), clicking points about 6 % of the frame away from the intended ones changed the median from 123 km/h to 178 km/h.
- **Frame rate.** Speed ∝ fps. The container value is used unless overridden. Variable-frame-rate phone videos may need an override.
- **Planarity.** The road must be close to flat across the measured area.
- **Ground point.** Shadows, tall vehicles and partial occlusion move the bottom edge of the box.

### 6.4 Demo calibrations (assumptions, not surveys)
`samples/ex-1.calibration.json` and `samples/ex-2.calibration.json` were built from lane-marking positions located automatically on a median background image, with **assumed** dimensions:
- `ex-1` (UK): lane width 3.65 m; lane-line cycle 9 m (2 m mark + 7 m gap); cross-sections approximated by image rows.
- `ex-2` (Germany): middle-lane width 3.5 m; lane-line cycle 18 m (6 m mark + 12 m gap).

These are typical standard values and were **not measured** for these specific roads. Speeds from them are illustrative estimates.

## 7. Installation and execution

See the README for commands. Summary:

| Task | Command |
|---|---|
| Install | `python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt` |
| Optional detector | `.venv/bin/python scripts/fetch_model.py` |
| Web app | `.venv/bin/uvicorn backend.main:app --port 8000` |
| CLI | `.venv/bin/python -m motiontrack.cli VIDEO -o OUT [--calibration FILE] [--size 960] [--detector auto/yolox/motion] [--fps N] [--frames N] [--saliency]` |
| Demos | `.venv/bin/python scripts/make_demos.py` |
| Original algorithm | `.venv/bin/python scripts/legacy_baseline.py VIDEO --frames 500` |
| Tests | `.venv/bin/python -m pytest -q` |

Environment variables: `MT_DATA_DIR`, `MT_MAX_UPLOAD_MB` (1024), `MT_MAX_DURATION_S` (900), `MT_RETENTION_HOURS` (72).

## 8. API

All responses are JSON unless noted; errors are `{"detail": "<message>"}`.

| Method & path | Description |
|---|---|
| `GET /api/health` | Status, ffmpeg and YOLOX availability, upload limits |
| `POST /api/videos` (multipart `file`) | Upload. 201 with `{id, filename, size_bytes, info:{width,height,fps,frame_count,duration_s,codec,has_audio}, preview}`. 415 unsupported extension, 413 too large, 422 unreadable/empty/too short/too long |
| `GET /api/videos/{id}` | Metadata |
| `GET /api/videos/{id}/stream` | Browser-playable video (Range supported); 409 while a preview is being transcoded |
| `GET /api/videos/{id}/poster` | First frame (JPEG) used for calibration |
| `DELETE /api/videos/{id}` | Remove an upload |
| `POST /api/jobs` | Start processing. Body: `video_id`, optional `process_size` (320–1920), `detector` (`auto`/`yolox`/`motion`), `detect_every`, `max_frames`, `start_frame`, `fps_override`, `compensate_camera`, `max_corners`, `diff_threshold`, `layers{flow_vectors,feature_trails,saliency,objects,hud,calibration}`, `calibration{points[4][2] normalised, width_m, length_m, label}`. 202 with the job |
| `GET /api/jobs/{id}` | `status` (queued/running/paused/completed/failed/cancelled), `frames_done`, `frames_total`, `progress`, `throughput_fps`, `eta_s`, `error`, `files` |
| `POST /api/jobs/{id}/pause` · `/resume` · `/cancel` | Job control |
| `GET /api/jobs/{id}/report` | Full `report.json` |
| `GET /api/jobs/{id}/files/{name}[?download=true]` | One of `annotated.mp4`, `report.json`, `tracks.csv`, `heatmap.png`, `trajectories.png`, `thumbnail.jpg` (whitelist) |
| `GET /api/demos` | Demonstration manifest |

`report.json` (`schema: motiontrack.report/1`) contains `video`, `processing` (incl. full config), `summary`, `direction_histogram`, `timeseries` (t, active features/objects, mean flow px/s, saliency coverage %, camera shift px), `tracks[]` and `notes[]`.

## 9. Testing methodology and results (new, reproducible)

### 9.1 Automated tests: `pytest` (35 tests, all passing)

Synthetic videos are generated in `tests/conftest.py`: a textured block moves at exactly 4 px/frame (100 px/s) to the right at 25 fps over a textured background. A second video adds ±3 px random camera shake. With a top-down calibration of 20 px/m, the true speed is 5 m/s = 18 km/h.

| Check | Truth | Measured |
|---|---|---|
| LK velocity, static camera | 100 px/s | 100.0 px/s |
| Direction | 0° (right) | 0.0°, “right” |
| Calibrated speed | 18.0 km/h | 18.0 km/h (IQR 18.0–19.0) |
| Calibrated speed with `fps_override=50` | 36.0 km/h | 36.0 km/h (IQR 35.8–36.9) |
| Shaky camera, compensation on | 1 object, 100 px/s, right | 1 track, 100.0 px/s, 0.1°; mean camera shift 3.55 px |
| Shaky camera, compensation off | — | 48 spurious tracks |

Other tests: homography maps the calibration corners exactly; perspective quads give equal widths at near and far edges; self-intersecting/collinear/out-of-range quads and invalid distances are rejected; H.264 output is decodable with the right size and frame count; `max_frames` and layer options; cancellation raises and removes the partial video; unreadable files raise `VideoError`. API tests cover health, rejection of `.txt` (415), fake `.mp4` (422) and empty files (422) without leaving files behind, Range streaming (206), unknown IDs (404), calibration/size/detector validation (422), the full job lifecycle with calibrated output, download headers, the file-name whitelist (including `..%2F` traversal), and pause/resume/cancel.

### 9.2 Sample footage (no ground truth available)

Throughput on an Apple-silicon Mac, CPU only, 960 px processing size:

| Run | Detector | Throughput |
|---|---|---|
| `ex-1`, 500 frames | motion | 52.8 fps |
| `ex-1`, 500 frames | YOLOX every 2nd frame | 20.9 fps |
| `ex-3` (4K portrait input), 240 frames | YOLOX | 17.6 fps |
| demos (375/375/360 frames) | YOLOX | 19.5 / 20.1 / 18.6 fps |

Speed estimates under the **assumed** calibrations (§6.4): on `ex-2`, 31 of 35 tracked vehicles received an estimate, median 124.9 km/h, range 77–172 km/h; on `ex-1` (first 20 s, 500 frames), 10 of 15 tracked vehicles received an estimate, median 112.8 km/h, range 102–138 km/h. Both are plausible for these road types, but they **cannot be verified** without true speeds, so they are not accuracy claims. For comparison, the original algorithm reports a median of 21.5 km/h on the same `ex-1` footage (§2.2).

Re-running `ex-2`'s first 15 s from a re-encoded 960×540 copy gave a median of 123.4 km/h, against 124.9 km/h from the 1080p original, which shows the estimate is stable across resolution and compression.

### 9.3 Manual end-to-end verification in a browser
Performed against the running app:
1. Landing page, method and speed sections, demo cards (videos load, Processed/Original toggle).
2. Drag-and-drop upload of a video → workspace with metadata chips and preview.
3. Calibration dialog: validation with fewer than 4 points; 4 points placed by clicking; distances entered; calibration status shown.
4. Start → live progress (frames, fps, ETA) → **Pause** (frame count stays fixed) → **Resume** → Completed.
5. Results: 8 stat tiles tagged Metadata / Measured / Estimate, motion-over-time chart with hover, direction rose, trajectories over the first frame, heat map, track table, notes.
6. Processed video plays in the browser (H.264, 960×540); compare view with synced seeking and an adjustable wipe.
7. All five downloads return `Content-Disposition: attachment` with non-empty bodies.
8. Refreshing the page restores the job from the URL.
9. Invalid uploads through the UI: `.txt`, fake `.mp4` and empty files each show a specific error message.
10. Layout at desktop width and at 375 px mobile width: no horizontal page overflow; the table and chart scroll inside their cards.

**Not tested:** uploads near the 1 GB limit; very long videos (> 2 min) end to end in the browser; Safari and Firefox (Chromium-based pane only); Windows/Linux hosts; GPU acceleration (not implemented).

## 10. Limitations and future work

- **Calibration is manual** and dominates speed accuracy. Future work: vanishing-point auto-calibration from lane lines, or calibration from known vehicle dimensions.
- **Ground point approximation** (box bottom-centre) is biased by shadows and tall vehicles; a 3-D box or wheel-contact estimate would help.
- **Detector:** YOLOX-S is a general COCO model run on CPU every 2nd frame. Night scenes, heavy rain and dense queues reduce recall. Motion-blob mode merges adjacent vehicles.
- **Association** is greedy IoU, so identity switches are possible under long occlusions. Appearance features (DeepSORT) or Hungarian matching would reduce them.
- **No ground-truth speed dataset** was available, so real-footage speeds remain unverified estimates. A validation set (e.g. BrnoCompSpeed) is the obvious next step.
- **Single worker** processes one job at a time; there are no user accounts, and uploads are local.
- **Camera compensation** models a similarity transform; strong parallax or zooming is not handled.
