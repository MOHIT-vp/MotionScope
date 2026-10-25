# MotionScope

**Lucas–Kanade optical flow, motion saliency and vehicle tracking for traffic video, in the browser.**



https://github.com/user-attachments/assets/dc9c5cb7-482c-4070-a8f8-2d9f5a5642cd


---

## What it does

Drop a traffic video into the web app and MotionScope will:

- **Track motion** with pyramidal Lucas–Kanade optical flow on Shi–Tomasi corners, with a forward–backward check, per-point trajectories and flow vectors.
- **Find where motion happens** with an accumulated motion-saliency map (`S = 0.9·S + 0.1·B`), after removing camera shake with a RANSAC similarity transform.
- **Follow every vehicle** with a persistent ID: YOLOX detections (or motion blobs as a fallback) linked by a SORT-style tracker whose prediction step is the LK flow inside each box.
- **Estimate speed only when it can.** Mark four points on the road that form a rectangle of known size; a ground-plane homography turns motion into metres. Without calibration, results stay in pixels.
- **Export everything**: an annotated H.264 video, a JSON report, a CSV of tracks, a motion heat map and a trajectory map.

| Kind | Examples | Source |
|---|---|---|
| **Metadata** | duration, frame rate, resolution | the video file |
| **Measured** | displacement, flow (px/s), direction, durations, feature counts | the pixels |
| **Estimate** | speed (km/h) with interquartile range | the pixels plus your calibration |

## Quick start

Requires Python 3.9+. No system ffmpeg is needed for the app (`imageio-ffmpeg` ships a binary).

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

Optional but recommended: the YOLOX-S vehicle detector (35 MB, OpenCV model zoo, Apache-2.0, checksum verified). Without it the app uses motion blobs.

```bash
.venv/bin/python scripts/fetch_model.py
```

Start the app and open <http://localhost:8000>:

```bash
.venv/bin/uvicorn backend.main:app --port 8000
```

Or process a video from the command line:

```bash
.venv/bin/python -m motiontrack.cli your-video.mp4 -o output --calibration samples/ex-1.calibration.json
```

## How it works

![Processing pipeline](docs/img/pipeline.svg)

1. **Resize and grayscale** each frame (long side 640–1280 px).
2. **Estimate camera motion** from background corners and remove it.
3. **Motion saliency**: thresholded frame differences, accumulated over time.
4. **Lucas–Kanade tracking** of corners sampled inside the saliency mask.
5. **Detect moving vehicles** (YOLOX every second frame, kept only if the box moves).
6. **Track and measure**: persistent IDs, displacement, direction, px/s, and speed when calibrated.
7. **Render and report**: video overlays, JSON, CSV and images.

Speed is a least-squares fit over a 0.6 s window of ground-plane positions (the bottom-centre of each box), reported as a median with its interquartile range, and only near the calibrated area. Full method, equations, API and limitations: [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md).

## Web app

- **Landing page**: drag-and-drop upload, a before/after view of real output, the method, and three demonstrations.
- **Analysis workspace**: original / processed / side-by-side compare, grouped settings (resolution, detector, overlays, calibration, advanced), live progress with pause, resume and cancel.
- **Motion report**: stat tiles tagged measured / estimate, motion over time, direction rose, trajectories, heat map, track table, and downloads.

The REST API behind it (`/api/videos`, `/api/jobs`, `/api/jobs/{id}/report`, …) is documented in [docs/TECHNICAL_REPORT.md §8](docs/TECHNICAL_REPORT.md#8-api).

## Project layout

```
motiontrack/        computer-vision package (no web dependencies)
  saliency.py       motion saliency + camera-motion estimation
  flow.py           Lucas–Kanade feature tracker with persistent IDs
  detection.py      motion-blob detector and optional YOLOX detector
  tracker.py        SORT-style tracker driven by LK flow
  calibration.py    ground-plane homography and windowed speed fit
  pipeline.py       end-to-end processing, measurements, report writing
  render.py         overlays, heat map, trajectory image
  video_io.py       probing, frame reading, H.264 writing
  cli.py            command-line entry point
backend/            FastAPI app: upload validation, job queue, result files
frontend/           static HTML/CSS/JS interface (no build step)
demos/              generated demonstration clips + manifest for the landing page
samples/            calibration files used by the demos
scripts/            fetch_model.py, make_demos.py, legacy_baseline.py
tests/              pytest suite with synthetic ground-truth videos
docs/               technical report, historical notes, design brief, media
legacy/             the original Colab implementation and project report
```

## Testing

```bash
.venv/bin/pip install -r requirements-dev.txt
```

```bash
.venv/bin/python -m pytest -q
```

35 tests cover the geometry, the pipeline and the API. Synthetic videos with known motion give exact ground truth: measured flow 100.0 px/s against a true 100 px/s, and calibrated speed 18.0 km/h against a true 18 km/h. Details in [docs/TECHNICAL_REPORT.md §9](docs/TECHNICAL_REPORT.md#9-testing-methodology-and-results-new-reproducible).

## Demonstrations

The three clips on the landing page are generated by `scripts/make_demos.py` with this pipeline; their statistics are copied from each run's report.

| Demo | Footage | Shows |
|---|---|---|
| Optical flow & feature trajectories | UK dual carriageway, 15 s | LK vectors and corner trails |
| Vehicle tracking with calibrated speed | German motorway, 15 s | IDs, trajectories, km/h estimates under an **assumed** lane-marking calibration |
| Motion saliency, hand-held night shot | 4K portrait phone video, 12 s | Saliency heat with camera-shake compensation |

The raw sample footage is not included in the repository. To regenerate the demos, place `ex-1.mp4`, `ex-2.mp4` and `ex-3.mp4` in the project root and run `scripts/make_demos.py`.

## Limitations

- Speed accuracy depends on the calibration; the demo calibrations use assumed standard road-marking dimensions, not surveyed ones.
- The ground point is the bottom-centre of a box, so tall vehicles, shadows and occlusion bias it.
- Motion-blob mode can merge vehicles driving close together; YOLOX mode needs the model file.
- CPU only: roughly 14–50 fps at 960 px depending on detector and resolution.

## Background

MotionScope rebuilds the computer-vision mini-project *Vehicle Speed Estimation and Motion Saliency Tracking*. The original Colab implementation and report are in [`legacy/`](legacy); what was recovered, what was wrong with the original speed estimate, and how it was fixed are described in [docs/HISTORICAL_NOTES.md](docs/HISTORICAL_NOTES.md) and [docs/TECHNICAL_REPORT.md §2](docs/TECHNICAL_REPORT.md#2-recovered-materials-historical-evidence).

## References

- B. D. Lucas and T. Kanade, *An iterative image registration technique with an application to stereo vision*, IJCAI 1981.
- J. Shi and C. Tomasi, *Good features to track*, CVPR 1994.
- A. Bewley et al., *Simple online and realtime tracking*, ICIP 2016.
- Z. Ge et al., *YOLOX: Exceeding YOLO series in 2021*, 2021.
