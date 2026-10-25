# Historical project materials

These files are the original project as it existed before reconstruction, kept in [`legacy/`](../legacy). The only edit is the removal of the team-member table from the report. `main.pdf` and the raw sample videos are kept locally and are not part of the repository.

| File | Notes |
|---|---|
| `legacy/untitled0.py` | Colab notebook export of the original implementation. Contains three pipelines (speed tracking, saliency overlay, plain LK keypoints). It needs Google Colab (`!pip`, `google.colab.files`) and does not run as a plain Python script. `scripts/legacy_baseline.py` is a faithful, runnable port of its first pipeline. |
| `legacy/Vehicle Speed Estimation Report.docx` | The original project report: introduction, literature survey, problem statement, methodology, equations (saliency accumulation, optical-flow constraint, speed formula), two output screenshots (“Speed Detection”, “Optical flow Detection”), conclusion, and references (Lucas & Kanade 1981; Shi & Tomasi 1994; Bewley et al. 2016). |
| `main.pdf` | University LaTeX report template (17 pages). It contains only placeholder text (“Lorem Ipsum”, «Project Title») and no project content. Kept locally and excluded from the repository (`.gitignore`). |
| `ex-1.mp4` | Input clip used for the report's screenshots (previously named `VIDEO-2026-01-06-10-57-47.mp4`). Not included in the repository. |

The reference repository `ShreyaPI/Lucas-Kanade-with-Motion-Saliency` holds the same three scripts (`LucasKanadeWithSaliency.py`, `SaliencyMap.py`, `lucaskanadewithKeypoints.py`) with identical parameters; the Colab notebook is a port of it. It contains no videos, only two screenshots, so no media from it is reused here.

Findings from the original report (as written in the original report, not re-verified):
- “The Lucas-Kanade optical flow method combined with motion saliency detection successfully captures vehicle motion patterns.”
- Future work proposed: “perspective-aware speed correction and automated calibration.” Perspective-aware speed through a ground-plane homography is now implemented; calibration is still manual.

The quantitative issues found when re-running the original algorithm are listed in `TECHNICAL_REPORT.md` §2.2.
