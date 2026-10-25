"""Run the ORIGINAL speed-tracking loop (legacy/untitled0.py, first pipeline) outside Colab.

This is a line-for-line port of the original Colab notebook logic (same
parameters, same fixed ``scaling_factor = 0.1`` m/px, same per-frame vehicle
re-numbering) with the Colab I/O removed.  It exists only to quantify how
the original behaves next to the reconstructed pipeline:

    python scripts/legacy_baseline.py ex-1.mp4 --frames 500 [-o legacy.mp4]

It prints statistics; with -o it also writes the original-style video.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np


def resize(image, width=512):
    h, w = image.shape[:2]
    r = width / float(w)
    return cv2.resize(image, (width, int(h * r)), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    feature_params = dict(maxCorners=200, qualityLevel=0.2, minDistance=5, blockSize=5)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    ret, old_frame = cap.read()
    old_frame = resize(old_frame)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    out = None
    if a.out:
        out = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, old_frame.shape[1::-1])
    prev_saliency = np.zeros_like(old_gray)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    frame_count, recalc_interval = 0, 30
    scaling_factor, frame_time = 0.1, 1 / fps
    speed_history, vehicle_positions = {}, {}

    redetections = 0
    labels_per_frame, all_label_speeds, max_id = [], [], 0
    while True:
        if a.frames and frame_count >= a.frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_saliency = cv2.absdiff(old_gray, frame_gray)
        _, motion_saliency = cv2.threshold(motion_saliency, 30, 255, cv2.THRESH_BINARY)
        saliency_diff = cv2.absdiff(prev_saliency, motion_saliency)
        saliency_change = np.sum(saliency_diff) / (np.sum(motion_saliency) + 1e-6)
        if frame_count % recalc_interval == 0 or p0 is None or len(p0) == 0 or saliency_change > 0.1:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=motion_saliency, **feature_params)
            redetections += 1
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new, good_old = p1[st == 1], p0[st == 1]
            new_vehicle_positions, new_speed_history = {}, {}
            for new, old in zip(good_new, good_old):
                a_, b = new.ravel()
                c, d = old.ravel()
                speed_kmh = np.sqrt((a_ - c) ** 2 + (b - d) ** 2) * scaling_factor / frame_time * 3.6
                if speed_kmh > 5:
                    assigned = False
                    for vehicle_id, (cx, cy) in vehicle_positions.items():
                        h, w = motion_saliency.shape
                        thr = 20 / (1 + motion_saliency[max(0, min(h - 1, int(b))), max(0, min(w - 1, int(a_)))] / 255.0)
                        if np.hypot(a_ - cx, b - cy) < thr:
                            new_vehicle_positions[vehicle_id] = ((cx + a_) / 2, (cy + b) / 2)
                            new_speed_history.setdefault(vehicle_id, speed_history.get(vehicle_id, [])).append(speed_kmh)
                            if len(new_speed_history[vehicle_id]) > 30:
                                new_speed_history[vehicle_id].pop(0)
                            assigned = True
                            break
                    if not assigned:
                        new_id = len(new_vehicle_positions)
                        new_vehicle_positions[new_id] = (a_, b)
                        new_speed_history[new_id] = [speed_kmh]
            vehicle_positions, speed_history = new_vehicle_positions, new_speed_history
            labels_per_frame.append(len(vehicle_positions))
            for vid, (x, y) in vehicle_positions.items():
                max_id = max(max_id, vid)
                s = float(np.mean(speed_history[vid]))
                all_label_speeds.append(s)
                if out is not None:
                    cv2.putText(frame, f"{s:.1f} km/h", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if out is not None:
            out.write(cv2.addWeighted(frame, 0.7, cv2.cvtColor(motion_saliency, cv2.COLOR_GRAY2BGR), 0.3, 0))
        frame_count += 1
        old_gray = frame_gray.copy()
        prev_saliency = motion_saliency.copy()

    cap.release()
    if out is not None:
        out.release()
    sp = np.array(all_label_speeds) if all_label_speeds else np.zeros(1)
    print(json.dumps({
        "frames": frame_count,
        "corner_redetections": redetections,
        "redetection_rate": round(redetections / max(1, frame_count), 3),
        "mean_speed_labels_per_frame": round(float(np.mean(labels_per_frame)), 1) if labels_per_frame else 0,
        "max_vehicle_id_used_in_a_frame": int(max_id),
        "label_speed_kmh_p5_p50_p95": [round(float(v), 1) for v in np.percentile(sp, [5, 50, 95])],
    }, indent=1))


if __name__ == "__main__":
    main()
