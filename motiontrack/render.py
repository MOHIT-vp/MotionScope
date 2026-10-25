"""Overlay rendering for the annotated output video and summary images."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np

from .flow import FlowSample

FONT = cv2.FONT_HERSHEY_SIMPLEX
AMBER = (40, 178, 255)      # BGR
CYAN = (230, 214, 64)
WHITE = (245, 245, 245)
INK = (24, 20, 18)

# Hue wheel LUT: motion direction → colour (same convention as the UI legend).
_HUE = cv2.cvtColor(
    np.array([[[int(h / 2), 210, 255] for h in range(360)]], np.uint8), cv2.COLOR_HSV2BGR
)[0]


def direction_color(dx: float, dy: float):
    ang = int(np.degrees(np.arctan2(-dy, dx))) % 360
    return tuple(int(c) for c in _HUE[ang])


def draw_saliency(frame: np.ndarray, accumulated: np.ndarray, strength: float = 0.55) -> np.ndarray:
    norm = np.clip(accumulated * 2.2, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    alpha = (norm.astype(np.float32) / 255.0 * strength)[..., None]
    return (frame * (1 - alpha) + heat * alpha).astype(np.uint8)


def draw_trails(frame: np.ndarray, trails: Dict[int, Iterable], min_len: int = 3) -> None:
    for pts in trails.values():
        pts = list(pts)
        if len(pts) < min_len:
            continue
        dx, dy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
        if dx * dx + dy * dy < 4.0:  # skip near-static corners to keep the view clean
            continue
        col = direction_color(dx, dy)
        n = len(pts)
        for i in range(1, n):
            a = i / n
            c = tuple(int(v * a + 30 * (1 - a)) for v in col)
            cv2.line(frame, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                     (int(pts[i][0]), int(pts[i][1])), c, 1, cv2.LINE_AA)


def draw_flow(frame: np.ndarray, samples: Sequence[FlowSample], gain: float = 3.0, min_mag: float = 0.4) -> None:
    for s in samples:
        mx, my = float(s.motion[0]), float(s.motion[1])
        if mx * mx + my * my < min_mag * min_mag:
            continue
        x, y = float(s.new[0]), float(s.new[1])
        col = direction_color(mx, my)
        cv2.arrowedLine(frame, (int(x - mx * gain), int(y - my * gain)), (int(x), int(y)),
                        col, 1, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(frame, (int(x), int(y)), 2, col, -1, cv2.LINE_AA)


def _label(frame: np.ndarray, text: str, org, color=AMBER, scale: float = 0.45) -> None:
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(org[0]), int(org[1])
    y = max(th + 6, y)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 8, y + 2), color, -1)
    cv2.putText(frame, text, (x + 4, y - 3), FONT, scale, INK, 1, cv2.LINE_AA)


def _brackets(frame, box, color, t=2):
    x, y, w, h = [int(v) for v in box]
    L = max(6, int(min(w, h) * 0.22))
    for (px, py, sx, sy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(frame, (px, py), (px + sx * L, py), color, t, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + sy * L), color, t, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)


def draw_objects(frame: np.ndarray, objects: List[dict]) -> None:
    """objects: dicts with box, label, trail (list of (x, y))."""
    for o in objects:
        trail = o.get("trail") or []
        if len(trail) > 1:
            pts = np.array(trail, np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, CYAN, 2, cv2.LINE_AA)
        _brackets(frame, o["box"], AMBER)
        x, y = o["box"][0], o["box"][1]
        _label(frame, o["label"], (x, y - 4))


def draw_calibration(frame: np.ndarray, quad: np.ndarray) -> None:
    overlay = frame.copy()
    cv2.fillPoly(overlay, [quad.reshape(-1, 1, 2)], (255, 200, 80))
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.polylines(frame, [quad.reshape(-1, 1, 2)], True, (255, 210, 120), 1, cv2.LINE_AA)
    for name, p in zip("ABCD", quad):
        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (255, 210, 120), -1, cv2.LINE_AA)
        cv2.putText(frame, name, (int(p[0]) + 5, int(p[1]) - 5), FONT, 0.4, (255, 210, 120), 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, lines: List[str]) -> None:
    pad, lh = 8, 17
    w = max(cv2.getTextSize(s, FONT, 0.42, 1)[0][0] for s in lines) + pad * 2
    h = lh * len(lines) + pad
    roi = frame[6:6 + h, 6:6 + w]
    if roi.size:
        frame[6:6 + h, 6:6 + w] = (roi * 0.35 + np.array(INK) * 0.65).astype(np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(frame, s, (6 + pad, 6 + pad + 9 + i * lh), FONT, 0.42,
                    AMBER if i == 0 else WHITE, 1, cv2.LINE_AA)


def heatmap_image(background: np.ndarray, energy: np.ndarray) -> np.ndarray:
    e = energy.astype(np.float32)
    if e.max() > 0:
        e = e / np.percentile(e[e > 0], 99) if np.any(e > 0) else e
    e = np.clip(e, 0, 1)
    heat = cv2.applyColorMap((e * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    a = (np.sqrt(e) * 0.75)[..., None]
    base = cv2.cvtColor(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return (base * (1 - a) + heat * a).astype(np.uint8)


def trajectories_image(background: np.ndarray, tracks: List[dict], scale_x: float, scale_y: float) -> np.ndarray:
    img = (background * 0.55).astype(np.uint8)
    for tr in tracks:
        pts = [(x / scale_x, y / scale_y) for x, y in tr["_trail_px"]]
        if len(pts) < 2:
            continue
        dx, dy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
        col = direction_color(dx, dy)
        arr = np.array(pts, np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [arr], False, col, 2, cv2.LINE_AA)
        cv2.circle(img, (int(pts[-1][0]), int(pts[-1][1])), 4, col, -1, cv2.LINE_AA)
        cv2.putText(img, f"#{tr['id']}", (int(pts[0][0]) + 4, int(pts[0][1])), FONT, 0.4, WHITE, 1, cv2.LINE_AA)
    return img
