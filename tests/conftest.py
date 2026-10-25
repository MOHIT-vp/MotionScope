"""Synthetic videos with known ground truth.

A textured block moves across a static textured background at a fixed
velocity.  Because the scene is generated, the true pixel velocity,
direction and (for a top-down camera) metric speed are known exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from motiontrack.video_io import H264Writer  # noqa: E402

W, H, FPS = 640, 480, 25.0
VX, VY = 4.0, 0.0          # px / frame  → 100 px/s to the right
N_FRAMES = 70


def make_video(path: Path, n=N_FRAMES, vx=VX, vy=VY, shake=0.0, seed=0) -> Path:
    rng = np.random.default_rng(seed)
    big = np.zeros((H + 40, W + 40, 3), np.uint8)
    # low-contrast textured background so the camera-motion estimator has corners
    base = rng.integers(70, 110, size=((H + 40) // 8 + 1, (W + 40) // 8 + 1, 1), dtype=np.uint8)
    big[:] = np.kron(base, np.ones((8, 8, 3), np.uint8))[: H + 40, : W + 40]
    obj = rng.integers(0, 255, size=(50 // 5, 80 // 5, 1), dtype=np.uint8)
    obj = np.kron(obj, np.ones((5, 5, 3), np.uint8))
    obj[..., 2] = 230  # strong colour so it differs from the background
    writer = H264Writer(path, W, H, FPS, crf=12)
    for i in range(n):
        frame = big.copy()
        x, y = int(round(40 + 20 + vx * i)), int(round(200 + 20 + vy * i))
        frame[y: y + 50, x: x + 80] = obj
        dx = dy = 20
        if shake:
            dx += int(round(rng.uniform(-shake, shake)))
            dy += int(round(rng.uniform(-shake, shake)))
        writer.write(np.ascontiguousarray(frame[dy: dy + H, dx: dx + W]))
    writer.close()
    return path


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    return make_video(tmp_path_factory.mktemp("vid") / "moving_block.mp4")


@pytest.fixture(scope="session")
def shaky_video(tmp_path_factory) -> Path:
    return make_video(tmp_path_factory.mktemp("vid") / "shaky_block.mp4", shake=3.0, seed=1)
