"""Download the optional YOLOX-S vehicle detector (OpenCV model zoo, Apache-2.0).

    python scripts/fetch_model.py

Without this file the pipeline still works and falls back to class-agnostic
motion-blob detection (background subtraction + motion saliency).
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/opencv/object_detection_yolox/resolve/main/"
       "object_detection_yolox_2022nov.onnx")
SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
DEST = Path(__file__).resolve().parent.parent / "models" / "object_detection_yolox_2022nov.onnx"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if DEST.is_file() and sha256(DEST) == SHA256:
        print(f"model already present: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, tmp)
    if sha256(tmp) != SHA256:
        tmp.unlink()
        print("checksum mismatch; download discarded", file=sys.stderr)
        return 1
    tmp.replace(DEST)
    print(f"saved {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
