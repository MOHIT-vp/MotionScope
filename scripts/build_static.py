"""Build the static showcase: the landing page and demos, without the backend.

    python scripts/build_static.py [output_dir]      # default: dist/static

The page runs in its static mode (no uploads; the upload area links to the
source). Used for hosts that serve files only, such as a static HF Space.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("app.js", "styles.css"):
        shutil.copy2(ROOT / "frontend" / name, out / name)
    html = (ROOT / "frontend/index.html").read_text()
    marker = '<html lang="en">'
    if marker not in html:
        raise SystemExit("frontend/index.html: <html lang=\"en\"> not found")
    (out / "index.html").write_text(html.replace(marker, '<html lang="en" data-mode="static">', 1))
    shutil.copytree(ROOT / "demos", out / "demos",
                    ignore=shutil.ignore_patterns(".DS_Store", "__pycache__"))
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / "static"
    print(build(target))
