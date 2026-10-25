"""Publish MotionScope as a Hugging Face Space.

    hf auth login                                            # once
    python scripts/deploy_space.py <hf-user>/MotionScope --static   # free static showcase
    python scripts/deploy_space.py <hf-user>/MotionScope            # Docker app (needs HF PRO)

--static uploads the landing page and demos built by scripts/build_static.py.
Without it, the Docker app is uploaded: the Dockerfile, the application code,
the frontend and the demo clips. Both use the Space card in deploy/huggingface/.
Re-running updates the Space.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_static import build  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

DOCKER_INCLUDE = [
    "Dockerfile",
    ".dockerignore",
    "requirements.txt",
    "backend/**",
    "frontend/**",
    "motiontrack/**",
    "demos/**",
    "samples/**",
    "scripts/fetch_model.py",
]
EXCLUDE = ["**/__pycache__/**", "**/*.pyc", "**/.DS_Store"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("space_id", help="Space id, e.g. your-hf-username/MotionScope")
    ap.add_argument("--static", action="store_true", help="publish the static showcase (free tier)")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is required: pip install -r requirements-dev.txt", file=sys.stderr)
        return 1

    api = HfApi()
    api.create_repo(args.space_id, repo_type="space", space_sdk="static" if args.static else "docker",
                    private=args.private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        if args.static:
            folder = build(Path(tmp) / "site")
            include, card = None, "README-static.md"
        else:
            folder, include, card = ROOT, DOCKER_INCLUDE, "README.md"
        api.upload_folder(repo_id=args.space_id, repo_type="space", folder_path=str(folder),
                          allow_patterns=include, ignore_patterns=EXCLUDE,
                          commit_message="Deploy MotionScope")
    api.upload_file(repo_id=args.space_id, repo_type="space",
                    path_or_fileobj=str(ROOT / "deploy/huggingface" / card), path_in_repo="README.md",
                    commit_message="Space card")
    print(f"https://huggingface.co/spaces/{args.space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
