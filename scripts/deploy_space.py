"""Publish MotionScope as a Hugging Face Space (Docker SDK).

    hf auth login                                   # once
    python scripts/deploy_space.py <hf-user>/MotionScope

Creates the Space if needed and uploads only what the container needs:
the Dockerfile, the application code, the frontend, the demo clips and the
Space README from deploy/huggingface/. Re-running updates the Space.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INCLUDE = [
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
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is required: pip install -r requirements-dev.txt", file=sys.stderr)
        return 1

    api = HfApi()
    api.create_repo(args.space_id, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.space_id, repo_type="space", folder_path=str(ROOT),
        allow_patterns=INCLUDE, ignore_patterns=EXCLUDE,
        commit_message="Deploy MotionScope",
    )
    api.upload_file(
        repo_id=args.space_id, repo_type="space",
        path_or_fileobj=str(ROOT / "deploy/huggingface/README.md"), path_in_repo="README.md",
        commit_message="Space card",
    )
    print(f"https://huggingface.co/spaces/{args.space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
