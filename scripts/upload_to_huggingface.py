#!/usr/bin/env python3
"""
One-time upload of the Elliptic Bitcoin CSVs to Hugging Face Datasets.

Prerequisites:
  1. pip install huggingface_hub
  2. huggingface-cli login   (or set HF_TOKEN)

Usage:
  python scripts/upload_to_huggingface.py
  python scripts/upload_to_huggingface.py --repo-id YOUR_USERNAME/elliptic-bitcoin
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "RIXESTO/elliptic-bitcoin"
RAW_DIR = ROOT / "data" / "raw"
FILES = [
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_features.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Elliptic dataset to Hugging Face")
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo (default: {DEFAULT_REPO_ID})",
    )
    args = parser.parse_args()

    missing = [name for name in FILES if not (RAW_DIR / name).exists()]
    if missing:
        raise SystemExit(
            "Missing local files in data/raw/:\n  "
            + "\n  ".join(missing)
            + "\n\nDownload from Kaggle first:\n"
            "  https://www.kaggle.com/datasets/ellipticco/elliptic-data-set"
        )

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    for filename in FILES:
        path = RAW_DIR / filename
        print(f"Uploading {filename} ({path.stat().st_size / 1e6:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=filename,
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    print(f"\nDone. Dataset available at: https://huggingface.co/datasets/{args.repo_id}")
    print("Update configs/config.yaml if you used a custom --repo-id.")


if __name__ == "__main__":
    main()
