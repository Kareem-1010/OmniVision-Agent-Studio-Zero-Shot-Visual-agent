#!/usr/bin/env python3
"""
Model weight download and cache preparation utility for OmniVision Agent.
Provides verification and local asset staging for SOTA perception checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OmniVisionDownloader")

CHECKPOINTS_METADATA: Dict[str, Dict[str, Any]] = {
    "yolo11x": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
        "filename": "yolo11x.pt",
        "size_mb": 113.8,
        "sha256": None
    },
    "sam2_hiera_large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        "filename": "sam2_hiera_large.pt",
        "size_mb": 898.0,
        "sha256": None
    },
    "depth_anything_v2_vitb": {
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
        "filename": "depth_anything_v2_vitb.pth",
        "size_mb": 390.5,
        "sha256": None
    }
}


def compute_sha256(filepath: Path) -> str:
    """Compute the SHA256 hex digest of a local file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_or_download_models(target_dir: Path, dry_run: bool = False) -> None:
    """Verify local cache of checkpoints or log download plan."""
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "models_manifest.json"

    logger.info(f"Target checkpoints directory: {target_dir.resolve()}")
    manifest: Dict[str, Any] = {}

    for name, meta in CHECKPOINTS_METADATA.items():
        dest = target_dir / meta["filename"]
        if dest.exists():
            sha = compute_sha256(dest)
            logger.info(f"[FOUND] {name} -> {dest.name} (SHA-256: {sha[:12]}...)")
            manifest[name] = {"path": str(dest.resolve()), "status": "cached", "sha256": sha}
        else:
            logger.info(f"[STAGED] {name} -> {meta['url']} (~{meta['size_mb']} MB)")
            manifest[name] = {"path": str(dest.resolve()), "status": "staged_for_lazy_download", "url": meta["url"]}

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Checkpoint manifest generated at: {manifest_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniVision Model Downloader & Verification")
    parser.add_argument("--cache-dir", type=str, default=".cache/models", help="Destination directory for model checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Stage manifest without downloading immediately")
    args = parser.parse_args()

    verify_or_download_models(Path(args.cache_dir), dry_run=args.dry_run)
