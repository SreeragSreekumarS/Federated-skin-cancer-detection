"""Audit a skin-lesion manifest before federated splitting or dataset merging.

The large-dataset reference paper highlights duplicate images as a source of
optimistic evaluation. This utility finds exact visual duplicates and reports
whether their labels or patient identifiers disagree.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from skinfed.data import read_manifest


def pixel_hash(image_path: str) -> str:
    """Hash decoded RGB pixels, so a copied image in another file format matches."""
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{rgb.width}x{rgb.height}".encode("ascii"))
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Find exact visual duplicates in a lesion manifest")
    parser.add_argument("--manifest", required=True, help="CSV with image_path,label and optional patient_id")
    parser.add_argument("--output", help="Where to write duplicate groups as CSV")
    args = parser.parse_args()
    frame, _ = read_manifest(args.manifest)
    frame["pixel_sha256"] = [pixel_hash(path) for path in frame.image_path]
    duplicates = frame[frame.duplicated("pixel_sha256", keep=False)].copy()
    duplicates["label_conflict"] = duplicates.groupby("pixel_sha256").label.transform("nunique").gt(1)
    duplicates["patient_conflict"] = (
        duplicates["patient_id"].astype(str).groupby(duplicates["pixel_sha256"]).transform("nunique").gt(1)
    )
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        duplicates.sort_values(["pixel_sha256", "image_path"]).to_csv(destination, index=False)
        print(f"Duplicate report: {destination.resolve()}")
    groups = duplicates.pixel_sha256.nunique()
    print(f"Images checked: {len(frame)}")
    print(f"Duplicate groups: {groups}; duplicate rows: {len(duplicates)}")
    print(f"Groups with label conflicts: {duplicates.loc[duplicates.label_conflict, 'pixel_sha256'].nunique()}")
    print(f"Groups spanning patient IDs: {duplicates.loc[duplicates.patient_conflict, 'pixel_sha256'].nunique()}")
    return 1 if len(duplicates) else 0


if __name__ == "__main__":
    raise SystemExit(main())
