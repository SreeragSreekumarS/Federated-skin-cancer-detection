"""Audit perceptual-hash matches between two image manifests."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from skinfed.data import read_manifest
from skinfed.duplicates import find_cross_dataset_phash_matches


def main() -> int:
    parser = argparse.ArgumentParser(description="Find near-duplicate images across two datasets using pHash")
    parser.add_argument("--reference-manifest", required=True, help="Reference image manifest, e.g. HAM10000")
    parser.add_argument("--candidate-manifest", required=True, help="Candidate images to audit, e.g. ISIC 2019")
    parser.add_argument("--output", required=True, help="CSV report of pHash matches")
    parser.add_argument("--max-hamming-distance", type=int, default=4)
    args = parser.parse_args()
    reference, _ = read_manifest(args.reference_manifest)
    candidates, _ = read_manifest(args.candidate_manifest)
    matches = find_cross_dataset_phash_matches(reference, candidates, args.max_hamming_distance)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output, index=False)
    duplicate_candidates = matches.candidate_image_id.nunique() if not matches.empty else 0
    print(f"Reference images: {len(reference)}")
    print(f"Candidate images: {len(candidates)}")
    print(f"Cross-dataset pHash pairs within distance {args.max_hamming_distance}: {len(matches)}")
    print(f"Candidate images with at least one match: {duplicate_candidates}")
    print(f"Report written to {output.resolve()}")
    return 1 if duplicate_candidates else 0


if __name__ == "__main__":
    raise SystemExit(main())
