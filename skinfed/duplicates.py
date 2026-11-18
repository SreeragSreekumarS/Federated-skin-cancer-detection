"""Perceptual-hash helpers for cross-dataset duplicate audits."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fftpack import dct


def perceptual_hash(path: str | Path, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Return a DCT perceptual hash as an integer (64 bits by default)."""
    size = hash_size * highfreq_factor
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("L").resize((size, size), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    transformed = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = transformed[:hash_size, :hash_size]
    median = float(np.median(low))
    result = 0
    for bit in (low > median).flat:
        result = (result << 1) | int(bit)
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class _ChunkIndex:
    """Exact chunk buckets with a guaranteed candidate for small Hamming radii."""
    def __init__(self, values: Iterable[int], radius: int, bit_count: int = 64):
        # With radius+1 disjoint chunks, any hash within `radius` bit flips has
        # at least one unchanged chunk. This avoids a quadratic all-pairs scan.
        self.chunk_count = min(bit_count, radius + 1)
        self.ranges = [
            (index * bit_count // self.chunk_count,
             (index + 1) * bit_count // self.chunk_count)
            for index in range(self.chunk_count)
        ]
        self.buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
        for value in values:
            for chunk_index, (start, end) in enumerate(self.ranges):
                width = end - start
                self.buckets[(chunk_index, (value >> start) & ((1 << width) - 1))].add(value)

    def query(self, value: int) -> set[int]:
        candidates: set[int] = set()
        for chunk_index, (start, end) in enumerate(self.ranges):
            width = end - start
            chunk = (value >> start) & ((1 << width) - 1)
            candidates.update(self.buckets.get((chunk_index, chunk), ()))
        return candidates


def find_cross_dataset_phash_matches(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    max_distance: int = 4,
    progress: Callable[[str], None] | None = print,
) -> pd.DataFrame:
    """Find candidate images perceptually close to any reference image.

    Frames need `image_path`; optional `image_id`, `source`, and `label` columns
    are copied into the returned pair report.
    """
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")
    reference_records: dict[int, list[int]] = defaultdict(list)
    for index, path in enumerate(reference.image_path.astype(str)):
        value = perceptual_hash(path)
        reference_records[value].append(index)
        if progress and (index + 1) % 1000 == 0:
            progress(f"pHash reference images: {index + 1}/{len(reference)}")

    hash_index = _ChunkIndex(reference_records, max_distance)

    rows: list[dict] = []
    for index, path in enumerate(candidates.image_path.astype(str)):
        candidate_hash = perceptual_hash(path)
        matching_hashes = [
            match_hash for match_hash in hash_index.query(candidate_hash)
            if hamming_distance(candidate_hash, match_hash) <= max_distance
        ]
        for match_hash in matching_hashes:
            distance = hamming_distance(candidate_hash, match_hash)
            for ref_index in reference_records[match_hash]:
                ref, candidate = reference.iloc[ref_index], candidates.iloc[index]
                row = {
                    "hamming_distance": distance,
                    "reference_image_id": ref.get("image_id", Path(str(ref.image_path)).stem),
                    "candidate_image_id": candidate.get("image_id", Path(str(candidate.image_path)).stem),
                    "reference_path": ref.image_path,
                    "candidate_path": candidate.image_path,
                    "reference_label": ref.get("label", ""),
                    "candidate_label": candidate.get("label", ""),
                    "reference_source": ref.get("source", ""),
                    "candidate_source": candidate.get("source", ""),
                    "reference_phash": f"{match_hash:016x}",
                    "candidate_phash": f"{candidate_hash:016x}",
                }
                rows.append(row)
        if progress and (index + 1) % 1000 == 0:
            progress(f"pHash candidate images: {index + 1}/{len(candidates)}")
    report = pd.DataFrame(rows, columns=[
        "hamming_distance", "reference_image_id", "candidate_image_id",
        "reference_path", "candidate_path", "reference_label", "candidate_label",
        "reference_source", "candidate_source", "reference_phash", "candidate_phash",
    ])
    if not report.empty:
        report = report.sort_values(["hamming_distance", "candidate_image_id"], ignore_index=True)
    return report
