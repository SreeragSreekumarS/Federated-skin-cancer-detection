"""Prepare leakage-filtered ISIC 2019 training and HAM10000 external-test manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from skinfed.duplicates import find_cross_dataset_phash_matches


ISIC_MERGED = {"AK": "akiec", "SCC": "akiec", "BCC": "bcc", "BKL": "bkl", "DF": "df", "MEL": "mel", "NV": "nv", "VASC": "vasc"}
ISIC_SCC_EXCLUDED = {"AK": "akiec", "BCC": "bcc", "BKL": "bkl", "DF": "df", "MEL": "mel", "NV": "nv", "VASC": "vasc"}
HAM_MAP = {"akiec": "akiec", "bcc": "bcc", "bkl": "bkl", "df": "df", "mel": "mel", "nv": "nv", "vasc": "vasc"}


def _isic_source(lesion_id: object) -> tuple[str, str]:
    lesion = "" if pd.isna(lesion_id) else str(lesion_id).strip()
    prefix = lesion.split("_", 1)[0] if lesion else ""
    source = {"HAM": "HAM", "BCN": "BCN", "MSK4": "MSK"}.get(prefix)
    if not lesion:
        return "MSK", ""
    if source is None:
        raise ValueError(f"Unrecognized ISIC 2019 lesion_id prefix in {lesion!r}")
    return source, lesion


def _read_isic(manifest_path: Path, metadata_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required_manifest = {"image_path", "label"}
    if not required_manifest.issubset(manifest.columns):
        raise ValueError(f"ISIC manifest must contain {sorted(required_manifest)}")
    manifest["image_path"] = manifest.image_path.astype(str)
    manifest["image_id"] = manifest.image_path.map(lambda value: Path(value).stem)
    metadata = pd.read_csv(metadata_path)
    if not {"image", "lesion_id"}.issubset(metadata.columns):
        raise ValueError("ISIC metadata must contain image and lesion_id columns")
    metadata["image_id"] = metadata.image.astype(str)
    if metadata.image_id.duplicated().any():
        raise ValueError("ISIC metadata has duplicate image identifiers")
    metadata["_metadata_match"] = True
    joined = manifest.merge(metadata, on="image_id", how="left", validate="one_to_one", suffixes=("", "_metadata"))
    if joined._metadata_match.isna().any():
        missing = joined.loc[joined._metadata_match.isna(), "image_id"].head(10).tolist()
        raise ValueError(f"ISIC metadata is missing {int(joined._metadata_match.isna().sum())} image IDs; examples: {missing}")
    sources, lesions = zip(*joined.lesion_id.map(_isic_source))
    joined["source"] = sources
    joined["lesion_id"] = lesions
    joined["patient_id"] = [lesion if lesion else image for lesion, image in zip(lesions, joined.image_id)]
    joined["label"] = joined.label.astype(str).str.upper()
    return joined


def _read_ham(metadata_path: Path, image_dir: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    required = {"image_id", "lesion_id", "dx"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"HAM10000 metadata must contain {sorted(required)}")
    image_paths = {
        path.stem: str(path.resolve())
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    ham = metadata.copy()
    ham["image_id"] = ham.image_id.astype(str)
    ham["image_path"] = ham.image_id.map(image_paths)
    missing = ham.image_path.isna()
    if missing.any():
        raise FileNotFoundError(f"{int(missing.sum())} HAM images missing; first: {ham.loc[missing, 'image_id'].iloc[0]}")
    ham["label"] = ham.dx.astype(str).str.lower().map(HAM_MAP)
    if ham.label.isna().any():
        raise ValueError("HAM metadata contains diagnosis labels outside the 7 standard classes")
    ham["lesion_id"] = ham.lesion_id.fillna("").astype(str)
    ham["patient_id"] = ham.lesion_id.where(ham.lesion_id.ne(""), ham.image_id)
    ham["source"] = "HAM"
    ham["age"] = pd.to_numeric(ham.get("age", pd.Series(index=ham.index, dtype=float)), errors="coerce")
    ham["sex"] = ham.get("sex", pd.Series("unknown", index=ham.index)).fillna("unknown")
    ham["site"] = ham.get("localization", pd.Series("unknown", index=ham.index)).fillna("unknown")
    return ham


def _write_manifest(frame: pd.DataFrame, path: Path) -> None:
    columns = ["image_path", "label", "patient_id", "image_id", "lesion_id", "source", "age", "sex", "site"]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[columns].to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove HAM10000 overlap from ISIC 2019 and create two harmonized label variants"
    )
    parser.add_argument("--isic-manifest", required=True)
    parser.add_argument("--isic-metadata", required=True)
    parser.add_argument("--ham-metadata", required=True)
    parser.add_argument("--ham-images", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phash-distance", type=int, default=4,
                        help="Exclude ISIC training images within this pHash Hamming distance from a HAM image")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    isic = _read_isic(Path(args.isic_manifest), Path(args.isic_metadata))
    ham = _read_ham(Path(args.ham_metadata), Path(args.ham_images))

    source_counts = isic.source.value_counts().sort_index()
    print("Verified ISIC 2019 source counts (lesion_id prefix / missing-ID rule):")
    print(source_counts.to_string())
    if set(source_counts.index) - {"HAM", "BCN", "MSK"}:
        raise ValueError(f"Unexpected ISIC source categories: {set(source_counts.index)}")
    if source_counts.get("HAM", 0) != len(ham):
        raise ValueError(
            f"Expected {len(ham)} HAM-prefixed ISIC rows, found {source_counts.get('HAM', 0)}; inspect source metadata"
        )

    ham_ids = set(ham.image_id)
    exact_overlap = isic.image_id.isin(ham_ids)
    if int(exact_overlap.sum()) != len(ham_ids):
        missing_ids = ham_ids - set(isic.loc[exact_overlap, "image_id"])
        print(f"Warning: {len(missing_ids)} HAM image IDs were not found in the ISIC manifest")
    print(f"ISIC image-ID overlap with HAM metadata: {int(exact_overlap.sum())}")
    isic_non_ham = isic.loc[~exact_overlap & isic.source.ne("HAM")].copy()

    phash_matches = find_cross_dataset_phash_matches(ham, isic_non_ham, args.phash_distance)
    phash_report = output_dir / "cross_dataset_phash_duplicates.csv"
    phash_matches.to_csv(phash_report, index=False)
    phash_candidate_ids = set(phash_matches.candidate_image_id.astype(str)) if not phash_matches.empty else set()
    isic_train_base = isic_non_ham.loc[~isic_non_ham.image_id.astype(str).isin(phash_candidate_ids)].copy()
    print(f"ISIC candidate images excluded by pHash (distance <= {args.phash_distance}): {len(phash_candidate_ids)}")
    print(f"Cross-dataset pHash pair report: {phash_report.resolve()}")

    isic_train_base["age"] = pd.to_numeric(isic_train_base.get("age", pd.Series(index=isic_train_base.index, dtype=float)), errors="coerce")
    isic_train_base["sex"] = isic_train_base.get("sex", pd.Series("unknown", index=isic_train_base.index)).fillna("unknown")
    isic_train_base["site"] = isic_train_base.get("site", pd.Series("unknown", index=isic_train_base.index)).fillna("unknown")
    outputs = {}
    for variant, mapping in (("merged", ISIC_MERGED), ("scc_excluded", ISIC_SCC_EXCLUDED)):
        train = isic_train_base.copy()
        if variant == "scc_excluded":
            train = train.loc[train.label.ne("SCC")].copy()
        unknown_labels = set(train.label) - set(mapping)
        if unknown_labels:
            raise ValueError(f"Unexpected ISIC labels in {variant} mode: {sorted(unknown_labels)}")
        train["label"] = train.label.map(mapping)
        train_path = output_dir / f"isic2019_train_{variant}.csv"
        ham_path = output_dir / f"ham10000_external_{variant}.csv"
        _write_manifest(train, train_path)
        _write_manifest(ham, ham_path)
        outputs[variant] = {
            "isic_train_manifest": str(train_path.resolve()),
            "ham_external_test_manifest": str(ham_path.resolve()),
            "isic_train_rows": int(len(train)),
            "isic_train_class_counts": {str(k): int(v) for k, v in train.label.value_counts().sort_index().items()},
        }

    audit = pd.DataFrame([
        {"source": source, "rows_before_filter": int(count),
         "lesion_groups_before_filter": int(isic.loc[isic.source.eq(source), "patient_id"].nunique())}
        for source, count in source_counts.items()
    ])
    audit.to_csv(output_dir / "isic_source_audit.csv", index=False)
    summary = {
        "ham_rows": int(len(ham)),
        "isic_rows_before_filter": int(len(isic)),
        "exact_ham_image_id_overlap_removed": int(exact_overlap.sum()),
        "isic_ham_source_rows_excluded": int(isic.source.eq("HAM").sum()),
        "phash_distance_threshold": int(args.phash_distance),
        "phash_matched_isic_images_removed": int(len(phash_candidate_ids)),
        "phash_pairs_reported": int(len(phash_matches)),
        "source_counts_before_filter": {str(k): int(v) for k, v in source_counts.items()},
        "variants": outputs,
    }
    (output_dir / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
