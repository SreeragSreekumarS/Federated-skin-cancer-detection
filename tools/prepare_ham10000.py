from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Create a safe HAM10000 image manifest")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    meta, image_dir = pd.read_csv(args.metadata), Path(args.images)
    required = {"image_id", "dx", "lesion_id", "age", "sex", "localization"}
    if not required.issubset(meta): raise ValueError(f"metadata needs {required}")
    # Store absolute paths: the manifest reader resolves relative entries from
    # the manifest's directory, which may differ from the project root.
    paths = {p.stem: p.resolve() for p in image_dir.rglob("*.jpg")}
    meta["image_path"] = meta.image_id.map(lambda image_id: str(paths.get(str(image_id), "")))
    meta = meta[meta.image_path.ne("")].copy()
    meta["label"] = meta.dx.astype(str)
    meta["patient_id"] = meta["lesion_id"].fillna(meta.image_id).astype(str)
    # Deliberately exclude dx_type: it encodes how the diagnosis was established.
    meta["age"] = pd.to_numeric(meta["age"], errors="coerce")
    meta["sex"] = meta["sex"].fillna("unknown")
    meta["site"] = meta["localization"].fillna("unknown")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    meta[["image_path", "label", "patient_id", "age", "sex", "site"]].to_csv(output, index=False)
    print(f"Wrote {len(meta)} rows to {output}")


if __name__ == "__main__": main()
