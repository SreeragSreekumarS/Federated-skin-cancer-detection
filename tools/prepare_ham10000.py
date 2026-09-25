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
    required = {"image_id", "dx"}
    if not required.issubset(meta): raise ValueError(f"metadata needs {required}")
    paths = {p.stem: p for p in image_dir.rglob("*.jpg")}
    meta["image_path"] = meta.image_id.map(lambda image_id: str(paths.get(str(image_id), "")))
    meta = meta[meta.image_path.ne("")].copy()
    meta["label"] = meta.dx.astype(str)
    meta["patient_id"] = meta.get("lesion_id", meta.image_id).fillna(meta.image_id).astype(str)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    meta[["image_path", "label", "patient_id"]].to_csv(output, index=False)
    print(f"Wrote {len(meta)} rows to {output}")


if __name__ == "__main__": main()
