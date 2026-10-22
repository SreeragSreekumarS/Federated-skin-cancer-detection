from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a project manifest from the ISIC 2019 training files"
    )
    parser.add_argument("--ground-truth", required=True, help="ISIC 2019 training ground-truth CSV")
    parser.add_argument("--images", required=True, help="Directory containing extracted training images")
    parser.add_argument("--metadata", help="Optional ISIC 2019 training metadata CSV")
    parser.add_argument("--output", required=True, help="Output manifest CSV")
    args = parser.parse_args()

    labels = pd.read_csv(args.ground_truth)
    if "image" not in labels.columns:
        raise ValueError("Ground-truth CSV must contain an 'image' column")
    class_columns = [column for column in labels.columns if column != "image"]
    if not class_columns:
        raise ValueError("Ground-truth CSV must contain one-hot diagnosis columns")

    labels[class_columns] = labels[class_columns].apply(pd.to_numeric, errors="raise")
    active = labels[class_columns].gt(0)
    invalid = active.sum(axis=1).ne(1)
    if invalid.any():
        raise ValueError(f"Expected exactly one positive diagnosis per row; found {int(invalid.sum())} invalid rows")
    labels["label"] = labels[class_columns].idxmax(axis=1).astype(str)
    labels["image"] = labels["image"].astype(str)

    image_dir = Path(args.images)
    paths = {
        path.stem: path.resolve()
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    }
    labels["image_path"] = labels["image"].map(lambda image_id: str(paths.get(image_id, "")))
    missing = labels["image_path"].eq("")
    if missing.any():
        examples = ", ".join(labels.loc[missing, "image"].head(5).tolist())
        raise FileNotFoundError(
            f"Could not find {int(missing.sum())} ground-truth images under {image_dir}; examples: {examples}"
        )

    if args.metadata:
        metadata = pd.read_csv(args.metadata)
        if "image" not in metadata.columns:
            raise ValueError("Metadata CSV must contain an 'image' column")
        metadata["image"] = metadata["image"].astype(str)
        if "lesion_id" in metadata.columns:
            metadata = metadata[["image", "lesion_id"]].drop_duplicates("image")
            labels = labels.merge(metadata, on="image", how="left", validate="one_to_one")
            labels["patient_id"] = labels["lesion_id"].fillna(labels["image"]).astype(str)
        else:
            labels["patient_id"] = labels["image"]
    else:
        labels["patient_id"] = labels["image"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels[["image_path", "label", "patient_id"]].to_csv(output, index=False)
    print(f"Wrote {len(labels)} rows across {labels['label'].nunique()} classes to {output}")


if __name__ == "__main__":
    main()
