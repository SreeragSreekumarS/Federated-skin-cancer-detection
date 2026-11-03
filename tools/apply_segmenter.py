from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from tools.train_segmenter import UNet


def main():
    parser = argparse.ArgumentParser(description="Use a trained segmenter to create lesion-focused classifier images")
    parser.add_argument("--manifest", required=True, help="Existing HAM10000 or ISIC 2019 classification manifest")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint made by train_segmenter.py")
    parser.add_argument("--output-dir", required=True, help="Directory for lesion-focused JPGs")
    parser.add_argument("--output-manifest", required=True, help="New manifest; original manifest is left untouched")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--padding", type=float, default=0.12, help="Fractional padding around predicted lesion bounding box")
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or not 0 <= args.padding <= 1:
        parser.error("threshold must be in (0,1), and padding must be in [0,1]")
    source_manifest = Path(args.manifest).resolve()
    frame = pd.read_csv(source_manifest)
    if not {"image_path", "label"}.issubset(frame.columns):
        raise ValueError("Manifest must contain image_path and label columns")
    frame["image_path"] = frame["image_path"].map(
        lambda value: str((source_manifest.parent / value).resolve()) if not Path(str(value)).is_absolute() else str(Path(str(value)).resolve())
    )
    missing = [p for p in frame.image_path if not Path(p).is_file()]
    if missing: raise FileNotFoundError(f"{len(missing)} manifest images are missing; first: {missing[0]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    size = int(checkpoint["size"])
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    with torch.inference_mode():
        for i, path in enumerate(frame.image_path):
            image_path = Path(path)
            with Image.open(image_path) as im: image = im.convert("RGB")
            w, h = image.size
            x = TF.resize(image, [size, size], interpolation=TF.InterpolationMode.BILINEAR)
            x = TF.normalize(TF.to_tensor(x), mean, std).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                probability = torch.sigmoid(model(x))[0, 0]
            mask = Image.fromarray((probability.float().cpu().numpy() >= args.threshold).astype("uint8") * 255)
            mask = mask.resize((w, h), Image.Resampling.NEAREST)
            bbox = mask.getbbox()
            if bbox:
                left, top, right, bottom = bbox
                pad_x = max(1, round((right - left) * args.padding)); pad_y = max(1, round((bottom - top) * args.padding))
                bbox = (max(0, left - pad_x), max(0, top - pad_y), min(w, right + pad_x), min(h, bottom + pad_y))
                focused = image.crop(bbox)
            else:
                # Avoid producing blank or misleading crops when inference finds no lesion.
                focused = image
            destination = output_dir / f"{i:06d}_{image_path.stem}.jpg"
            focused.save(destination, quality=95)
            output_paths.append(str(destination))
            if (i + 1) % 500 == 0 or i + 1 == len(frame): print(f"Processed {i + 1}/{len(frame)} images")
    frame["image_path"] = output_paths
    manifest_out = Path(args.output_manifest).resolve(); manifest_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest_out, index=False)
    print(f"Wrote {len(frame)} lesion-focused images to {output_dir}")
    print(f"Wrote new manifest to {manifest_out}")


if __name__ == "__main__": main()
