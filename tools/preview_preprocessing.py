"""Create a contact sheet showing preprocessing effects on random manifest images."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_paths(manifest: Path) -> pd.DataFrame:
    frame = pd.read_csv(manifest)
    if not {"image_path", "label"}.issubset(frame.columns):
        raise ValueError("Manifest must contain image_path and label columns")
    root = manifest.resolve().parent
    frame["image_path"] = frame.image_path.map(
        lambda value: str((root / str(value)).resolve())
        if not Path(str(value)).is_absolute() else str(Path(str(value)))
    )
    missing = [path for path in frame.image_path if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest images are missing; first: {missing[0]}")
    return frame


def _apply_hair_removal(image: Image.Image) -> Image.Image:
    # Reuse the exact DullRazor-style implementation used by training.
    from skinfed.data import RandomHairRemoval
    return RandomHairRemoval(probability=1.0)(image)


def _apply_color_constancy(image: Image.Image) -> Image.Image:
    from skinfed.data import GrayWorldColorConstancy
    return GrayWorldColorConstancy()(image)


def _card(image: Image.Image | None, title: str, detail: str, size: int = 256) -> Image.Image:
    card = Image.new("RGB", (size, size + 64), "white")
    draw = ImageDraw.Draw(card)
    draw.text((8, 5), title, fill="black")
    draw.text((8, 23), detail[:40], fill=(80, 80, 80))
    if image is not None:
        image = ImageOps.contain(image.convert("RGB"), (size - 16, size - 16))
        card.paste(image, ((size - image.width) // 2, 52 + (size - 16 - image.height) // 2))
    else:
        draw.text((20, 150), "Unavailable in this environment", fill=(150, 0, 0))
    return card


def _preview_variants(image: Image.Image, config: dict) -> list[tuple[str, str, Image.Image | None]]:
    configured_enhancement = bool(config.get("image_enhancement", False))
    hair_probability = float(config.get("hair_removal_probability", 0.0))
    configured_color = bool(config.get("color_constancy", False))
    variants: list[tuple[str, str, Image.Image | None]] = [("Original", "input image", image)]

    try:
        hair = _apply_hair_removal(image)
        hair_detail = f"train p={hair_probability:g}" if hair_probability else "preview only; off"
    except (ImportError, RuntimeError):
        hair, hair_detail = None, "OpenCV needed for hair removal"
    variants.append(("Hair removal", hair_detail, hair))

    contrast_detail = "train p=0.10" if configured_enhancement else "preview only; off"
    sharpness_detail = "train p=0.10" if configured_enhancement else "preview only; off"
    equalize_detail = "train p=0.05" if configured_enhancement else "preview only; off"
    contrast = ImageOps.autocontrast(image)
    sharpness = ImageEnhance.Sharpness(image).enhance(1.25)
    equalized = ImageOps.equalize(image)
    variants.extend([
        ("Auto-contrast", contrast_detail, contrast),
        ("Sharpness 1.25x", sharpness_detail, sharpness),
        ("Equalization", equalize_detail, equalized),
    ])

    if configured_color:
        try:
            color_corrected = _apply_color_constancy(image)
            color_detail = "enabled in config"
        except ImportError:
            color_corrected, color_detail = None, "unavailable"
    else:
        color_corrected, color_detail = _apply_color_constancy(image), "preview only; off"
    variants.append(("Color constancy", color_detail, color_corrected))

    # Diagnostic view only: force each optional operation once so the visual
    # effect is easy to inspect. Training applies them stochastically.
    combined = color_corrected.copy() if color_corrected is not None else image.copy()
    if hair is not None:
        combined = hair
    combined = ImageOps.autocontrast(combined)
    combined = ImageEnhance.Sharpness(combined).enhance(1.25)
    combined = ImageOps.equalize(combined)
    variants.append(("Combined (forced)", "diagnostic preview", combined))
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/preprocessing_preview")
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    manifest_path = Path(args.manifest)
    frame = _resolve_paths(manifest_path)
    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(len(frame), size=min(args.samples, len(frame)), replace=False)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cards = []
    for number, row_index in enumerate(selected, 1):
        row = frame.iloc[int(row_index)]
        with Image.open(row.image_path) as opened:
            original = opened.convert("RGB")
        variants = _preview_variants(original, config)
        cards = [
            _card(image, title, detail)
            for title, detail, image in variants
        ]
        row_sheet = Image.new("RGB", (sum(card.width for card in cards), cards[0].height), "white")
        for card_index, card in enumerate(cards):
            row_sheet.paste(card, (card_index * card.width, 0))
        image_id = str(row.get("image_id", Path(row.image_path).stem))
        row_path = output_dir / f"{number:02d}_{image_id}_preprocessing.png"
        row_sheet.save(row_path)
        all_cards.append(row_sheet)
        print(f"Saved {row_path.resolve()}")

    sheet = Image.new("RGB", (all_cards[0].width, sum(row.height for row in all_cards)), "white")
    y_offset = 0
    for row_sheet in all_cards:
        sheet.paste(row_sheet, (0, y_offset))
        y_offset += row_sheet.height
    contact_path = output_dir / "preprocessing_contact_sheet.png"
    sheet.save(contact_path)
    print(f"Contact sheet: {contact_path.resolve()}")
    print("Each optional transform is forced in its own preview panel for visibility; training uses configured probabilities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
