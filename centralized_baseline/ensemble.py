"""Evaluate an ensemble of centralized checkpoints on their shared held-out split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from skinfed.calibration import fit_class_thresholds, fit_temperature, threshold_predictions
from skinfed.config import load_config
from skinfed.data import cache_resized_images, read_manifest
from skinfed.federated import predict_logits
from skinfed.metrics import classification_metrics
from skinfed.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate and evaluate a centralized checkpoint ensemble")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="Repeat for each run's best_model.pt (3–5 recommended)")
    parser.add_argument("--output", default="runs/centralized_ensemble")
    args = parser.parse_args()
    if len(args.checkpoint) < 2:
        parser.error("provide at least two --checkpoint paths")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame, class_names = read_manifest(args.manifest)
    models, configs = [], []
    reference_assignments = None
    split_data = {}
    for checkpoint_path in args.checkpoint:
        run_dir = Path(checkpoint_path).resolve().parent
        config = load_config(run_dir / "config.yaml")
        if config.cache_resized_images:
            cache = Path(args.manifest).resolve().parent / f".resized_cache_{config.image_size}"
            run_frame = cache_resized_images(frame.copy(), config.image_size, cache)
        else:
            run_frame = frame
        assignments = pd.read_csv(run_dir / "split_assignments.csv").sort_values("manifest_index")[
            ["manifest_index", "split", "label", "patient_id"]
        ].reset_index(drop=True)
        if reference_assignments is None:
            reference_assignments = assignments
            split_data = {name: assignments.loc[assignments.split == name, "manifest_index"].to_numpy(dtype=int)
                          for name in ("validation", "test")}
        elif not reference_assignments.equals(assignments):
            raise ValueError("All checkpoints must use identical manifest rows, labels, groups, and split assignments")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["class_names"] != class_names:
            raise ValueError(f"Class order differs in {checkpoint_path}")
        model = build_model(
            len(class_names), pretrained=False, use_cbam=config.use_cbam,
            classifier_hidden_dim=config.classifier_hidden_dim,
            classifier_dropout=config.classifier_dropout,
            drop_path_rate=config.drop_path_rate, model_name=checkpoint.get("model_name", config.model_name),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append((model, run_frame, config))
        configs.append(config)

    val_logits, test_logits = [], []
    val_targets = test_targets = None
    for model, run_frame, config in models:
        val, y_val = predict_logits(model, run_frame, split_data["validation"], config, device)
        test, y_test = predict_logits(model, run_frame, split_data["test"], config, device)
        if val_targets is not None and (not np.array_equal(val_targets, y_val) or not np.array_equal(test_targets, y_test)):
            raise ValueError("Target order differs between ensemble members")
        val_targets, test_targets = y_val, y_test
        val_logits.append(val)
        test_logits.append(test)

    mean_val_logits = np.mean(val_logits, axis=0)
    mean_test_logits = np.mean(test_logits, axis=0)
    temperature = fit_temperature(mean_val_logits, val_targets)
    val_probs = torch.softmax(torch.from_numpy(mean_val_logits) / temperature, dim=1).numpy()
    thresholds = fit_class_thresholds(val_probs, val_targets, class_names)
    probabilities = torch.softmax(torch.from_numpy(mean_test_logits) / temperature, dim=1).numpy()
    predicted_labels = threshold_predictions(probabilities, thresholds, class_names)
    metrics, matrix = classification_metrics(test_targets, probabilities, class_names, predicted_labels)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration.json").write_text(json.dumps({"temperature": temperature, "class_thresholds": thresholds}, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps({"members": [str(Path(p).resolve()) for p in args.checkpoint], "tta_views": [c.tta_views if c.tta else 1 for c in configs], "calibration": {"temperature": temperature, "class_thresholds": thresholds}, "test": metrics}, indent=2), encoding="utf-8")
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(output / "confusion_matrix.csv")
    np.savez_compressed(output / "test_predictions.npz", class_names=np.asarray(class_names), targets=test_targets,
                        probabilities=probabilities, predicted_labels=predicted_labels)
    print(f"Ensemble results saved to: {output}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
