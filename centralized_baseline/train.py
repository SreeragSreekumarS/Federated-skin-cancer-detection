"""Train the same classifier centrally on the exact federated training split."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from skinfed.config import load_config, save_config
from skinfed.data import make_federated_splits, read_manifest
from skinfed.federated import FocalCrossEntropy, evaluate, make_loader
from skinfed.model import build_model
from skinfed.plots import save_result_plots


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a centralized baseline using the federated model and split"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="config/gtx1650.yaml")
    parser.add_argument("--output", default="runs/centralized")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is unavailable; this run will be slow on CPU.")
    else:
        print(f"Using {torch.cuda.get_device_name(0)}")

    frame, class_names = read_manifest(args.manifest)
    client_train, client_val, client_test, val_indices, test_indices = make_federated_splits(
        frame, config.clients, config.dirichlet_alpha,
        config.train_fraction, config.val_fraction, config.seed,
    )
    # These indices are exactly those used by train.py. Pooling happens only
    # after the fixed patient-to-hospital assignments and within-hospital splits.
    train_indices = np.concatenate(client_train)
    output = Path(args.output) / config.run_name
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.yaml")
    (output / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    assignments = []
    for hospital, split_lists in enumerate(zip(client_train, client_val, client_test), 1):
        for split_name, indices in zip(("train", "validation", "test"), split_lists):
            for index in indices:
                row = frame.iloc[int(index)]
                assignments.append({
                    "manifest_index": int(index),
                    "image_path": row.image_path,
                    "label": row.label,
                    "patient_id": row.patient_id,
                    "hospital": hospital,
                    "split": split_name,
                })
    pd.DataFrame(assignments).sort_values("manifest_index").to_csv(
        output / "split_assignments.csv", index=False
    )
    pd.DataFrame({
        "split": ["train", "validation", "test"],
        "images": [len(train_indices), len(val_indices), len(test_indices)],
        "patients_or_lesions": [
            frame.iloc[idx].patient_id.astype(str).nunique()
            for idx in (train_indices, val_indices, test_indices)
        ],
    }).to_csv(output / "split_summary.csv", index=False)

    counts = frame.iloc[train_indices].target.value_counts().reindex(
        range(len(class_names)), fill_value=1
    ).to_numpy()
    class_weights = torch.tensor(
        counts.sum() / (len(counts) * counts), dtype=torch.float32
    )
    loss_weights = torch.ones_like(class_weights) if config.balanced_sampling else class_weights
    criterion = FocalCrossEntropy(
        loss_weights.to(device), config.focal_gamma, config.label_smoothing
    )
    model = build_model(
        len(class_names), pretrained=not args.no_pretrained,
        use_cbam=config.use_cbam,
        classifier_hidden_dim=config.classifier_hidden_dim,
        classifier_dropout=config.classifier_dropout,
    ).to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad = config.freeze_backbone_rounds == 0
    scaler = torch.amp.GradScaler(
        device.type, enabled=config.amp and device.type == "cuda"
    )

    best_f1, stale, lr_stale = -1.0, 0, 0
    current_lr = config.learning_rate
    history: list[dict] = []
    # Each centralized round makes local_epochs passes through the pooled
    # training set, matching the per-example number of passes in a full-client
    # federated round. Validation and early stopping remain round-based.
    for round_number in range(1, config.rounds + 1):
        frozen = round_number <= config.freeze_backbone_rounds
        for parameter in model.features.parameters():
            parameter.requires_grad = not frozen
        optimizer = torch.optim.AdamW(
            filter(lambda parameter: parameter.requires_grad, model.parameters()),
            lr=current_lr, weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.local_epochs)
        )
        loader = make_loader(
            frame, train_indices, config.image_size, config.batch_size,
            config.num_workers, True, config.seed + round_number,
            config.strong_augmentation, config.balanced_sampling,
        )
        model.train()
        for _ in range(config.local_epochs):
            optimizer.zero_grad(set_to_none=True)
            for step, (images, targets) in enumerate(loader, 1):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    enabled=config.amp and device.type == "cuda",
                ):
                    loss = criterion(model(images), targets) / config.gradient_accumulation
                scaler.scale(loss).backward()
                if step % config.gradient_accumulation == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        metrics, _ = evaluate(model, frame, val_indices, config, device, class_names)
        metrics.update({"round": round_number, "learning_rate": current_lr})
        history.append(metrics)
        print(
            f"round {round_number}: val macro-F1={metrics['macro_f1']:.4f}, "
            f"balanced accuracy={metrics['balanced_accuracy']:.4f}"
        )
        if metrics["macro_f1"] > best_f1:
            best_f1, stale, lr_stale = metrics["macro_f1"], 0, 0
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "image_size": config.image_size,
                "classifier_hidden_dim": config.classifier_hidden_dim,
                "classifier_dropout": config.classifier_dropout,
            }, output / "best_model.pt")
        else:
            stale += 1
            lr_stale += 1
            if lr_stale >= config.lr_patience and current_lr > config.min_learning_rate:
                updated_lr = max(config.min_learning_rate, current_lr * config.lr_factor)
                if updated_lr < current_lr:
                    current_lr, lr_stale = updated_lr, 0
                    print(f"Reducing centralized learning rate to {current_lr:.2e}")
            if stale >= config.early_stopping_patience:
                print("Early stopping: validation macro-F1 did not improve.")
                break

    checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, matrix, targets, probabilities = evaluate(
        model, frame, test_indices, config, device, class_names,
        return_probabilities=True,
    )
    (output / "metrics.json").write_text(
        json.dumps({"validation_history": history, "test": test_metrics}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        output / "confusion_matrix.csv"
    )
    test_by_hospital: dict[str, dict] = {}
    hospital_matrices: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {
        "class_names": np.asarray(class_names),
        "combined_targets": targets,
        "combined_probabilities": probabilities,
    }
    hospital_dir = output / "hospital_metrics"
    hospital_dir.mkdir(exist_ok=True)
    for hospital, indices in enumerate(client_test, 1):
        key = f"hospital_{hospital}"
        metrics, hospital_matrix, hospital_targets, hospital_probs = evaluate(
            model, frame, indices, config, device, class_names,
            return_probabilities=True,
        )
        test_by_hospital[key] = metrics
        hospital_matrices[key] = hospital_matrix
        predictions[f"{key}_targets"] = hospital_targets
        predictions[f"{key}_probabilities"] = hospital_probs
        pd.DataFrame(hospital_matrix, index=class_names, columns=class_names).to_csv(
            hospital_dir / f"{key}_confusion_matrix.csv"
        )
    np.savez_compressed(output / "test_predictions.npz", **predictions)
    save_result_plots(
        history, test_metrics, matrix, class_names, targets, probabilities,
        hospital_matrices, output,
    )
    macro_f1 = np.asarray([item["macro_f1"] for item in test_by_hospital.values()])
    balanced_accuracy = np.asarray(
        [item["balanced_accuracy"] for item in test_by_hospital.values()]
    )
    (output / "hospital_metrics.json").write_text(json.dumps({
        "final_centralized_model_by_hospital": test_by_hospital,
        "combined_test": test_metrics,
        "hospital_summary": {
            "macro_f1_mean": float(macro_f1.mean()),
            "macro_f1_std": float(macro_f1.std()),
            "worst_hospital_macro_f1": float(macro_f1.min()),
            "balanced_accuracy_mean": float(balanced_accuracy.mean()),
            "balanced_accuracy_std": float(balanced_accuracy.std()),
        },
    }, indent=2), encoding="utf-8")
    print(f"Centralized baseline saved to: {output}")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
