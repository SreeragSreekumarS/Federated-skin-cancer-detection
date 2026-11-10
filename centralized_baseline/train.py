"""Train the same classifier centrally using one global group-aware split."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from skinfed.config import load_config, save_config
from skinfed.data import METADATA_FEATURE_DIM, _split_indices, cache_resized_images, encode_metadata, make_federated_splits, read_manifest
from skinfed.federated import (
    FocalCrossEntropy, ModelEMA, evaluate, make_loader, predict_logits,
    mixup_cutmix, optimizer_parameter_groups,
)
from skinfed.calibration import apply_decision_rule, select_decision_rule
from skinfed.metrics import classification_metrics
from skinfed.model import build_model
from skinfed.model import freeze_features
from skinfed.plots import save_result_plots


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one centralized model using a global patient/lesion split"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="config/gtx1650.yaml")
    parser.add_argument("--output", default="runs/centralized")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-cbam", action="store_true", help="Disable the MobileNetV3 CBAM attention block")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision if training becomes numerically unstable")
    parser.add_argument("--seed", type=int, help="Training seed; split_seed stays fixed for repeated-seed ensembles")
    parser.add_argument("--run-name")
    parser.add_argument("--model-name", help="Model name, including timm models such as mobilenetv4_conv_medium")
    parser.add_argument("--image-size", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None: config.seed = args.seed
    if args.run_name: config.run_name = args.run_name
    if args.model_name: config.model_name = args.model_name
    if args.image_size: config.image_size = args.image_size
    if args.no_cbam: config.use_cbam = False
    if args.no_amp: config.amp = False
    config.validate()
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is unavailable; this run will be slow on CPU.")
    else:
        print(f"Using {torch.cuda.get_device_name(0)}")

    frame, class_names = read_manifest(args.manifest)
    if config.use_metadata:
        encode_metadata(frame)
    if config.cache_resized_images:
        cache_dir = Path(args.manifest).resolve().parent / f".resized_cache_{config.image_size}"
        print(f"Preparing resized-image cache at {cache_dir}")
        frame = cache_resized_images(frame, config.image_size, cache_dir)
    hospital_partition = None
    if config.partition_mode == "hospital_first":
        hospital_partition = make_federated_splits(
            frame, config.clients, config.dirichlet_alpha,
            config.train_fraction, config.val_fraction, config.split_seed,
        )
        client_train, client_val, client_test, val_indices, test_indices = hospital_partition
        train_indices = np.concatenate(client_train)
    else:
        train_indices, val_indices, test_indices = _split_indices(
            frame, config.train_fraction, config.val_fraction, config.split_seed
        )
    output = Path(args.output) / config.run_name
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.yaml")
    (output / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    assignments = []
    if hospital_partition is not None:
        for hospital, (train_part, val_part, test_part) in enumerate(
            zip(hospital_partition[0], hospital_partition[1], hospital_partition[2]), 1
        ):
            for split_name, indices in (("train", train_part), ("validation", val_part), ("test", test_part)):
                for index in indices:
                    row = frame.iloc[int(index)]
                    assignments.append({
                        "manifest_index": int(index),
                        "image_path": getattr(row, "source_image_path", row.image_path),
                        "label": row.label,
                        "patient_id": row.patient_id,
                        "hospital_id": hospital,
                        "split": split_name,
                    })
    else:
        for split_name, indices in (("train", train_indices), ("validation", val_indices), ("test", test_indices)):
            for index in indices:
                row = frame.iloc[int(index)]
                assignments.append({
                    "manifest_index": int(index),
                    "image_path": getattr(row, "source_image_path", row.image_path),
                    "label": row.label,
                    "patient_id": row.patient_id,
                    "split": split_name,
                })
    pd.DataFrame(assignments).sort_values("manifest_index").to_csv(
        output / "split_assignments.csv", index=False
    )
    pd.DataFrame({
        "split": ["train", "validation", "test"],
        "images": [len(train_indices), len(val_indices), len(test_indices)],
        "patient_or_lesion_groups": [
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
        drop_path_rate=config.drop_path_rate,
        model_name=config.model_name,
        metadata_dim=METADATA_FEATURE_DIM if config.use_metadata else 0,
    ).to(device)
    ema = ModelEMA(model, config.ema_decay)
    freeze_features(model, frozen=config.freeze_backbone_rounds > 0)
    scaler = torch.amp.GradScaler(
        device.type, enabled=config.amp and device.type == "cuda"
    )

    best_f1, stale, lr_stale = -1.0, 0, 0
    current_lr = config.learning_rate
    history: list[dict] = []
    # Each centralized round makes local_epochs passes through the full global
    # training partition. Emit live batch progress and validate after each epoch.
    epoch_number = 0
    stop_training = False
    for round_number in range(1, config.rounds + 1):
        frozen = round_number <= config.freeze_backbone_rounds
        freeze_features(model, frozen=frozen)
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(model, config, current_lr),
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.local_epochs)
        )
        loader = make_loader(
            frame, train_indices, config.image_size, config.batch_size,
            config.num_workers, True, config.seed + round_number,
            config.strong_augmentation, config.balanced_sampling,
            config.cache_resized_images, config.use_metadata,
        )
        model.train()
        for _ in range(config.local_epochs):
            epoch_number += 1
            epoch_started = time.perf_counter()
            train_loss_sum, train_correct, train_samples = 0.0, 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            progress = tqdm(loader, desc=f"epoch {epoch_number} (round {round_number})", unit="step", dynamic_ncols=True)
            for step, batch in enumerate(progress, 1):
                if config.use_metadata:
                    images, metadata, targets = batch
                    metadata = metadata.to(device, non_blocking=True)
                else:
                    images, targets = batch
                    metadata = None
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                images, targets_a, targets_b, mix_lambda = mixup_cutmix(images, targets, config)
                with torch.autocast(
                    device_type=device.type,
                    enabled=config.amp and device.type == "cuda",
                ):
                    logits = model(images, metadata) if config.use_metadata else model(images)
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(
                            "Training produced non-finite logits. Retry with --no-amp and a lower learning rate."
                        )
                    loss = criterion(logits, targets_a)
                    if targets_b is not None:
                        loss = mix_lambda * loss + (1 - mix_lambda) * criterion(logits, targets_b)
                    loss = loss / config.gradient_accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Training produced a non-finite loss. Retry with --no-amp and a lower learning rate."
                    )
                batch_size = targets.size(0)
                train_loss_sum += float(loss.detach()) * config.gradient_accumulation * batch_size
                batch_predictions = logits.detach().argmax(dim=1)
                if targets_b is None:
                    train_correct += float((batch_predictions == targets_a).sum())
                else:
                    train_correct += float(
                        mix_lambda * (batch_predictions == targets_a).sum()
                        + (1 - mix_lambda) * (batch_predictions == targets_b).sum()
                    )
                train_samples += batch_size
                scaler.scale(loss).backward()
                if step % config.gradient_accumulation == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)
                elapsed = time.perf_counter() - epoch_started
                progress.set_postfix(
                    loss=f"{train_loss_sum / max(1, train_samples):.4f}",
                    accuracy=f"{train_correct / max(1, train_samples):.4f}",
                    elapsed=f"{elapsed:.0f}s",
                    step=f"{elapsed / step * 1000:.0f}ms",
                    refresh=False,
                )
            progress.close()
            scheduler.step()
            metrics, _ = evaluate(ema.ema, frame, val_indices, config, device, class_names)
            metrics.update({
                "round": round_number, "epoch": epoch_number,
                "learning_rate": current_lr,
                "train_loss": train_loss_sum / max(1, train_samples),
                "train_accuracy": train_correct / max(1, train_samples),
                "epoch_seconds": time.perf_counter() - epoch_started,
                "milliseconds_per_step": (time.perf_counter() - epoch_started) / max(1, len(loader)) * 1000,
            })
            history.append(metrics)
            print(
                f"epoch {epoch_number}: loss={metrics['train_loss']:.4f} "
                f"accuracy={metrics['train_accuracy']:.4f} "
                f"val_loss={metrics['log_loss']:.4f} val_accuracy={metrics['accuracy']:.4f} "
                f"val_macro_f1={metrics['macro_f1']:.4f} "
                f"time={metrics['epoch_seconds']:.1f}s "
                f"step={metrics['milliseconds_per_step']:.0f}ms"
            )
            if metrics["macro_f1"] > best_f1:
                best_f1, stale, lr_stale = metrics["macro_f1"], 0, 0
                torch.save({
                    "model_state": ema.ema.state_dict(),
                    "class_names": class_names,
                    "model_name": config.model_name,
                    "image_size": config.image_size,
                    "classifier_hidden_dim": config.classifier_hidden_dim,
                    "classifier_dropout": config.classifier_dropout,
                    "drop_path_rate": config.drop_path_rate,
                    "metadata_dim": METADATA_FEATURE_DIM if config.use_metadata else 0,
                }, output / "best_model.pt")
            else:
                stale += 1
                lr_stale += 1
                if lr_stale >= config.lr_patience and current_lr > config.min_learning_rate:
                    updated_lr = max(config.min_learning_rate, current_lr * config.lr_factor)
                    if updated_lr < current_lr:
                        current_lr, lr_stale = updated_lr, 0
                        print(f"Reducing centralized learning rate to {current_lr:.2e} on next round")
                if stale >= config.early_stopping_patience:
                    print("Early stopping: validation macro-F1 did not improve.")
                    stop_training = True
            if stop_training:
                break
        if stop_training:
            break

    checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=False)
    ema.ema.load_state_dict(checkpoint["model_state"])
    validation_logits, validation_targets = predict_logits(ema.ema, frame, val_indices, config, device)
    train_counts = frame.iloc[train_indices].target.value_counts().reindex(
        range(len(class_names)), fill_value=0
    ).to_numpy()
    decision = select_decision_rule(
        validation_logits, validation_targets, class_names, train_counts,
        metric=config.decision_metric,
        preserve_melanoma_sensitivity=config.preserve_melanoma_sensitivity,
    )
    test_logits, targets = predict_logits(ema.ema, frame, test_indices, config, device)
    raw_probabilities = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
    raw_predictions = test_logits.argmax(axis=1)
    raw_metrics, raw_matrix = classification_metrics(
        targets, raw_probabilities, class_names, raw_predictions
    )
    probabilities, selected_predictions = apply_decision_rule(test_logits, decision, class_names)
    selected_metrics, matrix = classification_metrics(
        targets, probabilities, class_names, selected_predictions
    )
    (output / "metrics.json").write_text(
        json.dumps({
            "validation_history": history,
            "decision_selection": decision,
            "raw_argmax": raw_metrics,
            "selected": selected_metrics,
            "test": selected_metrics,
        }, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        output / "confusion_matrix.csv"
    )
    pd.DataFrame(raw_matrix, index=class_names, columns=class_names).to_csv(
        output / "confusion_matrix_raw_argmax.csv"
    )
    predictions: dict[str, np.ndarray] = {
        "class_names": np.asarray(class_names),
        "targets": targets,
        "raw_probabilities": raw_probabilities,
        "raw_predictions": raw_predictions,
        "calibrated_probabilities": probabilities,
        "selected_predictions": selected_predictions,
        "probabilities": probabilities,
        "predicted_labels": selected_predictions,
    }
    np.savez_compressed(output / "test_predictions.npz", **predictions)
    save_result_plots(
        history, selected_metrics, matrix, class_names, targets, probabilities,
        {}, output,
    )
    print(f"Centralized baseline saved to: {output}")
    print(json.dumps({"raw_argmax": raw_metrics, "selected": selected_metrics,
                      "decision_rule": decision["rule"]}, indent=2))


if __name__ == "__main__":
    main()
