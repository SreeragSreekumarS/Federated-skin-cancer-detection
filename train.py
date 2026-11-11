from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from skinfed.config import load_config, save_config
from skinfed.data import METADATA_FEATURE_DIM, cache_resized_images, encode_metadata, make_federated_splits, read_manifest
from skinfed.federated import ModelEMA, aggregate, evaluate, predict_logits, train_client
from skinfed.calibration import apply_decision_rule, select_decision_rule
from skinfed.metrics import classification_metrics
from skinfed.model import build_model
from skinfed.plots import save_result_plots


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description="Memory-efficient federated skin-lesion training")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="config/gtx1650.yaml")
    parser.add_argument("--output", default="runs")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download/use ImageNet weights (only for offline smoke tests)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision for numerical stability")
    parser.add_argument("--aggregation-weighting", choices=("samples", "quality"), help="Federated aggregation weights")
    parser.add_argument("--seed", type=int, help="Training seed; split_seed remains fixed for repeated-seed ensembles")
    parser.add_argument("--run-name")
    parser.add_argument("--model-name", help="Model name, including timm models such as mobilenetv4_conv_medium")
    parser.add_argument("--image-size", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None: config.seed = args.seed
    if args.run_name: config.run_name = args.run_name
    if args.model_name: config.model_name = args.model_name
    if args.image_size: config.image_size = args.image_size
    if args.no_amp: config.amp = False
    if args.aggregation_weighting: config.aggregation_weighting = args.aggregation_weighting
    config.validate(); seed_everything(config.seed)
    config.partition_mode = "hospital_first"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": print("WARNING: CUDA is unavailable; this run will be slow on CPU.")
    else: print(f"Using {torch.cuda.get_device_name(0)}")
    print(f"Federated aggregation weighting: {config.aggregation_weighting}")
    frame, class_names = read_manifest(args.manifest)
    if config.use_metadata:
        encode_metadata(frame)
    if config.cache_resized_images:
        cache_dir = Path(args.manifest).resolve().parent / f".resized_cache_{config.image_size}"
        print(f"Preparing resized-image cache at {cache_dir}")
        frame = cache_resized_images(frame, config.image_size, cache_dir)
    client_indices, client_val_indices, client_test_indices, val_indices, test_indices = make_federated_splits(
        frame, config.clients, config.dirichlet_alpha, config.train_fraction, config.val_fraction,
        config.split_seed, config.min_groups_per_class_per_hospital,
    )
    output = Path(args.output) / config.run_name; output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.yaml")
    (output / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    assignments = []
    for hospital, (train_part, val_part, test_part) in enumerate(
        zip(client_indices, client_val_indices, client_test_indices), 1
    ):
        for split_name, split_indices in (("train", train_part), ("validation", val_part), ("test", test_part)):
            for index in split_indices:
                row = frame.iloc[int(index)]
                assignments.append({
                    "manifest_index": int(index),
                    "image_path": getattr(row, "source_image_path", row.image_path),
                    "label": row.label,
                    "patient_id": row.patient_id,
                    "hospital_id": hospital,
                    "split": split_name,
                })
    pd.DataFrame(assignments).sort_values("manifest_index").to_csv(
        output / "split_assignments.csv", index=False
    )
    distributions = {}
    for split_name, split_indices in (("train", client_indices), ("validation", client_val_indices), ("test", client_test_indices)):
        distributions[split_name] = pd.DataFrame(
            {f"hospital_{i + 1}": frame.iloc[idx].label.value_counts() for i, idx in enumerate(split_indices)}
        ).fillna(0).astype(int)
        distributions[split_name].to_csv(output / f"hospital_{split_name}_class_distribution.csv")
    counts = frame.iloc[np.concatenate(client_indices)].target.value_counts().reindex(range(len(class_names)), fill_value=1).to_numpy()
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
    model = build_model(len(class_names), pretrained=not args.no_pretrained, use_cbam=config.use_cbam,
                        classifier_hidden_dim=config.classifier_hidden_dim, classifier_dropout=config.classifier_dropout,
                        drop_path_rate=config.drop_path_rate, model_name=config.model_name,
                        metadata_dim=METADATA_FEATURE_DIM if config.use_metadata else 0).to(device)
    # The server updates once per round, so its EMA decay is shorter than the
    # per-optimizer-step EMA used by local/centralized trainers.
    ema = ModelEMA(model, config.server_ema_decay)
    best_f1, stale, lr_stale, current_lr, history = -1.0, 0, 0, config.learning_rate, []
    for round_number in range(1, config.rounds + 1):
        rng = np.random.default_rng(config.seed + round_number)
        selected_count = max(1, math.ceil(config.clients * config.client_fraction))
        selected = rng.choice(config.clients, selected_count, replace=False)
        updates = [
            train_client(model, frame, client_indices[i], client_val_indices[i], config, device, weights, round_number, current_lr)
            for i in selected
        ]
        aggregate(model, updates, config.aggregation_weighting)
        ema.update(model)
        metrics, _ = evaluate(ema.ema, frame, val_indices, config, device, class_names)
        metrics["round"] = round_number; metrics["learning_rate"] = current_lr; metrics["selected_hospitals"] = [int(x) + 1 for x in selected]
        metrics["local_validation"] = {
            f"hospital_{int(client) + 1}": update[3] for client, update in zip(selected, updates)
        }
        history.append(metrics)
        print(f"round {round_number}: val macro-F1={metrics['macro_f1']:.4f}, balanced accuracy={metrics['balanced_accuracy']:.4f}")
        if metrics["macro_f1"] > best_f1:
            best_f1, stale, lr_stale = metrics["macro_f1"], 0, 0
            torch.save({"model_state": ema.ema.state_dict(), "class_names": class_names,
                        "model_name": config.model_name, "image_size": config.image_size,
                        "classifier_hidden_dim": config.classifier_hidden_dim,
                        "classifier_dropout": config.classifier_dropout,
                        "drop_path_rate": config.drop_path_rate,
                        "metadata_dim": METADATA_FEATURE_DIM if config.use_metadata else 0}, output / "best_model.pt")
        else:
            stale += 1
            lr_stale += 1
            if lr_stale >= config.lr_patience and current_lr > config.min_learning_rate:
                updated_lr = max(config.min_learning_rate, current_lr * config.lr_factor)
                if updated_lr < current_lr:
                    current_lr = updated_lr
                    lr_stale = 0
                    print(f"Reducing client learning rate to {current_lr:.2e}")
            if stale >= config.early_stopping_patience:
                print("Early stopping: validation macro-F1 did not improve."); break
    checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=False)
    ema.ema.load_state_dict(checkpoint["model_state"])
    validation_logits, validation_targets = predict_logits(ema.ema, frame, val_indices, config, device)
    train_counts = frame.iloc[np.concatenate(client_indices)].target.value_counts().reindex(
        range(len(class_names)), fill_value=0
    ).to_numpy()
    decision = select_decision_rule(
        validation_logits, validation_targets, class_names, train_counts,
        metric=config.decision_metric,
        preserve_melanoma_sensitivity=config.preserve_melanoma_sensitivity,
    )
    test_logits, test_targets = predict_logits(ema.ema, frame, test_indices, config, device)
    raw_test_probabilities = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
    raw_test_predictions = test_logits.argmax(axis=1)
    raw_test_metrics, raw_matrix = classification_metrics(
        test_targets, raw_test_probabilities, class_names, raw_test_predictions
    )
    test_probabilities, selected_test_predictions = apply_decision_rule(test_logits, decision, class_names)
    test_metrics, matrix = classification_metrics(
        test_targets, test_probabilities, class_names, selected_test_predictions
    )
    (output / "metrics.json").write_text(json.dumps({
        "validation_history": history,
        "decision_selection": decision,
        "raw_argmax": raw_test_metrics,
        "selected": test_metrics,
        "test": test_metrics,
    }, indent=2), encoding="utf-8")
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(output / "confusion_matrix.csv")
    pd.DataFrame(raw_matrix, index=class_names, columns=class_names).to_csv(
        output / "confusion_matrix_raw_argmax.csv"
    )
    hospital_test_metrics = {}
    hospital_matrices = {}
    prediction_payload = {
        "class_names": np.asarray(class_names),
        "combined_targets": test_targets,
        "combined_raw_probabilities": raw_test_probabilities,
        "combined_raw_predictions": raw_test_predictions,
        "combined_probabilities": test_probabilities,
        "combined_selected_predictions": selected_test_predictions,
    }
    hospital_dir = output / "hospital_metrics"; hospital_dir.mkdir(exist_ok=True)
    for hospital, indices in enumerate(client_test_indices, 1):
        hospital_logits, hospital_targets = predict_logits(ema.ema, frame, indices, config, device)
        hospital_raw_probabilities = torch.softmax(torch.from_numpy(hospital_logits), dim=1).numpy()
        hospital_raw_predictions = hospital_logits.argmax(axis=1)
        hospital_raw_metrics, _ = classification_metrics(
            hospital_targets, hospital_raw_probabilities, class_names, hospital_raw_predictions
        )
        hospital_probabilities, hospital_predictions = apply_decision_rule(
            hospital_logits, decision, class_names
        )
        metrics, hospital_matrix = classification_metrics(
            hospital_targets, hospital_probabilities, class_names, hospital_predictions
        )
        hospital_test_metrics[f"hospital_{hospital}"] = {
            "raw_argmax": hospital_raw_metrics,
            "selected": metrics,
        }
        hospital_matrices[f"hospital_{hospital}"] = hospital_matrix
        prediction_payload[f"hospital_{hospital}_targets"] = hospital_targets
        prediction_payload[f"hospital_{hospital}_raw_probabilities"] = hospital_raw_probabilities
        prediction_payload[f"hospital_{hospital}_raw_predictions"] = hospital_raw_predictions
        prediction_payload[f"hospital_{hospital}_probabilities"] = hospital_probabilities
        prediction_payload[f"hospital_{hospital}_selected_predictions"] = hospital_predictions
        pd.DataFrame(hospital_matrix, index=class_names, columns=class_names).to_csv(
            hospital_dir / f"hospital_{hospital}_confusion_matrix.csv"
        )
    np.savez_compressed(output / "test_predictions.npz", **prediction_payload)
    save_result_plots(history, test_metrics, matrix, class_names, test_targets, test_probabilities,
                      hospital_matrices, output)
    macro_f1 = np.array([metrics["selected"]["macro_f1"] for metrics in hospital_test_metrics.values()])
    balanced_accuracy = np.array([metrics["selected"]["balanced_accuracy"] for metrics in hospital_test_metrics.values()])
    hospital_report = {
        "final_federated_model_by_hospital": hospital_test_metrics,
        "combined_raw_argmax": raw_test_metrics,
        "combined_selected": test_metrics,
        "combined_test": test_metrics,
        "hospital_summary": {
            "macro_f1_mean": float(macro_f1.mean()), "macro_f1_std": float(macro_f1.std()), "worst_hospital_macro_f1": float(macro_f1.min()),
            "balanced_accuracy_mean": float(balanced_accuracy.mean()), "balanced_accuracy_std": float(balanced_accuracy.std()),
        },
    }
    (output / "hospital_metrics.json").write_text(json.dumps(hospital_report, indent=2), encoding="utf-8")
    print(f"Plots saved to: {output / 'plots'}")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
