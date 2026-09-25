from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from skinfed.config import load_config, save_config
from skinfed.data import make_federated_splits, read_manifest
from skinfed.federated import aggregate, evaluate, train_client
from skinfed.model import build_model


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description="Memory-efficient federated skin-lesion training")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="config/gtx1650.yaml")
    parser.add_argument("--output", default="runs")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download/use ImageNet weights (only for offline smoke tests)")
    args = parser.parse_args()
    config = load_config(args.config); seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": print("WARNING: CUDA is unavailable; this run will be slow on CPU.")
    else: print(f"Using {torch.cuda.get_device_name(0)}")
    frame, class_names = read_manifest(args.manifest)
    client_indices, client_val_indices, client_test_indices, val_indices, test_indices = make_federated_splits(
        frame, config.clients, config.dirichlet_alpha, config.train_fraction, config.val_fraction, config.seed
    )
    output = Path(args.output) / config.run_name; output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.yaml")
    (output / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    distributions = {}
    for split_name, split_indices in (("train", client_indices), ("validation", client_val_indices), ("test", client_test_indices)):
        distributions[split_name] = pd.DataFrame(
            {f"hospital_{i + 1}": frame.iloc[idx].label.value_counts() for i, idx in enumerate(split_indices)}
        ).fillna(0).astype(int)
        distributions[split_name].to_csv(output / f"hospital_{split_name}_class_distribution.csv")
    counts = frame.iloc[np.concatenate(client_indices)].target.value_counts().reindex(range(len(class_names)), fill_value=1).to_numpy()
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
    model = build_model(len(class_names), pretrained=not args.no_pretrained, use_cbam=config.use_cbam,
                        classifier_hidden_dim=config.classifier_hidden_dim, classifier_dropout=config.classifier_dropout).to(device)
    best_f1, stale, lr_stale, current_lr, history = -1.0, 0, 0, config.learning_rate, []
    for round_number in range(1, config.rounds + 1):
        rng = np.random.default_rng(config.seed + round_number)
        selected_count = max(1, math.ceil(config.clients * config.client_fraction))
        selected = rng.choice(config.clients, selected_count, replace=False)
        updates = [
            train_client(model, frame, client_indices[i], client_val_indices[i], config, device, weights, round_number, current_lr)
            for i in selected
        ]
        aggregate(model, updates)
        metrics, _ = evaluate(model, frame, val_indices, config, device, class_names)
        metrics["round"] = round_number; metrics["learning_rate"] = current_lr; metrics["selected_hospitals"] = [int(x) + 1 for x in selected]
        metrics["local_validation"] = {
            f"hospital_{int(client) + 1}": update[3] for client, update in zip(selected, updates)
        }
        history.append(metrics)
        print(f"round {round_number}: val macro-F1={metrics['macro_f1']:.4f}, balanced accuracy={metrics['balanced_accuracy']:.4f}")
        if metrics["macro_f1"] > best_f1:
            best_f1, stale, lr_stale = metrics["macro_f1"], 0, 0
            torch.save({"model_state": model.state_dict(), "class_names": class_names, "image_size": config.image_size,
                        "classifier_hidden_dim": config.classifier_hidden_dim, "classifier_dropout": config.classifier_dropout}, output / "best_model.pt")
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
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, matrix = evaluate(model, frame, test_indices, config, device, class_names)
    (output / "metrics.json").write_text(json.dumps({"validation_history": history, "test": test_metrics}, indent=2), encoding="utf-8")
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(output / "confusion_matrix.csv")
    hospital_test_metrics = {}
    hospital_dir = output / "hospital_metrics"; hospital_dir.mkdir(exist_ok=True)
    for hospital, indices in enumerate(client_test_indices, 1):
        metrics, hospital_matrix = evaluate(model, frame, indices, config, device, class_names)
        hospital_test_metrics[f"hospital_{hospital}"] = metrics
        pd.DataFrame(hospital_matrix, index=class_names, columns=class_names).to_csv(
            hospital_dir / f"hospital_{hospital}_confusion_matrix.csv"
        )
    macro_f1 = np.array([metrics["macro_f1"] for metrics in hospital_test_metrics.values()])
    balanced_accuracy = np.array([metrics["balanced_accuracy"] for metrics in hospital_test_metrics.values()])
    hospital_report = {
        "final_federated_model_by_hospital": hospital_test_metrics,
        "combined_test": test_metrics,
        "hospital_summary": {
            "macro_f1_mean": float(macro_f1.mean()), "macro_f1_std": float(macro_f1.std()), "worst_hospital_macro_f1": float(macro_f1.min()),
            "balanced_accuracy_mean": float(balanced_accuracy.mean()), "balanced_accuracy_std": float(balanced_accuracy.std()),
        },
    }
    (output / "hospital_metrics.json").write_text(json.dumps(hospital_report, indent=2), encoding="utf-8")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
