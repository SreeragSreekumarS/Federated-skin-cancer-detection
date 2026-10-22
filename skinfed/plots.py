from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_training_curves(history: list[dict], plots_dir: Path) -> None:
    rounds = [item["round"] for item in history]
    clients = sorted({
        name
        for item in history
        for name in item.get("local_validation", {})
    })
    if not rounds:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for client in clients:
        train_accuracy, val_accuracy, train_loss, val_loss = [], [], [], []
        for item in history:
            values = item.get("local_validation", {}).get(client, {})
            train_accuracy.append(100 * values.get("train_accuracy", np.nan))
            train_loss.append(values.get("train_loss", np.nan))
            val_accuracy.append(100 * values.get("accuracy", np.nan))
            val_loss.append(values.get("log_loss", np.nan))
        client_label = client.replace("hospital_", "Client ")
        train_line, = axes[0].plot(rounds, train_accuracy, marker="o", markersize=3,
                                   label=f"{client_label} train")
        axes[0].plot(rounds, val_accuracy, marker=".", markersize=4, linestyle="--",
                     color=train_line.get_color(), label=f"{client_label} validation")
        train_line, = axes[1].plot(rounds, train_loss, marker="o", markersize=3,
                                   label=f"{client_label} train")
        axes[1].plot(rounds, val_loss, marker=".", markersize=4, linestyle="--",
                     color=train_line.get_color(), label=f"{client_label} validation")
    axes[0].set_title("Client Accuracy by Federated Round")
    axes[0].set_xlabel("Federated round")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[1].set_title("Client Loss by Federated Round")
    axes[1].set_xlabel("Federated round")
    axes[1].set_ylabel("Loss")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    _save(fig, plots_dir / "client_training_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(rounds, [100 * item["macro_f1"] for item in history], marker="o", label="Macro F1")
    axes[0].plot(rounds, [100 * item["balanced_accuracy"] for item in history], marker="s", label="Balanced accuracy")
    axes[0].set_title("Global Validation Scores")
    axes[0].set_ylabel("Score (%)")
    axes[0].set_ylim(0, 100)
    axes[1].plot(rounds, [item["log_loss"] for item in history], marker="o", color="tab:red")
    axes[1].set_title("Global Validation Log Loss")
    axes[1].set_ylabel("Log loss")
    for axis in axes:
        axis.set_xlabel("Federated round")
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    _save(fig, plots_dir / "global_validation_curves.png")


def _plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(max(6, len(class_names) * 0.9), max(5, len(class_names) * 0.75)))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis, label="Image count")
    axis.set(
        xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted label", ylabel="True label", title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, f"{matrix[row, col]:d}", ha="center", va="center",
                      color="white" if matrix[row, col] > threshold else "black", fontsize=9)
    _save(fig, path)


def _plot_class_auc(metrics: dict, class_names: list[str], plots_dir: Path) -> None:
    values = [metrics["per_class"][name]["roc_auc_ovr"] for name in class_names]
    fig, axis = plt.subplots(figsize=(max(7, len(class_names) * 1.1), 4.8))
    positions = np.arange(len(class_names))
    bars = axis.bar(positions, [np.nan if value is None else value for value in values], color="steelblue")
    axis.set_xticks(positions, class_names, rotation=35, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("One-vs-rest ROC AUC")
    axis.set_title("Per-Class ROC AUC on Combined Held-Out Test Set")
    axis.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, 0.02 if value is None else value + 0.02,
                  "N/A" if value is None else f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    _save(fig, plots_dir / "per_class_auc.png")


def _plot_curves(y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str], plots_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 6))
    plotted = 0
    for class_index, class_name in enumerate(class_names):
        binary_target = (y_true == class_index).astype(int)
        if np.unique(binary_target).size < 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(binary_target, probabilities[:, class_index])
        auc_score = roc_auc_score(binary_target, probabilities[:, class_index])
        axis.plot(false_positive_rate, true_positive_rate, label=f"{class_name} (AUC = {auc_score:.3f})")
        plotted += 1
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate",
             title="One-vs-Rest ROC Curves — Combined Held-Out Test Set")
    axis.grid(True, alpha=0.3)
    if plotted:
        axis.legend(loc="lower right", fontsize="small")
    else:
        axis.text(0.5, 0.5, "ROC curves unavailable: test split has only one class", ha="center", va="center")
    _save(fig, plots_dir / "roc_curves.png")

    fig, axis = plt.subplots(figsize=(7.2, 6))
    plotted = 0
    for class_index, class_name in enumerate(class_names):
        binary_target = (y_true == class_index).astype(int)
        if np.unique(binary_target).size < 2:
            continue
        precision, recall, _ = precision_recall_curve(binary_target, probabilities[:, class_index])
        ap = average_precision_score(binary_target, probabilities[:, class_index])
        axis.plot(recall, precision, label=f"{class_name} (AP = {ap:.3f})")
        plotted += 1
    axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
             title="One-vs-Rest Precision–Recall Curves — Combined Held-Out Test Set")
    axis.grid(True, alpha=0.3)
    if plotted:
        axis.legend(loc="best", fontsize="small")
    else:
        axis.text(0.5, 0.5, "Curves unavailable: test split has only one class", ha="center", va="center")
    _save(fig, plots_dir / "precision_recall_curves.png")


def save_result_plots(
    history: list[dict],
    test_metrics: dict,
    test_matrix: np.ndarray,
    class_names: list[str],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    hospital_matrices: dict[str, np.ndarray],
    output_dir: str | Path,
) -> None:
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_training_curves(history, plots_dir)
    _plot_confusion_matrix(test_matrix, class_names, plots_dir / "confusion_matrix_heatmap.png",
                           "Combined Held-Out Test Confusion Matrix")
    _plot_class_auc(test_metrics, class_names, plots_dir)
    _plot_curves(y_true, probabilities, class_names, plots_dir)
    for hospital, matrix in hospital_matrices.items():
        _plot_confusion_matrix(matrix, class_names,
                               plots_dir / f"{hospital}_confusion_matrix_heatmap.png",
                               f"Held-Out Test Confusion Matrix — {hospital.replace('_', ' ').title()}")
