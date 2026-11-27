from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, cohen_kappa_score,
    confusion_matrix, log_loss, matthews_corrcoef, precision_recall_fscore_support,
    roc_auc_score, top_k_accuracy_score,
)


def checkpoint_score(metrics: dict, metric: str) -> float:
    """Return a maximization score for a validation metric."""
    value = float(metrics[metric])
    if not np.isfinite(value):
        raise ValueError(f"Validation checkpoint metric {metric!r} is not finite: {value}")
    return -value if metric == "log_loss" else value


def classification_metrics(targets: list[int], probabilities: np.ndarray, class_names: list[str], predicted_labels: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    y_true = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape != (len(y_true), len(class_names)):
        raise ValueError("Probability matrix shape must be (number of targets, number of classes)")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Predicted probabilities must be finite and non-negative")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("Each prediction must have positive total probability")
    # Normalize after float32 inference/autocast to satisfy sklearn's strict
    # multiclass probability validation and keep all metrics consistent.
    probabilities = probabilities / row_sums
    y_pred = probabilities.argmax(axis=1) if predicted_labels is None else np.asarray(predicted_labels, dtype=int)
    if y_pred.shape != y_true.shape or (y_pred < 0).any() or (y_pred >= len(class_names)).any():
        raise ValueError("predicted_labels must contain one valid class index per target")
    labels = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    present = support > 0
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    for i in labels:
        tp = matrix[i, i]; fp = matrix[:, i].sum() - tp; fn = matrix[i, :].sum() - tp
        tn = matrix.sum() - tp - fp - fn
        positives = (y_true == i).astype(int)
        if positives.min() == positives.max():
            class_auc, class_average_precision = None, None
        else:
            class_auc = float(roc_auc_score(positives, probabilities[:, i]))
            class_average_precision = float(average_precision_score(positives, probabilities[:, i]))
        per_class[class_names[i]] = {
            "precision": float(precision[i]), "positive_predictive_value": float(precision[i]),
            "sensitivity": float(recall[i]), "recall": float(recall[i]),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            # One-vs-rest accuracy (TP + TN) / N; macro-average is reported
            # separately from ordinary multiclass top-1 accuracy.
            "one_vs_rest_accuracy": float((tp + tn) / matrix.sum()) if matrix.sum() else 0.0,
            "negative_predictive_value": float(tn / (tn + fn)) if tn + fn else 0.0,
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
            "f1": float(f1[i]), "roc_auc_ovr": class_auc,
            "average_precision": class_average_precision,
            "true_positives": int(tp), "true_negatives": int(tn),
            "false_positives": int(fp), "false_negatives": int(fn), "support": int(support[i]),
        }
    # Average only defined one-vs-rest AUCs. Calling sklearn's multiclass
    # scorer with the full label set warns when a small client split omits a
    # class, even though those unsupported classes cannot contribute an AUC.
    valid_aucs = [
        item["roc_auc_ovr"] for item in per_class.values()
        if item["roc_auc_ovr"] is not None
    ]
    auc = float(np.mean(valid_aucs)) if valid_aucs else None
    valid_average_precisions = [
        item["average_precision"] for item in per_class.values()
        if item["average_precision"] is not None
    ]
    macro_average_precision = float(np.mean(valid_average_precisions)) if valid_average_precisions else None
    macro_ovr_accuracy = float(np.mean([
        item["one_vs_rest_accuracy"] for item in per_class.values()
    ])) if per_class else 0.0
    try:
        top_2_accuracy = top_k_accuracy_score(y_true, probabilities, k=min(2, len(labels)), labels=labels)
    except ValueError:
        top_2_accuracy = None
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="micro", zero_division=0
    )
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_ovr_accuracy": macro_ovr_accuracy,
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1_present": float(f1[present].mean()) if present.any() else 0.0,
        "macro_precision": float(macro_precision), "macro_recall": float(macro_recall), "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision), "weighted_recall": float(weighted_recall), "weighted_f1": float(weighted_f1),
        "micro_precision": float(micro_precision), "micro_recall": float(micro_recall), "micro_f1": float(micro_f1),
        "macro_auc_ovr": None if auc is None else float(auc),
        "macro_average_precision_ovr": None if macro_average_precision is None else float(macro_average_precision),
        "top_2_accuracy": None if top_2_accuracy is None else float(top_2_accuracy),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "per_class": per_class,
    }
    return results, matrix
