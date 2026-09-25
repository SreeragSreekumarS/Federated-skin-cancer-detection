from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, log_loss, matthews_corrcoef, precision_recall_fscore_support,
    roc_auc_score, top_k_accuracy_score,
)


def classification_metrics(targets: list[int], probabilities: np.ndarray, class_names: list[str]) -> tuple[dict, np.ndarray]:
    y_true = np.asarray(targets)
    y_pred = probabilities.argmax(axis=1)
    labels = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
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
            "negative_predictive_value": float(tn / (tn + fn)) if tn + fn else 0.0,
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
            "f1": float(f1[i]), "roc_auc_ovr": class_auc,
            "average_precision": class_average_precision,
            "true_positives": int(tp), "true_negatives": int(tn),
            "false_positives": int(fp), "false_negatives": int(fn), "support": int(support[i]),
        }
    try:
        auc = roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average="macro")
    except ValueError:  # A held-out split without every class cannot define multiclass AUC.
        auc = None
    try:
        macro_average_precision = average_precision_score(
            np.eye(len(labels), dtype=int)[y_true], probabilities, average="macro"
        )
    except ValueError:
        macro_average_precision = None
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
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
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
