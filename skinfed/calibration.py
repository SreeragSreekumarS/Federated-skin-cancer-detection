from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Fit a single temperature on validation logits only."""
    values = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(targets, dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = torch.nn.functional.cross_entropy(values / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def fit_class_thresholds(probabilities: np.ndarray, targets: np.ndarray, class_names: list[str]) -> dict[str, float]:
    """Fit coarse one-vs-rest F1 thresholds and shrink toward 0.5 for rare classes."""
    thresholds = {}
    candidates = np.arange(0.2, 0.81, 0.1)
    for class_index, class_name in enumerate(class_names):
        truth = np.asarray(targets) == class_index
        if truth.all() or not truth.any():
            thresholds[class_name] = 0.5
            continue
        scores = [f1_score(truth, probabilities[:, class_index] >= value, zero_division=0) for value in candidates]
        fitted = float(candidates[int(np.argmax(scores))])
        shrink = int(truth.sum()) / (int(truth.sum()) + 50.0)
        thresholds[class_name] = float(0.5 + shrink * (fitted - 0.5))
    return thresholds


def threshold_predictions(probabilities: np.ndarray, thresholds: dict[str, float], class_names: list[str]) -> np.ndarray:
    limits = np.asarray([thresholds[name] for name in class_names])
    margins = probabilities - limits[None, :]
    accepted = margins >= 0
    # Prefer the passing class with the strongest threshold margin; if none pass,
    # retain the ordinary highest-probability prediction.
    margins[~accepted] = -np.inf
    predictions = margins.argmax(axis=1)
    no_pass = ~accepted.any(axis=1)
    predictions[no_pass] = probabilities[no_pass].argmax(axis=1)
    return predictions


def _metric_values(targets: np.ndarray, predictions: np.ndarray, class_names: list[str]) -> dict:
    _, _, macro_f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=np.arange(len(class_names)), average="macro", zero_division=0
    )
    values = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(macro_f1),
    }
    melanoma = next((i for i, name in enumerate(class_names) if "mel" in name.lower()), None)
    if melanoma is not None:
        positives = targets == melanoma
        values["melanoma_sensitivity"] = float((predictions[positives] == melanoma).mean()) if positives.any() else None
    return values


def select_decision_rule(
    validation_logits: np.ndarray,
    validation_targets: np.ndarray,
    class_names: list[str],
    train_counts: np.ndarray,
    metric: str = "accuracy",
    preserve_melanoma_sensitivity: bool = True,
) -> dict:
    """Select among raw argmax, validation-F1 thresholds, and prior logit adjustment."""
    temperature = fit_temperature(validation_logits, validation_targets)
    val_probs = torch.softmax(torch.as_tensor(validation_logits) / temperature, dim=1).numpy()
    thresholds = fit_class_thresholds(val_probs, validation_targets, class_names)
    priors = np.asarray(train_counts, dtype=np.float64)
    priors = np.maximum(priors, 1) / max(1, np.maximum(priors, 1).sum())
    log_priors = np.log(priors)

    candidates = [{"rule": "argmax", "strength": 0.0, "predictions": validation_logits.argmax(axis=1)}]
    threshold_pred = threshold_predictions(val_probs, thresholds, class_names)
    candidates.append({"rule": "class_thresholds", "strength": 0.0, "predictions": threshold_pred})
    for strength in (0.05, 0.1, 0.2, 0.35, 0.5):
        adjusted = validation_logits / temperature - strength * log_priors[None, :]
        candidates.append({"rule": "logit_adjustment", "strength": strength, "predictions": adjusted.argmax(axis=1)})

    raw_metrics = _metric_values(validation_targets, candidates[0]["predictions"], class_names)
    for candidate in candidates:
        values = _metric_values(validation_targets, candidate["predictions"], class_names)
        candidate["validation_metrics"] = values
        if candidate["rule"] != "argmax" and preserve_melanoma_sensitivity:
            raw_recall = raw_metrics.get("melanoma_sensitivity")
            candidate_recall = values.get("melanoma_sensitivity")
            if raw_recall is not None and candidate_recall is not None and candidate_recall < raw_recall:
                candidate["ineligible"] = "melanoma sensitivity below raw argmax"

    baseline = candidates[0]
    selected = baseline
    for candidate in candidates[1:]:
        if candidate.get("ineligible"):
            continue
        if candidate["validation_metrics"][metric] > selected["validation_metrics"][metric] + 1e-12:
            selected = candidate
    return {
        "rule": selected["rule"],
        "strength": selected["strength"],
        "temperature": temperature,
        "class_thresholds": thresholds,
        "class_priors": priors.tolist(),
        "selection_metric": metric,
        "preserve_melanoma_sensitivity": preserve_melanoma_sensitivity,
        "selected_validation_metrics": selected["validation_metrics"],
        "candidate_validation_metrics": [
            {key: value for key, value in candidate.items() if key != "predictions"}
            for candidate in candidates
        ],
    }


def apply_decision_rule(logits: np.ndarray, selection: dict, class_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    temperature = float(selection["temperature"])
    probabilities = torch.softmax(torch.as_tensor(logits) / temperature, dim=1).numpy()
    rule = selection["rule"]
    if rule == "class_thresholds":
        predictions = threshold_predictions(probabilities, selection["class_thresholds"], class_names)
    elif rule == "logit_adjustment":
        priors = np.asarray(selection["class_priors"], dtype=np.float64)
        adjusted = logits / temperature - float(selection["strength"]) * np.log(priors)[None, :]
        probabilities = torch.softmax(torch.as_tensor(adjusted), dim=1).numpy()
        predictions = adjusted.argmax(axis=1)
    else:
        predictions = logits.argmax(axis=1)
    return probabilities, predictions
