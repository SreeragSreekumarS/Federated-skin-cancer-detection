from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score


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
    """Choose one-vs-rest F1 thresholds from validation probabilities."""
    thresholds = {}
    candidates = np.linspace(0.05, 0.95, 91)
    for class_index, class_name in enumerate(class_names):
        truth = np.asarray(targets) == class_index
        if truth.all() or not truth.any():
            thresholds[class_name] = 0.5
            continue
        scores = [f1_score(truth, probabilities[:, class_index] >= value, zero_division=0) for value in candidates]
        thresholds[class_name] = float(candidates[int(np.argmax(scores))])
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
