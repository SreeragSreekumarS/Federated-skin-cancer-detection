from __future__ import annotations

from pathlib import Path
import random
from itertools import combinations
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset
from torchvision import transforms


REQUIRED_COLUMNS = {"image_path", "label"}


def read_manifest(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    path = Path(path).resolve()
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["image_path", "label"]).copy()
    frame["image_path"] = frame["image_path"].map(
        lambda value: str((path.parent / value).resolve()) if not Path(value).is_absolute() else str(Path(value))
    )
    absent = [p for p in frame.image_path if not Path(p).is_file()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} image paths do not exist; first: {absent[0]}")
    classes = sorted(frame.label.astype(str).unique().tolist())
    frame["target"] = frame.label.astype(str).map({name: i for i, name in enumerate(classes)})
    if "patient_id" not in frame:
        frame["patient_id"] = [f"image-{i}" for i in range(len(frame))]
    return frame.reset_index(drop=True), classes


def _split_indices(frame: pd.DataFrame, train: float, val: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(frame))
    groups = frame.patient_id.astype(str).to_numpy()
    labels = frame.target.to_numpy()
    # Select grouped, stratified folds closest to the requested holdout size and
    # class proportions. This keeps lesions together while reducing empty-class
    # validation/test splits caused by unstratified group sampling.
    try:
        splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
        folds = [holdout for _, holdout in splitter.split(indices, labels, groups)]
        class_counts = np.bincount(labels)
        target_holdout = 1.0 - train

        def split_cost(holdout_idx: np.ndarray, target_fraction: float) -> float:
            size_error = abs(len(holdout_idx) / len(indices) - target_fraction)
            held_counts = np.bincount(labels[holdout_idx], minlength=len(class_counts))
            class_error = np.abs(held_counts / np.maximum(class_counts, 1) - target_fraction).mean()
            missing_class_penalty = np.count_nonzero((class_counts > 0) & (held_counts == 0))
            return size_error + class_error + missing_class_penalty

        holdout_idx = min(
            (np.concatenate([folds[i] for i in choice]) for choice in combinations(range(10), max(1, round(target_holdout * 10)))),
            key=lambda candidate: split_cost(candidate, target_holdout),
        )
        train_idx = np.setdiff1d(indices, holdout_idx, assume_unique=False)
        holdout_labels = labels[holdout_idx]
        holdout_groups = groups[holdout_idx]
        inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed + 1)
        inner_splits = list(inner.split(holdout_idx, holdout_labels, holdout_groups))
        val_ratio = val / (1.0 - train)

        def inner_cost(val_rel: np.ndarray) -> float:
            class_counts = np.bincount(holdout_labels, minlength=len(np.bincount(labels)))
            val_counts = np.bincount(holdout_labels[val_rel], minlength=len(class_counts))
            class_error = np.abs(val_counts / np.maximum(class_counts, 1) - val_ratio).mean()
            missing_class_penalty = np.count_nonzero((class_counts > 0) & (val_counts == 0))
            return abs(len(val_rel) / len(holdout_idx) - val_ratio) + class_error + missing_class_penalty

        val_rel, test_rel = min(
            ((a, b) for a, b in inner_splits),
            key=lambda pair: inner_cost(pair[0]),
        )
        return train_idx, holdout_idx[val_rel], holdout_idx[test_rel]
    except ValueError:
        # Small datasets may not have enough distinct groups for grouped stratification.
        # Keep groups separate whenever possible; only fall back to sample stratification
        # when a group-aware split cannot be formed at all.
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train, random_state=seed)
        try:
            outer = GroupShuffleSplit(n_splits=1, train_size=train, random_state=seed)
            train_idx, holdout_idx = next(outer.split(indices, labels, groups))
        except ValueError:
            train_idx, holdout_idx = next(splitter.split(indices, labels))
        try:
            inner = GroupShuffleSplit(n_splits=1, train_size=val / (1 - train), random_state=seed + 1)
            val_rel, test_rel = next(inner.split(holdout_idx, labels[holdout_idx], groups[holdout_idx]))
        except ValueError:
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=val / (1 - train), random_state=seed + 1)
            val_rel, test_rel = next(splitter.split(holdout_idx, labels[holdout_idx]))
        return train_idx, holdout_idx[val_rel], holdout_idx[test_rel]


def _dirichlet_partition(indices: np.ndarray, labels: np.ndarray, clients: int, alpha: float, rng: np.random.Generator) -> list[np.ndarray]:
    """Allocate one split to simulated hospitals while retaining class imbalance."""
    parts: list[list[int]] = [[] for _ in range(clients)]
    for cls in np.unique(labels[indices]):
        cls_idx = indices[labels[indices] == cls].copy()
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(np.repeat(alpha, clients))
        cuts = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
        for client, shard in enumerate(np.split(cls_idx, cuts)):
            parts[client].extend(shard.tolist())
    # A hospital needs at least one example in each requested split to report a metric.
    empty = [client for client, part in enumerate(parts) if not part]
    for client in empty:
        donor = max(range(clients), key=lambda candidate: len(parts[candidate]))
        if len(parts[donor]) < 2:
            raise ValueError("Not enough examples to allocate a non-empty split to every client")
        parts[client].append(parts[donor].pop())
    return [np.array(sorted(part), dtype=int) for part in parts]


def make_federated_splits(frame: pd.DataFrame, clients: int, alpha: float, train: float, val: float, seed: int):
    """Create non-IID hospital train/validation/test subsets plus combined indices.

    Patients are separated before allocation, so no patient can appear in a
    hospital's training and evaluation partitions. Hospital membership is a
    simulation unless the input manifest supplies real institution identifiers.
    """
    train_idx, val_idx, test_idx = _split_indices(frame, train, val, seed)
    rng = np.random.default_rng(seed)
    labels = frame.target.to_numpy()
    client_train = _dirichlet_partition(train_idx, labels, clients, alpha, rng)
    client_val = _dirichlet_partition(val_idx, labels, clients, alpha, rng)
    client_test = _dirichlet_partition(test_idx, labels, clients, alpha, rng)
    return client_train, client_val, client_test, val_idx, test_idx


def transforms_for(image_size: int, training: bool, strong_augmentation: bool = False):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if training:
        augmentations = [
            transforms.Resize(int(image_size * 1.12), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15), transforms.ColorJitter(0.12, 0.12, 0.08, 0.03),
        ]
        # Optional robustness transforms are deliberately conservative. Keep them
        # off by default because overly synthetic lesion images can hurt calibration.
        if strong_augmentation:
            augmentations.extend([
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
                transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.12)], p=0.10),
            ])
        augmentations.extend([transforms.ToTensor(), normalize])
        if strong_augmentation:
            augmentations.append(transforms.RandomErasing(p=0.10, scale=(0.01, 0.04), ratio=(0.5, 2.0)))
        return transforms.Compose(augmentations)
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.12), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size), transforms.ToTensor(), normalize,
    ])


class LesionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, indices: np.ndarray, transform):
        self.frame, self.indices, self.transform = frame, indices, transform

    def __len__(self): return len(self.indices)

    def __getitem__(self, item):
        row = self.frame.iloc[int(self.indices[item])]
        with Image.open(row.image_path) as image:
            return self.transform(image.convert("RGB")), int(row.target)
