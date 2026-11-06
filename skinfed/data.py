from __future__ import annotations

from pathlib import Path
import random
from itertools import combinations
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
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
    except ValueError as error:
        # Never fall back to row-level splitting: that can put images from the
        # same patient/lesion in different partitions. For small clients use
        # group-aware random splits, and fail clearly if three partitions cannot
        # be formed from the available groups.
        try:
            outer = GroupShuffleSplit(n_splits=1, train_size=train, random_state=seed)
            train_idx, holdout_idx = next(outer.split(indices, labels, groups))
            inner = GroupShuffleSplit(n_splits=1, train_size=val / (1 - train), random_state=seed + 1)
            val_rel, test_rel = next(inner.split(holdout_idx, labels[holdout_idx], groups[holdout_idx]))
        except ValueError as grouped_error:
            raise ValueError(
                "Cannot create train/validation/test partitions without splitting a patient/lesion group. "
                "Provide more distinct patient_id values or adjust the split fractions."
            ) from grouped_error
        return train_idx, holdout_idx[val_rel], holdout_idx[test_rel]


def _assign_groups_to_hospitals(frame: pd.DataFrame, clients: int, alpha: float, seed: int) -> list[np.ndarray]:
    """Assign every patient/lesion group to exactly one simulated hospital."""
    rng = np.random.default_rng(seed)
    labels = frame.target.to_numpy()
    group_values = frame.patient_id.astype(str).to_numpy()
    group_frame = pd.DataFrame({"group": group_values, "label": labels})
    # A group's majority label drives non-IID allocation; tie-breaking is stable.
    group_labels = group_frame.groupby("group")["label"].agg(
        lambda values: int(values.value_counts().sort_index().idxmax())
    )
    hospitals: list[list[str]] = [[] for _ in range(clients)]
    for cls in sorted(group_labels.unique()):
        groups = group_labels.index[group_labels.to_numpy() == cls].to_numpy().copy()
        rng.shuffle(groups)
        proportions = rng.dirichlet(np.repeat(alpha, clients))
        cuts = (np.cumsum(proportions)[:-1] * len(groups)).astype(int)
        for hospital, shard in enumerate(np.split(groups, cuts)):
            hospitals[hospital].extend(shard.tolist())
    # Ensure each simulated hospital has groups available for downstream splits.
    empty = [hospital for hospital, groups in enumerate(hospitals) if not groups]
    for hospital in empty:
        donor = max(range(clients), key=lambda candidate: len(hospitals[candidate]))
        if len(hospitals[donor]) < 4:
            raise ValueError("Not enough patient/lesion groups to assign every hospital and create three splits")
        hospitals[hospital].append(hospitals[donor].pop())
    return [np.flatnonzero(np.isin(group_values, groups)) for groups in hospitals]


def make_federated_splits(frame: pd.DataFrame, clients: int, alpha: float, train: float, val: float, seed: int):
    """Create non-IID hospital train/validation/test subsets plus combined indices.

    Each patient/lesion group is assigned to one hospital first. Train,
    validation, and test are then split independently within each hospital.
    Hospital membership is simulated unless the manifest supplies institution IDs.
    """
    hospital_indices = _assign_groups_to_hospitals(frame, clients, alpha, seed)
    client_train, client_val, client_test = [], [], []
    for hospital, indices in enumerate(hospital_indices):
        hospital_frame = frame.iloc[indices].reset_index(drop=True)
        local_train, local_val, local_test = _split_indices(hospital_frame, train, val, seed + hospital)
        client_train.append(indices[local_train])
        client_val.append(indices[local_val])
        client_test.append(indices[local_test])
    val_indices = np.concatenate(client_val)
    test_indices = np.concatenate(client_test)
    return client_train, client_val, client_test, val_indices, test_indices


class ResizeWithPad:
    """Resize the complete image to fit a square, padding without cropping or stretching."""
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.pad(
            image,
            (self.image_size, self.image_size),
            method=Image.Resampling.BICUBIC,
            color=(123, 116, 103),
            centering=(0.5, 0.5),
        )


def transforms_for(image_size: int, training: bool, strong_augmentation: bool = False):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if training:
        augmentations = [
            ResizeWithPad(image_size),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15, fill=(123, 116, 103)),
            transforms.ColorJitter(0.12, 0.12, 0.08, 0.03),
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
        ResizeWithPad(image_size), transforms.ToTensor(), normalize,
    ])


class LesionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, indices: np.ndarray, transform):
        self.frame, self.indices, self.transform = frame, indices, transform

    def __len__(self): return len(self.indices)

    def __getitem__(self, item):
        row = self.frame.iloc[int(self.indices[item])]
        with Image.open(row.image_path) as image:
            return self.transform(image.convert("RGB")), int(row.target)
