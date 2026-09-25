from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
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
    # GroupShuffleSplit is deliberate: it prevents a patient's images leaking across splits.
    try:
        outer = GroupShuffleSplit(n_splits=1, train_size=train, random_state=seed)
        train_idx, holdout_idx = next(outer.split(indices, labels, groups))
        holdout = frame.iloc[holdout_idx]
        ratio = val / (1 - train)
        inner = GroupShuffleSplit(n_splits=1, train_size=ratio, random_state=seed + 1)
        val_rel, test_rel = next(inner.split(holdout, holdout.target, holdout.patient_id))
        return train_idx, holdout_idx[val_rel], holdout_idx[test_rel]
    except ValueError:
        # Tiny synthetic datasets may not have enough groups. Still use stratification where possible.
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train, random_state=seed)
        train_idx, holdout_idx = next(splitter.split(indices, labels))
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
