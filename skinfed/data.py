from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
import threading
import warnings
from pathlib import Path
import random
from itertools import combinations
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms


REQUIRED_COLUMNS = {"image_path", "label"}
SEX_VALUES = ("female", "male", "unknown")
SITE_VALUES = (
    "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital",
    "hand", "head_neck", "lower_extremity", "neck", "oral_genital",
    "palms_soles", "scalp", "torso", "trunk", "upper_extremity", "unknown",
)
METADATA_FEATURE_DIM = 2 + len(SEX_VALUES) + len(SITE_VALUES)


class RandomHairRemoval:
    """DullRazor-style black-hat detection and inpainting, applied stochastically."""
    def __init__(self, probability: float = 0.0):
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.probability <= 0 or random.random() >= self.probability:
            return image
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Hair removal requires opencv-python-headless; install the updated requirements.txt."
            ) from exc
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        hair = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(hair, 10, 255, cv2.THRESH_BINARY)
        cleaned = cv2.inpaint(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, 2, cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))


class GrayWorldColorConstancy:
    """Normalize per-channel means using the Gray World assumption."""
    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        channel_means = rgb.mean(axis=(0, 1))
        nonzero = channel_means > 1e-6
        if not nonzero.any():
            return image
        target = float(channel_means[nonzero].mean())
        gains = np.ones(3, dtype=np.float32)
        gains[nonzero] = target / channel_means[nonzero]
        corrected = np.clip(rgb * gains[None, None, :], 0, 255).astype(np.uint8)
        return Image.fromarray(corrected, mode="RGB")


def encode_metadata(frame: pd.DataFrame) -> np.ndarray:
    """Encode only age, sex, and anatomical site with fixed vocabularies."""
    missing = {"age", "sex", "site"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Metadata training requires manifest columns {sorted(missing)}; "
            "rebuild the manifest with the dataset metadata option."
        )
    age = pd.to_numeric(frame["age"], errors="coerce").to_numpy(dtype=np.float32)
    age_missing = ~np.isfinite(age)
    age = np.clip(np.where(age_missing, 0.0, age) / 100.0, 0.0, 1.0)
    features = [age[:, None], age_missing.astype(np.float32)[:, None]]

    def normalize(value) -> str:
        if pd.isna(value) or not str(value).strip():
            return "unknown"
        return str(value).strip().lower().replace(" ", "_").replace("-", "_")

    sex = np.asarray([normalize(value) for value in frame["sex"]])
    sex = np.where(np.isin(sex, SEX_VALUES), sex, "unknown")
    features.append(np.stack([sex == name for name in SEX_VALUES], axis=1).astype(np.float32))

    site = np.asarray([normalize(value) for value in frame["site"]])
    aliases = {
        "head/neck": "head_neck", "oral/genital": "oral_genital",
        "palms/soles": "palms_soles", "lower_extremity": "lower_extremity",
        "upper_extremity": "upper_extremity",
    }
    site = np.asarray([aliases.get(value, value) for value in site])
    site = np.where(np.isin(site, SITE_VALUES), site, "unknown")
    features.append(np.stack([site == name for name in SITE_VALUES], axis=1).astype(np.float32))
    return np.concatenate(features, axis=1)


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
        row_support = pd.Series(labels).value_counts().min()
        group_support = pd.DataFrame({"group": groups, "label": labels}).groupby("label")["group"].nunique().min()
        n_splits = min(10, int(row_support), int(group_support))
        # Three or more folds preserve the requested 70/15/15 proportions better.
        # Smaller rare-class counts use the group-only fallback below without
        # asking sklearn to make impossible stratified folds.
        if n_splits < 3:
            raise ValueError("Too few groups per class for stratified grouped folds")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
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
            (np.concatenate([folds[i] for i in choice]) for choice in combinations(
                range(n_splits), max(1, round(target_holdout * n_splits))
            )),
            key=lambda candidate: split_cost(candidate, target_holdout),
        )
        train_idx = np.setdiff1d(indices, holdout_idx, assume_unique=False)
        holdout_labels = labels[holdout_idx]
        holdout_groups = groups[holdout_idx]
        val_ratio = val / (1.0 - train)
        inner_row_support = pd.Series(holdout_labels).value_counts().min()
        inner_group_support = pd.DataFrame({"group": holdout_groups, "label": holdout_labels}).groupby("label")["group"].nunique().min()
        if min(int(inner_row_support), int(inner_group_support)) >= 2:
            inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed + 1)
            inner_splits = list(inner.split(holdout_idx, holdout_labels, holdout_groups))
        else:
            inner = GroupShuffleSplit(n_splits=2, train_size=val_ratio, random_state=seed + 1)
            inner_splits = list(inner.split(holdout_idx, holdout_labels, holdout_groups))

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


def _assign_groups_to_hospitals(
    frame: pd.DataFrame, clients: int, alpha: float, seed: int,
    min_groups_per_class_per_hospital: int = 3,
) -> list[np.ndarray]:
    """Assign every patient/lesion group to exactly one simulated hospital."""
    rng = np.random.default_rng(seed)
    labels = frame.target.to_numpy()
    group_values = frame.patient_id.astype(str).to_numpy()
    group_frame = pd.DataFrame({"group": group_values, "label": labels})
    # A group's majority label drives non-IID allocation; tie-breaking is stable.
    group_labels = group_frame.groupby("group")["label"].agg(
        lambda values: int(values.value_counts().sort_index().idxmax())
    )
    class_names = sorted(frame.label.astype(str).unique().tolist())
    hospitals: list[list[str]] = [[] for _ in range(clients)]
    for cls in sorted(group_labels.unique()):
        groups = group_labels.index[group_labels.to_numpy() == cls].to_numpy().copy()
        rng.shuffle(groups)
        proportions = rng.dirichlet(np.repeat(alpha, clients))
        if len(groups) < clients * min_groups_per_class_per_hospital:
            warnings.warn(
                f"Class {class_names[int(cls)]} "
                f"has {len(groups)} lesion/patient groups; cannot meet the requested minimum "
                f"of {min_groups_per_class_per_hospital} per hospital across {clients} hospitals.",
                RuntimeWarning,
                stacklevel=2,
            )
        feasible_minimum = min(min_groups_per_class_per_hospital, len(groups) // clients)
        counts = np.full(clients, feasible_minimum, dtype=int)
        remaining = len(groups) - int(counts.sum())
        if remaining:
            counts += rng.multinomial(remaining, proportions)
        cuts = np.cumsum(counts)[:-1]
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


def make_federated_splits(
    frame: pd.DataFrame, clients: int, alpha: float, train: float, val: float, seed: int,
    min_groups_per_class_per_hospital: int = 3,
):
    """Create non-IID hospital train/validation/test subsets plus combined indices.

    Each patient/lesion group is assigned to one hospital first. Train,
    validation, and test are then split independently within each hospital.
    Hospital membership is simulated unless the manifest supplies institution IDs.
    """
    hospital_indices = _assign_groups_to_hospitals(
        frame, clients, alpha, seed, min_groups_per_class_per_hospital
    )
    client_train, client_val, client_test = [], [], []
    for hospital, indices in enumerate(hospital_indices):
        hospital_frame = frame.iloc[indices].reset_index(drop=True)
        local_train, local_val, local_test = _split_indices(hospital_frame, train, val, seed + hospital)
        client_train.append(indices[local_train])
        client_val.append(indices[local_val])
        client_test.append(indices[local_test])
    assigned_rows = np.concatenate([
        part for parts in zip(client_train, client_val, client_test) for part in parts
    ])
    if len(assigned_rows) != len(frame) or len(np.unique(assigned_rows)) != len(frame):
        raise RuntimeError("Hospital-first partition did not assign each manifest row exactly once")
    group_membership: dict[str, tuple[int, str]] = {}
    for hospital, parts in enumerate(zip(client_train, client_val, client_test)):
        for split_name, indices in zip(("train", "validation", "test"), parts):
            for group in frame.iloc[indices].patient_id.astype(str).unique():
                previous = group_membership.setdefault(group, (hospital, split_name))
                if previous != (hospital, split_name):
                    raise RuntimeError(f"Patient/lesion group {group!r} crosses a hospital or split boundary")

    group_labels = frame.groupby(frame.patient_id.astype(str))["label"].agg(
        lambda values: str(values.astype(str).value_counts().sort_index().idxmax())
    )
    class_names = sorted(frame.label.astype(str).unique().tolist())
    rows = []
    for hospital, parts in enumerate(zip(client_train, client_val, client_test), 1):
        for split_name, indices in zip(("train", "validation", "test"), parts):
            groups = frame.iloc[indices].patient_id.astype(str).unique()
            counts = group_labels.reindex(groups).value_counts()
            rows.append({
                "hospital": f"hospital_{hospital}", "split": split_name,
                **{name: int(counts.get(name, 0)) for name in class_names},
            })
    class_table = pd.DataFrame(rows).set_index(["hospital", "split"])
    print("Hospital × split × class counts (patient/lesion groups):")
    print(class_table.to_string())
    missing = [
        f"{hospital}/{split}/{class_name}"
        for (hospital, split), values in class_table.iterrows()
        for class_name, count in values.items() if count == 0
    ]
    if missing:
        warnings.warn(
            "Some hospital splits have no groups for a class: " + ", ".join(missing),
            RuntimeWarning,
            stacklevel=2,
        )
    val_indices = np.concatenate(client_val)
    test_indices = np.concatenate(client_test)
    return client_train, client_val, client_test, val_indices, test_indices


def make_image_level_federated_splits(
    frame: pd.DataFrame, clients: int, alpha: float, train: float, val: float, seed: int,
):
    """Paper-style image-level client allocation and local random splits.

    This intentionally ignores patient/lesion groups and can leak images from
    one lesion across train/validation/test. Use only as a labeled replication.
    """
    rng = np.random.default_rng(seed)
    labels = frame.target.to_numpy()
    client_rows: list[list[int]] = [[] for _ in range(clients)]
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        rng.shuffle(rows)
        counts = rng.multinomial(len(rows), rng.dirichlet(np.repeat(alpha, clients)))
        for client, shard in enumerate(np.split(rows, np.cumsum(counts)[:-1])):
            client_rows[client].extend(shard.tolist())

    client_train, client_val, client_test = [], [], []
    for client, rows in enumerate(client_rows):
        rows = np.asarray(rows, dtype=int)
        client_labels = labels[rows]
        try:
            train_rows, holdout_rows = train_test_split(
                rows, train_size=train, random_state=seed + client,
                stratify=client_labels,
            )
            holdout_labels = labels[holdout_rows]
            val_share = val / (1.0 - train)
            val_rows, test_rows = train_test_split(
                holdout_rows, train_size=val_share, random_state=seed + client + 1000,
                stratify=holdout_labels,
            )
        except ValueError:
            # Rare classes can make stratification impossible in small client shards.
            train_rows, holdout_rows = train_test_split(
                rows, train_size=train, random_state=seed + client,
            )
            val_rows, test_rows = train_test_split(
                holdout_rows, train_size=val / (1.0 - train), random_state=seed + client + 1000,
            )
        client_train.append(np.asarray(train_rows, dtype=int))
        client_val.append(np.asarray(val_rows, dtype=int))
        client_test.append(np.asarray(test_rows, dtype=int))

    val_indices = np.concatenate(client_val)
    test_indices = np.concatenate(client_test)
    print("WARNING: image-level replication split; lesion groups may cross train/validation/test.")
    return client_train, client_val, client_test, val_indices, test_indices


def make_source_splits(
    frame: pd.DataFrame, train: float, val: float, seed: int,
    source_order: tuple[str, ...] = ("BCN", "MSK"),
):
    """Split natural ISIC source clients by lesion group, then split each client."""
    if "source" not in frame:
        raise ValueError("partition_mode=source requires a source column in the ISIC training manifest")
    observed = set(frame.source.fillna("").astype(str))
    if observed != set(source_order):
        raise ValueError(
            "partition_mode=source expects only BCN and MSK training data after HAM exclusion; "
            f"found {sorted(observed)}"
        )
    if not 0 < train < 1 or not 0 < val < 1 or train + val >= 1:
        raise ValueError("Train and validation fractions must be positive and leave a non-empty test split")

    client_train: list[np.ndarray] = []
    client_val: list[np.ndarray] = []
    client_test: list[np.ndarray] = []
    group_owners: dict[str, str] = {}
    for client_index, source in enumerate(source_order):
        indices = np.flatnonzero(frame.source.astype(str).to_numpy() == source)
        if not len(indices):
            raise ValueError(f"Natural source client {source} has no images")
        subset = frame.iloc[indices].reset_index(drop=True)
        local_train, local_val, local_test = _split_indices(subset, train, val, seed + client_index)
        client_train.append(indices[local_train])
        client_val.append(indices[local_val])
        client_test.append(indices[local_test])
        for group in subset.patient_id.astype(str).unique():
            previous = group_owners.setdefault(group, source)
            if previous != source:
                raise RuntimeError(f"Lesion group {group!r} appears in multiple source clients")

    all_parts = [part for client_parts in zip(client_train, client_val, client_test) for part in client_parts]
    assigned = np.concatenate(all_parts)
    if len(assigned) != len(frame) or len(np.unique(assigned)) != len(frame):
        raise RuntimeError("Source partition did not assign every training image exactly once")

    class_names = sorted(frame.label.astype(str).unique().tolist())
    group_labels = frame.groupby(frame.patient_id.astype(str))["label"].agg(
        lambda values: str(values.astype(str).value_counts().sort_index().idxmax())
    )
    rows = []
    missing = []
    for source_index, source in enumerate(source_order):
        for split_name, indices in zip(
            ("train", "validation", "test"),
            (client_train[source_index], client_val[source_index], client_test[source_index]),
        ):
            groups = frame.iloc[indices].patient_id.astype(str).unique()
            counts = group_labels.reindex(groups).value_counts()
            values = {name: int(counts.get(name, 0)) for name in class_names}
            rows.append({"source": source, "split": split_name, **values})
            missing.extend(f"{source}/{split_name}/{name}" for name, count in values.items() if count == 0)
    table = pd.DataFrame(rows).set_index(["source", "split"])
    print("Natural source × split × class counts (lesion groups):")
    print(table.to_string())
    if missing:
        warnings.warn(
            "Some source splits have no lesion groups for a class: " + ", ".join(missing),
            RuntimeWarning,
            stacklevel=2,
        )
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


def cache_resized_images(frame: pd.DataFrame, image_size: int, cache_dir: str | Path) -> pd.DataFrame:
    """Create reusable square-resized JPEGs and return a frame pointing at them."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_to_cache: dict[str, str] = {}
    jobs: dict[str, tuple[Path, Path]] = {}
    for value in frame.image_path.astype(str).unique():
        source = Path(value).resolve()
        stat = source.stat()
        key_source = f"{source}|{stat.st_size}|{stat.st_mtime_ns}|{image_size}"
        key = hashlib.sha256(key_source.encode("utf-8")).hexdigest()
        cached = cache_dir / f"{key}.jpg"
        source_to_cache[value] = str(cached)
        if not cached.is_file():
            jobs[str(cached)] = (source, cached)

    def resize_one(source: Path, cached: Path) -> None:
        with Image.open(source) as image:
            resized = ResizeWithPad(image_size)(image.convert("RGB"))
            temporary = cached.with_name(
                f"{cached.stem}.tmp-{os.getpid()}-{threading.get_ident()}.jpg"
            )
            resized.save(temporary, format="JPEG", quality=95)
            temporary.replace(cached)

    if jobs:
        workers = min(8, max(1, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(resize_one, source, cached) for source, cached in jobs.values()]
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                if completed % 1000 == 0 or completed == len(futures):
                    print(f"Resized cache: {completed}/{len(futures)} images")
    result = frame.copy()
    if "source_image_path" not in result:
        result["source_image_path"] = result.image_path.astype(str)
    result["image_path"] = result.image_path.astype(str).map(source_to_cache)
    return result


def transforms_for(
    image_size: int,
    training: bool,
    strong_augmentation: bool = False,
    already_resized: bool = False,
    image_enhancement: bool = False,
    hair_removal_probability: float = 0.0,
    color_constancy: bool = False,
    normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    normalize = transforms.Normalize(normalization_mean, normalization_std)
    preprocessing = [] if already_resized else [ResizeWithPad(image_size)]
    if color_constancy:
        preprocessing.append(GrayWorldColorConstancy())
    if training:
        augmentations = [
            *preprocessing,
            RandomHairRemoval(hair_removal_probability),
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
        # Small, randomized color/contrast corrections for training only.
        # Validation and test keep the original image distribution.
        if image_enhancement:
            augmentations.extend([
                transforms.RandomApply([transforms.RandomAutocontrast()], p=0.10),
                transforms.RandomApply([transforms.RandomAdjustSharpness(1.25)], p=0.10),
                transforms.RandomApply([transforms.RandomEqualize()], p=0.05),
            ])
        augmentations.extend([transforms.ToTensor(), normalize])
        if strong_augmentation:
            augmentations.append(transforms.RandomErasing(p=0.10, scale=(0.01, 0.04), ratio=(0.5, 2.0)))
        return transforms.Compose(augmentations)
    return transforms.Compose([*preprocessing, transforms.ToTensor(), normalize])


class LesionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, indices: np.ndarray, transform, use_metadata: bool = False):
        self.frame, self.indices, self.transform = frame, indices, transform
        self.metadata = encode_metadata(frame) if use_metadata else None

    def __len__(self): return len(self.indices)

    def __getitem__(self, item):
        row = self.frame.iloc[int(self.indices[item])]
        with Image.open(row.image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        if self.metadata is not None:
            return image_tensor, torch.from_numpy(self.metadata[int(self.indices[item])]), int(row.target)
        return image_tensor, int(row.target)
