from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import yaml


@dataclass
class TrainConfig:
    seed: int = 42
    run_name: str = "gtx1650_federated_mobilenetv3"
    clients: int = 3
    dirichlet_alpha: float = 0.5
    rounds: int = 30
    client_fraction: float = 1.0
    local_epochs: int = 2
    batch_size: int = 16
    gradient_accumulation: int = 2
    num_workers: int = 2
    image_size: int = 160
    use_cbam: bool = True
    classifier_hidden_dim: int = 256
    classifier_dropout: float = 0.30
    learning_rate: float = 3e-4
    lr_patience: int = 3
    lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    fedprox_mu: float = 1e-3
    focal_gamma: float = 1.5
    label_smoothing: float = 0.05
    early_stopping_patience: int = 8
    freeze_backbone_rounds: int = 2
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    tta: bool = True
    amp: bool = True
    strong_augmentation: bool = False

    def validate(self) -> None:
        if self.clients < 2 or self.rounds < 1 or self.batch_size < 1:
            raise ValueError("clients must be >= 2; rounds and batch_size must be positive")
        if self.dirichlet_alpha <= 0 or not 0 < self.client_fraction <= 1:
            raise ValueError("dirichlet_alpha must be positive and client_fraction must be in (0, 1]")
        if self.classifier_hidden_dim < 1 or not 0 <= self.classifier_dropout < 1:
            raise ValueError("classifier_hidden_dim must be positive and classifier_dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.min_learning_rate <= 0 or not 0 < self.lr_factor < 1 or self.lr_patience < 1:
            raise ValueError("learning-rate settings must be positive, lr_factor must be in (0, 1), and lr_patience must be positive")
        if abs(self.train_fraction + self.val_fraction + self.test_fraction - 1) > 1e-6:
            raise ValueError("train_fraction + val_fraction + test_fraction must equal 1")


def load_config(path: str | Path) -> TrainConfig:
    with open(path, encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    config = TrainConfig(**values)
    config.validate()
    return config


def save_config(config: TrainConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(asdict(config), stream, sort_keys=False)
