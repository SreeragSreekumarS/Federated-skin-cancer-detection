"""Run one tiny federated round on generated images; no medical data is used or retained."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from skinfed.config import TrainConfig
from skinfed.data import make_federated_splits, read_manifest
from skinfed.federated import aggregate, evaluate, train_client
from skinfed.model import build_model


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainConfig(clients=3, dirichlet_alpha=8.0, rounds=1, local_epochs=1,
                         batch_size=4, gradient_accumulation=1, num_workers=0,
                         image_size=96, early_stopping_patience=1, freeze_backbone_rounds=1,
                         amp=device.type == "cuda", tta=False)
    with tempfile.TemporaryDirectory(prefix="skinfed-smoke-") as temporary:
        root = Path(temporary)
        records = []
        # 45 patient-distinct images: enough samples for every split/client.
        for label, base in enumerate((45, 125, 205)):
            for sample in range(15):
                noise = np.random.default_rng(label * 100 + sample).integers(-18, 19, (112, 112, 3), dtype=np.int16)
                image = np.clip(base + noise, 0, 255).astype(np.uint8)
                path = root / f"class_{label}_{sample}.png"
                Image.fromarray(image).save(path)
                records.append({"image_path": str(path), "label": f"class_{label}", "patient_id": f"p_{label}_{sample}"})
        manifest = root / "manifest.csv"
        pd.DataFrame(records).to_csv(manifest, index=False)
        frame, classes = read_manifest(manifest)
        client_indices, client_val_indices, client_test_indices, val_indices, test_indices = make_federated_splits(
            frame, config.clients, config.dirichlet_alpha, config.train_fraction, config.val_fraction, config.seed
        )
        counts = frame.iloc[np.concatenate(client_indices)].target.value_counts().reindex(range(len(classes)), fill_value=1).to_numpy()
        weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
        model = build_model(len(classes), pretrained=False, use_cbam=config.use_cbam).to(device)
        updates = [
            train_client(model, frame, indices, client_val_indices[i], config, device, weights, 1)
            for i, indices in enumerate(client_indices)
        ]
        aggregate(model, updates)
        metrics, _ = evaluate(model, frame, test_indices, config, device, classes)
        print("SMOKE TEST PASSED")
        print(f"Device: {device}")
        print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
