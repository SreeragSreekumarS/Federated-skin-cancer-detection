from __future__ import annotations
from collections import OrderedDict
import copy, math, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data import LesionDataset, transforms_for
from .metrics import classification_metrics


class FocalCrossEntropy(nn.Module):
    def __init__(self, weights: torch.Tensor, gamma: float, label_smoothing: float):
        super().__init__(); self.register_buffer("weights", weights); self.gamma = gamma; self.smoothing = label_smoothing
    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weights, reduction="none", label_smoothing=self.smoothing)
        return ((1 - torch.exp(-ce)).pow(self.gamma) * ce).mean()


def state_cpu(model: nn.Module) -> OrderedDict:
    return OrderedDict((key, value.detach().cpu().clone()) for key, value in model.state_dict().items())


def aggregate(global_model: nn.Module, updates: list[tuple[OrderedDict, int, float, dict]]) -> None:
    # Weight client updates by data volume and bounded validation quality, preventing a tiny client dominating.
    weights = np.array([samples * max(0.25, quality) for _, samples, quality, _ in updates], dtype=float)
    weights /= weights.sum()
    merged = OrderedDict()
    for key, initial in global_model.state_dict().items():
        if not torch.is_floating_point(initial):
            merged[key] = updates[0][0][key]
        else:
            merged[key] = sum(update[key].to(torch.float32) * float(weight) for (update, _, _, _), weight in zip(updates, weights)).to(initial.dtype)
    global_model.load_state_dict(merged, strict=True)


def make_loader(frame, indices, image_size, batch_size, workers, training, seed, strong_augmentation=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(LesionDataset(frame, indices, transforms_for(image_size, training, strong_augmentation)), batch_size=batch_size, shuffle=training,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0, generator=generator)


def train_client(global_model, frame, indices, val_indices, config, device, class_weights, round_number, learning_rate=None):
    model = copy.deepcopy(global_model).to(device)
    frozen = round_number <= config.freeze_backbone_rounds
    for parameter in model.features.parameters(): parameter.requires_grad = not frozen
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate or config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.local_epochs))
    criterion = FocalCrossEntropy(class_weights.to(device), config.focal_gamma, config.label_smoothing)
    reference = {n: p.detach().clone() for n, p in model.named_parameters()} if config.fedprox_mu else {}
    # cuda.amp is retained here for compatibility with PyTorch 2.2, which is
    # common on GTX 1650 laptops. It is disabled automatically on CPU.
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == "cuda")
    loader = make_loader(frame, indices, config.image_size, config.batch_size, config.num_workers, True, config.seed + round_number, config.strong_augmentation)
    model.train(); optimizer.zero_grad(set_to_none=True)
    for _ in range(config.local_epochs):
        for step, (images, targets) in enumerate(loader, 1):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                loss = criterion(model(images), targets)
                if reference:
                    prox = sum((p - reference[n]).pow(2).sum() for n, p in model.named_parameters())
                    loss = loss + 0.5 * config.fedprox_mu * prox
                loss = loss / config.gradient_accumulation
            scaler.scale(loss).backward()
            if step % config.gradient_accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    local_metrics, _ = evaluate(model, frame, val_indices, config, device, [str(x) for x in sorted(frame.label.astype(str).unique())])
    quality = local_metrics["macro_f1"]
    update = state_cpu(model)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return update, len(indices), quality, local_metrics


@torch.inference_mode()
def evaluate(model, frame, indices, config, device, class_names):
    loader = make_loader(frame, indices, config.image_size, config.batch_size, config.num_workers, False, config.seed,)
    model.eval(); probs, targets = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
            logits = model(images)
            if config.tta:
                logits = (logits + model(torch.flip(images, dims=[3]))) / 2
        probs.append(torch.softmax(logits, 1).float().cpu().numpy()); targets.extend(labels.numpy().tolist())
    return classification_metrics(targets, np.concatenate(probs), class_names)
