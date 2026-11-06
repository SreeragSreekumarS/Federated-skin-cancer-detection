from __future__ import annotations
from collections import OrderedDict
import copy, math, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
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


class ModelEMA:
    """Exponential moving average of trainable weights, with current model buffers."""
    def __init__(self, model: nn.Module, decay: float):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_parameters = dict(self.ema.named_parameters())
        for name, parameter in model.named_parameters():
            ema_parameters[name].mul_(self.decay).add_(parameter.detach(), alpha=1 - self.decay)
        # Copy BatchNorm running statistics and integer counters directly.
        ema_buffers = dict(self.ema.named_buffers())
        for name, buffer in model.named_buffers():
            ema_buffers[name].copy_(buffer)


def optimizer_parameter_groups(model: nn.Module, config, head_learning_rate: float):
    """Use a higher classifier-head LR and layer-wise decay through features."""
    scale = head_learning_rate / config.learning_rate
    groups = []
    feature_layers = list(model.features.children())
    last_layer = max(0, len(feature_layers) - 1)
    for layer_index, layer in enumerate(feature_layers):
        parameters = [parameter for parameter in layer.parameters() if parameter.requires_grad]
        if parameters:
            layer_rate = config.backbone_learning_rate * scale * (
                config.layerwise_lr_decay ** (last_layer - layer_index)
            )
            groups.append({"params": parameters, "lr": layer_rate})
    head_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("features.") and parameter.requires_grad
    ]
    if head_parameters:
        groups.append({"params": head_parameters, "lr": head_learning_rate})
    return groups


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


def make_loader(frame, indices, image_size, batch_size, workers, training, seed, strong_augmentation=False,
                balanced_sampling=False, cache_resized=False):
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if training and balanced_sampling:
        targets = frame.iloc[indices].target.to_numpy()
        counts = np.bincount(targets, minlength=int(frame.target.max()) + 1)
        sample_weights = 1.0 / np.sqrt(np.maximum(counts[targets], 1))
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(indices), replacement=True, generator=generator,
        )
    return DataLoader(LesionDataset(
        frame, indices,
        transforms_for(image_size, training, strong_augmentation, already_resized=cache_resized),
    ), batch_size=batch_size,
                      shuffle=training and sampler is None, sampler=sampler,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0, generator=generator)


def mixup_cutmix(images: torch.Tensor, targets: torch.Tensor, config):
    """Apply one batch-level MixUp or CutMix transform when selected."""
    available = [name for name, alpha in (
        ("mixup", config.mixup_alpha), ("cutmix", config.cutmix_alpha)
    ) if alpha > 0]
    if len(images) < 2 or not available or torch.rand(()) >= config.mixup_cutmix_probability:
        return images, targets, None, 1.0
    method = available[int(torch.randint(len(available), ()).item())]
    alpha = config.mixup_alpha if method == "mixup" else config.cutmix_alpha
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(len(images), device=images.device)
    if method == "mixup":
        return images * lam + images[permutation] * (1 - lam), targets, targets[permutation], lam

    height, width = images.shape[-2:]
    cut_ratio = np.sqrt(1 - lam)
    cut_width, cut_height = int(width * cut_ratio), int(height * cut_ratio)
    center_x = int(torch.randint(width, ()).item())
    center_y = int(torch.randint(height, ()).item())
    x1, x2 = max(0, center_x - cut_width // 2), min(width, center_x + cut_width // 2)
    y1, y2 = max(0, center_y - cut_height // 2), min(height, center_y + cut_height // 2)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (width * height))
    return mixed, targets, targets[permutation], lam


def train_client(global_model, frame, indices, val_indices, config, device, class_weights, round_number, learning_rate=None):
    model = copy.deepcopy(global_model).to(device)
    frozen = round_number <= config.freeze_backbone_rounds
    for parameter in model.features.parameters(): parameter.requires_grad = not frozen
    effective_lr = learning_rate or config.learning_rate
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, config, effective_lr),
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.local_epochs))
    # Balanced sampling and inverse-frequency loss weights compound each other;
    # use neutral class weights when batches are rebalanced.
    loss_weights = torch.ones_like(class_weights) if config.balanced_sampling else class_weights
    criterion = FocalCrossEntropy(loss_weights.to(device), config.focal_gamma, config.label_smoothing)
    ema = ModelEMA(model, config.ema_decay)
    reference = {n: p.detach().clone() for n, p in model.named_parameters()} if config.fedprox_mu else {}
    # The scaler is disabled automatically when local training runs on CPU.
    scaler = torch.amp.GradScaler(device.type, enabled=config.amp and device.type == "cuda")
    loader = make_loader(frame, indices, config.image_size, config.batch_size, config.num_workers, True,
                         config.seed + round_number, config.strong_augmentation, config.balanced_sampling,
                         config.cache_resized_images)
    model.train(); optimizer.zero_grad(set_to_none=True)
    train_loss_sum, train_correct, train_samples = 0.0, 0, 0
    for _ in range(config.local_epochs):
        for step, (images, targets) in enumerate(loader, 1):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            images, targets_a, targets_b, mix_lambda = mixup_cutmix(images, targets, config)
            with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                logits = model(images)
                task_loss = criterion(logits, targets_a)
                if targets_b is not None:
                    task_loss = mix_lambda * task_loss + (1 - mix_lambda) * criterion(logits, targets_b)
                loss = task_loss
                if reference:
                    prox = sum((p - reference[n]).pow(2).sum() for n, p in model.named_parameters())
                    loss = loss + 0.5 * config.fedprox_mu * prox
                loss = loss / config.gradient_accumulation
            batch_size = targets.size(0)
            train_loss_sum += float(task_loss.detach()) * batch_size
            predictions = logits.detach().argmax(dim=1)
            train_correct += int((predictions == targets_a).sum()) if targets_b is None else int(
                mix_lambda * (predictions == targets_a).sum() + (1 - mix_lambda) * (predictions == targets_b).sum()
            )
            train_samples += batch_size
            scaler.scale(loss).backward()
            if step % config.gradient_accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                ema.update(model)
        scheduler.step()
    local_metrics, _ = evaluate(ema.ema, frame, val_indices, config, device, [str(x) for x in sorted(frame.label.astype(str).unique())])
    quality = local_metrics["macro_f1"]
    local_metrics["train_accuracy"] = train_correct / max(1, train_samples)
    local_metrics["train_loss"] = train_loss_sum / max(1, train_samples)
    update = state_cpu(model)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return update, len(indices), quality, local_metrics


@torch.inference_mode()
def evaluate(model, frame, indices, config, device, class_names, return_probabilities=False):
    loader = make_loader(
        frame, indices, config.image_size, config.batch_size, config.num_workers,
        False, config.seed, cache_resized=config.cache_resized_images,
    )
    model.eval(); probs, targets = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
            logits = model(images)
            if config.tta:
                logits = (logits + model(torch.flip(images, dims=[3]))) / 2
        probs.append(torch.softmax(logits, 1).float().cpu().numpy()); targets.extend(labels.numpy().tolist())
    probabilities = np.concatenate(probs)
    metrics, matrix = classification_metrics(targets, probabilities, class_names)
    if return_probabilities:
        return metrics, matrix, np.asarray(targets, dtype=int), probabilities
    return metrics, matrix
