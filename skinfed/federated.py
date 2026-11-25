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
from .model import freeze_features, model_backbone_layers


class FocalCrossEntropy(nn.Module):
    def __init__(self, weights: torch.Tensor, gamma: float, label_smoothing: float):
        super().__init__(); self.register_buffer("weights", weights); self.gamma = gamma; self.smoothing = label_smoothing
    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weights, reduction="none", label_smoothing=self.smoothing)
        if self.gamma == 0:
            return ce.mean()
        return ((1 - torch.exp(-ce)).pow(self.gamma) * ce).mean()


def configured_loss_weights(class_weights: torch.Tensor, config) -> torch.Tensor:
    if config.class_weighting == "none":
        return torch.ones_like(class_weights)
    if config.class_weighting == "inverse_frequency":
        return class_weights
    return torch.ones_like(class_weights) if config.balanced_sampling else class_weights


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
    backbone_layers = model_backbone_layers(model)
    backbone_ids = {id(parameter) for layer in backbone_layers for parameter in layer.parameters()}
    last_layer = max(0, len(backbone_layers) - 1)
    for layer_index, layer in enumerate(backbone_layers):
        parameters = [parameter for parameter in layer.parameters() if parameter.requires_grad]
        if parameters:
            layer_rate = config.backbone_learning_rate * scale * (
                config.layerwise_lr_decay ** (last_layer - layer_index)
            )
            groups.append({"params": parameters, "lr": layer_rate})
    head_parameters = [
        parameter for name, parameter in model.named_parameters()
        if id(parameter) not in backbone_ids and parameter.requires_grad
    ]
    if head_parameters:
        groups.append({"params": head_parameters, "lr": head_learning_rate})
    return groups


def aggregate(global_model: nn.Module, updates: list[tuple[OrderedDict, int, float, dict]], weighting: str = "samples",
              batchnorm_policy: str = "average", server_momentum: float = 0.0,
              momentum_state: dict[str, torch.Tensor] | None = None) -> None:
    """Aggregate local EMA states, optionally freezing BN stats and using FedAvgM."""
    if weighting not in {"samples", "quality"}:
        raise ValueError("weighting must be samples or quality")
    if batchnorm_policy not in {"average", "freeze_stats"}:
        raise ValueError("batchnorm_policy must be average or freeze_stats")
    if weighting == "samples":
        weights = np.array([samples for _, samples, _, _ in updates], dtype=float)
    else:
        weights = np.array([samples * max(0.25, quality) for _, samples, quality, _ in updates], dtype=float)
    weights /= weights.sum()
    merged = OrderedDict()
    for key, initial in global_model.state_dict().items():
        is_bn_stat = key.endswith(("running_mean", "running_var", "num_batches_tracked"))
        if batchnorm_policy == "freeze_stats" and is_bn_stat:
            merged[key] = initial
            continue
        if not torch.is_floating_point(initial):
            merged[key] = updates[0][0][key] if not is_bn_stat else initial
        else:
            merged[key] = sum(update[key].to(torch.float32) * float(weight) for (update, _, _, _), weight in zip(updates, weights)).to(initial.dtype)
    if server_momentum:
        if momentum_state is None:
            raise ValueError("momentum_state is required when server_momentum is enabled")
        for key, initial in global_model.state_dict().items():
            if not torch.is_floating_point(initial):
                continue
            if batchnorm_policy == "freeze_stats" and key.endswith(("running_mean", "running_var")):
                continue
            delta = merged[key].to(device=initial.device, dtype=torch.float32) - initial.to(torch.float32)
            velocity = momentum_state.get(key)
            velocity = torch.zeros_like(delta) if velocity is None else velocity.to(delta.device)
            velocity.mul_(server_momentum).add_(delta)
            momentum_state[key] = velocity.detach().cpu()
            merged[key] = (initial.to(torch.float32) + velocity).to(initial.dtype)
    global_model.load_state_dict(merged, strict=True)


def apply_batchnorm_policy(model: nn.Module, policy: str) -> None:
    """Keep BN affine parameters trainable while optionally fixing running stats."""
    if policy == "freeze_stats":
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def make_loader(frame, indices, image_size, batch_size, workers, training, seed, strong_augmentation=False,
                balanced_sampling=False, cache_resized=False, use_metadata=False, image_enhancement=False,
                hair_removal_probability=0.0, color_constancy=False, balanced_sampling_power=0.5):
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if training and balanced_sampling:
        targets = frame.iloc[indices].target.to_numpy()
        counts = np.bincount(targets, minlength=int(frame.target.max()) + 1)
        sample_weights = 1.0 / np.maximum(counts[targets], 1) ** balanced_sampling_power
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(indices), replacement=True, generator=generator,
        )
    return DataLoader(LesionDataset(
        frame, indices,
        transforms_for(image_size, training, strong_augmentation, already_resized=cache_resized,
                       image_enhancement=image_enhancement,
                       hair_removal_probability=hair_removal_probability if training else 0.0,
                       color_constancy=color_constancy),
        use_metadata=use_metadata,
    ), batch_size=batch_size,
                      shuffle=training and sampler is None, sampler=sampler,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0, generator=generator)


def mixup_cutmix(images: torch.Tensor, targets: torch.Tensor, config, metadata=None):
    """Apply one batch-level MixUp or CutMix transform when selected."""
    available = [name for name, alpha in (
        ("mixup", config.mixup_alpha), ("cutmix", config.cutmix_alpha)
    ) if alpha > 0]
    if len(images) < 2 or not available or torch.rand(()) >= config.mixup_cutmix_probability:
        return images, targets, None, 1.0, metadata
    method = available[int(torch.randint(len(available), ()).item())]
    alpha = config.mixup_alpha if method == "mixup" else config.cutmix_alpha
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(len(images), device=images.device)
    if method == "mixup":
        mixed_metadata = metadata
        if metadata is not None:
            mixed_metadata = metadata * lam + metadata[permutation] * (1 - lam)
        return images * lam + images[permutation] * (1 - lam), targets, targets[permutation], lam, mixed_metadata

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
    mixed_metadata = metadata
    if metadata is not None:
        mixed_metadata = metadata * lam + metadata[permutation] * (1 - lam)
    return mixed, targets, targets[permutation], lam, mixed_metadata


def train_client(global_model, frame, indices, val_indices, config, device, class_weights, round_number, learning_rate=None):
    model = copy.deepcopy(global_model).to(device)
    frozen = round_number <= config.freeze_backbone_rounds
    freeze_features(model, frozen, config.freeze_early_stages)
    effective_lr = learning_rate or config.learning_rate
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, config, effective_lr),
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.local_epochs))
    # Balanced sampling and inverse-frequency loss weights compound each other;
    # use neutral class weights when batches are rebalanced.
    loss_weights = configured_loss_weights(class_weights, config)
    criterion = FocalCrossEntropy(loss_weights.to(device), config.focal_gamma, config.label_smoothing)
    local_counts = np.bincount(
        frame.iloc[indices].target.to_numpy(), minlength=len(class_weights)
    ).astype(np.float64)
    logit_adjustment = None
    if config.client_logit_adjustment_strength > 0:
        # Laplace smoothing handles classes absent at a client. The adjustment
        # is used only in that client's training loss, not during evaluation.
        local_prior = (local_counts + 1.0) / (local_counts.sum() + len(local_counts))
        logit_adjustment = config.client_logit_adjustment_strength * torch.log(
            torch.as_tensor(local_prior, dtype=torch.float32, device=device)
        )
    ema = ModelEMA(model, config.ema_decay)
    reference = {n: p.detach().clone() for n, p in model.named_parameters()} if config.fedprox_mu else {}
    # The scaler is disabled automatically when local training runs on CPU.
    scaler = torch.amp.GradScaler(device.type, enabled=config.amp and device.type == "cuda")
    loader = make_loader(frame, indices, config.image_size, config.batch_size, config.num_workers, True,
                         config.seed + round_number, config.strong_augmentation, config.balanced_sampling,
                         config.cache_resized_images, config.use_metadata, config.image_enhancement,
                         config.hair_removal_probability, config.color_constancy,
                         config.balanced_sampling_power)
    model.train()
    apply_batchnorm_policy(model, config.batchnorm_policy)
    optimizer.zero_grad(set_to_none=True)
    train_loss_sum, train_correct, train_samples = 0.0, 0, 0
    for _ in range(config.local_epochs):
        model.train()
        apply_batchnorm_policy(model, config.batchnorm_policy)
        for step, batch in enumerate(loader, 1):
            if config.use_metadata:
                images, metadata, targets = batch
                metadata = metadata.to(device, non_blocking=True)
            else:
                images, targets = batch
                metadata = None
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            images, targets_a, targets_b, mix_lambda, metadata = mixup_cutmix(images, targets, config, metadata)
            with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                logits = model(images, metadata) if config.use_metadata else model(images)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Federated training produced non-finite logits; retry with --no-amp and lower learning rates.")
                loss_logits = logits if logit_adjustment is None else logits + logit_adjustment
                task_loss = criterion(loss_logits, targets_a)
                if targets_b is not None:
                    task_loss = mix_lambda * task_loss + (1 - mix_lambda) * criterion(loss_logits, targets_b)
                loss = task_loss
                if reference:
                    prox = sum((p - reference[n]).pow(2).sum() for n, p in model.named_parameters())
                    loss = loss + 0.5 * config.fedprox_mu * prox
                loss = loss / config.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError("Federated training produced a non-finite loss; retry with --no-amp and lower learning rates.")
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
    quality = local_metrics["macro_f1_present"]
    local_metrics["aggregation_quality_macro_f1_present"] = quality
    local_metrics["train_accuracy"] = train_correct / max(1, train_samples)
    local_metrics["train_loss"] = train_loss_sum / max(1, train_samples)
    # Score and transmit the same EMA model, including its EMA-updated BN buffers.
    update = state_cpu(ema.ema)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return update, len(indices), quality, local_metrics


def dihedral_views(images: torch.Tensor, count: int) -> list[torch.Tensor]:
    """Build the 8 square symmetries used by dihedral test-time augmentation."""
    if count == 1:
        return [images]
    horizontal = torch.flip(images, dims=[3])
    if count == 2:
        return [images, horizontal]
    vertical = torch.flip(images, dims=[2])
    both = torch.flip(images, dims=[2, 3])
    all_views = [images, horizontal, vertical, both]
    if count == 4:
        return all_views
    transposed = images.transpose(2, 3)
    all_views.extend([
        transposed, torch.flip(transposed, dims=[3]),
        torch.flip(transposed, dims=[2]), torch.flip(transposed, dims=[2, 3]),
    ])
    return all_views[:count]


@torch.inference_mode()
def predict_logits(model, frame, indices, config, device) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(
        frame, indices, config.image_size, config.batch_size, config.num_workers,
        False, config.seed, cache_resized=config.cache_resized_images,
        use_metadata=config.use_metadata,
        image_enhancement=False,
        color_constancy=config.color_constancy,
        balanced_sampling_power=config.balanced_sampling_power,
    )
    model.eval(); logits_all, targets = [], []
    view_count = config.tta_views if config.tta else 1
    for batch in loader:
        if config.use_metadata:
            images, metadata, labels = batch
            metadata = metadata.to(device, non_blocking=True)
        else:
            images, labels = batch
            metadata = None
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
            logits = torch.stack([
                model(view, metadata) if config.use_metadata else model(view)
                for view in dihedral_views(images, view_count)
            ]).mean(dim=0)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                "Model inference produced non-finite logits before probability calculation. "
                "Check for numerical instability; try disabling AMP and lowering the learning rate."
            )
        logits_all.append(logits.float().cpu().numpy())
        targets.extend(labels.numpy().tolist())
    return np.concatenate(logits_all), np.asarray(targets, dtype=int)


@torch.inference_mode()
def evaluate(model, frame, indices, config, device, class_names, return_probabilities=False):
    logits, targets = predict_logits(model, frame, indices, config, device)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    metrics, matrix = classification_metrics(targets, probabilities, class_names)
    if return_probabilities:
        return metrics, matrix, targets, probabilities
    return metrics, matrix
