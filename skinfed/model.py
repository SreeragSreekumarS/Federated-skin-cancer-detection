from __future__ import annotations
from types import MethodType
import warnings
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.ops import StochasticDepth


class CBAM(nn.Module):
    """Small channel-and-spatial attention block for dermoscopic feature maps.

    This preserves MobileNetV3's low memory footprint while giving the classifier
    an explicit way to emphasize lesion-relevant regions, inspired by CA Y-Net.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.channel = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channel_attention = torch.sigmoid(
            self.channel(nn.functional.adaptive_avg_pool2d(features, 1))
            + self.channel(nn.functional.adaptive_max_pool2d(features, 1))
        )
        features = features * channel_attention
        spatial_input = torch.cat((features.mean(dim=1, keepdim=True), features.amax(dim=1, keepdim=True)), dim=1)
        return features * torch.sigmoid(self.spatial(spatial_input))


class GeMPool(nn.Module):
    """Learnable generalized-mean pooling for non-negative feature maps."""
    def __init__(self, initial_power: float = 3.0) -> None:
        super().__init__()
        self.power = nn.Parameter(torch.tensor(float(initial_power)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        power = self.power.clamp(1.0, 6.0)
        features = nn.functional.relu(features).clamp_min(1e-6)
        return nn.functional.avg_pool2d(features.pow(power), kernel_size=features.shape[-2:]).pow(1.0 / power)


class TimmMetadataClassifier(nn.Module):
    """Attach the project's approved age/sex/site MLP to a timm image model."""
    def __init__(self, backbone: nn.Module, num_classes: int, metadata_dim: int,
                 hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = backbone
        feature_dim = int(backbone.num_features)
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64), nn.LayerNorm(64), nn.Hardswish()
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + 64, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.Hardswish(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor, metadata: torch.Tensor | None = None) -> torch.Tensor:
        if metadata is None:
            raise ValueError("This model requires age/sex/site metadata inputs")
        features = self.backbone.forward_features(images)
        features = self.backbone.forward_head(features, pre_logits=True)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return self.classifier(torch.cat((features, self.metadata_encoder(metadata)), dim=1))


class AttentionMobileNetV3(nn.Module):
    # MobileNetV3-Large outputs these channel widths at progressively later
    # stages: 40 (index 6), 112 (index 12), and 960 (index 16).
    MULTISCALE_STAGES = {6: 40, 12: 112, 16: 960}

    def __init__(self, base: nn.Module, num_classes: int, use_cbam: bool,
                 classifier_hidden_dim: int, classifier_dropout: float,
                 metadata_dim: int = 0, multiscale_head: bool = False,
                 multiscale_feature_dim: int = 128, pooling: str = "avg") -> None:
        super().__init__()
        self.features = base.features
        self.attention = CBAM(960) if use_cbam else nn.Identity()
        self.avgpool = base.avgpool
        self.pooling = pooling
        self.gem_pool = GeMPool() if pooling == "gem" and not multiscale_head else None
        # A compact regularized head gives CBAM-refined features room to adapt to
        # dermoscopic classes without materially changing the lightweight backbone.
        # MobileNetV3-Large's feature extractor emits 960 channels.  Its final
        # ImageNet classifier later expands these to 1280 channels, so using the
        # last classifier layer's input size here would not match `features`.
        feature_dim = base.classifier[0].in_features
        self.multiscale_head = multiscale_head
        self.multiscale_feature_dim = multiscale_feature_dim
        if multiscale_head:
            self.multiscale_pool = nn.ModuleList([
                GeMPool() if pooling == "gem" else nn.AdaptiveAvgPool2d(1)
                for _ in self.MULTISCALE_STAGES
            ])
            self.multiscale_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(channels, multiscale_feature_dim),
                    nn.LayerNorm(multiscale_feature_dim),
                    nn.Hardswish(),
                    nn.Dropout(classifier_dropout / 2),
                )
                for channels in self.MULTISCALE_STAGES.values()
            ])
            classifier_input_dim = multiscale_feature_dim * len(self.MULTISCALE_STAGES)
        else:
            classifier_input_dim = feature_dim
        self.metadata_dim = metadata_dim
        self.metadata_encoder = (
            nn.Sequential(nn.Linear(metadata_dim, 64), nn.LayerNorm(64), nn.Hardswish())
            if metadata_dim else None
        )
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim + (64 if metadata_dim else 0), classifier_hidden_dim),
            # LayerNorm remains valid for a final short client batch, unlike
            # BatchNorm1d which can fail when a simulated hospital has one sample.
            nn.LayerNorm(classifier_hidden_dim), nn.Hardswish(), nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor, metadata: torch.Tensor | None = None) -> torch.Tensor:
        if self.multiscale_head:
            stage_features = {}
            features = images
            for index, layer in enumerate(self.features):
                features = layer(features)
                if index in self.MULTISCALE_STAGES:
                    stage_features[index] = features
            stage_features[16] = self.attention(stage_features[16])
            pooled_features = [
                pool(stage_features[index]).flatten(1)
                for pool, index in zip(self.multiscale_pool, self.MULTISCALE_STAGES)
            ]
            features = torch.cat([
                projection(pooled)
                for projection, pooled in zip(self.multiscale_projections, pooled_features)
            ], dim=1)
        else:
            features = self.attention(self.features(images))
            pooled = self.gem_pool(features) if self.gem_pool is not None else self.avgpool(features)
            features = torch.flatten(pooled, 1)
        if self.metadata_encoder is not None:
            if metadata is None:
                raise ValueError("This model requires age/sex/site metadata inputs")
            features = torch.cat((features, self.metadata_encoder(metadata)), dim=1)
        elif metadata is not None:
            raise ValueError("Metadata was supplied to a model configured without metadata")
        return self.classifier(features)


def build_model(num_classes: int, pretrained: bool = True, use_cbam: bool = True,
                classifier_hidden_dim: int = 256, classifier_dropout: float = 0.30,
                drop_path_rate: float = 0.10,
                model_name: str = "mobilenet_v3_large", metadata_dim: int = 0,
                multiscale_head: bool = False, multiscale_feature_dim: int = 128,
                pooling: str = "avg") -> nn.Module:
    if multiscale_head and model_name != "mobilenet_v3_large":
        raise ValueError("multiscale_head is currently implemented for mobilenet_v3_large only")
    if model_name != "mobilenet_v3_large":
        try:
            import timm
        except ImportError as error:
            raise RuntimeError(
                f"Model '{model_name}' requires timm; install a recent release with `pip install -U timm`."
            ) from error
        if not timm.is_model(model_name):
            raise ValueError(
                f"Model '{model_name}' is unavailable in timm {timm.__version__}. "
                "Upgrade timm and confirm that the model name is registered."
            )
        if use_cbam:
            warnings.warn("CBAM is specific to the MobileNetV3 wrapper; the timm model uses its native head.", stacklevel=2)
        if metadata_dim:
            backbone = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0,
                drop_rate=classifier_dropout, drop_path_rate=drop_path_rate,
            )
            return TimmMetadataClassifier(
                backbone, num_classes, metadata_dim, classifier_hidden_dim, classifier_dropout,
            )
        return timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=classifier_dropout,
            drop_path_rate=drop_path_rate,
        )

    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    base = mobilenet_v3_large(weights=weights)
    residual_blocks = [
        module for module in base.features
        if getattr(module, "use_res_connect", False) and hasattr(module, "block")
    ]
    if drop_path_rate and residual_blocks:
        denominator = max(1, len(residual_blocks) - 1)
        for index, module in enumerate(residual_blocks):
            probability = drop_path_rate * index / denominator
            module.add_module("drop_path", StochasticDepth(probability, mode="row"))
            module.forward = MethodType(_stochastic_residual_forward, module)
    return AttentionMobileNetV3(
        base, num_classes, use_cbam, classifier_hidden_dim, classifier_dropout,
        metadata_dim, multiscale_head, multiscale_feature_dim,
        pooling,
    )


def _stochastic_residual_forward(module: nn.Module, images: torch.Tensor) -> torch.Tensor:
    residual = module.block(images)
    return module.drop_path(residual) + images


def model_backbone_layers(model: nn.Module) -> list[nn.Module]:
    """Return ordered feature stages for layer-wise LR decay and freezing."""
    if hasattr(model, "features"):
        return list(model.features.children())
    if hasattr(model, "blocks"):
        layers = [
            getattr(model, name) for name in ("conv_stem", "bn1", "act1")
            if isinstance(getattr(model, name, None), nn.Module)
        ]
        layers.extend(list(model.blocks.children()))
        layers.extend(
            getattr(model, name) for name in ("conv_head", "norm_head", "act2")
            if isinstance(getattr(model, name, None), nn.Module)
        )
        return layers
    if hasattr(model, "stages") and hasattr(model, "stem"):
        layers = [model.stem, *list(model.stages.children())]
        final_conv = getattr(model, "final_conv", None)
        if isinstance(final_conv, nn.Module) and not isinstance(final_conv, nn.Identity):
            layers.append(final_conv)
        return layers
    if hasattr(model, "backbone") and isinstance(model.backbone, nn.Module):
        return model_backbone_layers(model.backbone)
    raise TypeError(f"Cannot identify backbone stages for {type(model).__name__}")


def freeze_features(model: nn.Module, frozen: bool, early_stages: int = 0) -> None:
    for index, layer in enumerate(model_backbone_layers(model)):
        for parameter in layer.parameters():
            parameter.requires_grad = not (frozen or index < early_stages)
