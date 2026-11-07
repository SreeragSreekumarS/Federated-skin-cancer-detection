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


class AttentionMobileNetV3(nn.Module):
    def __init__(self, base: nn.Module, num_classes: int, use_cbam: bool, classifier_hidden_dim: int, classifier_dropout: float) -> None:
        super().__init__()
        self.features = base.features
        self.attention = CBAM(960) if use_cbam else nn.Identity()
        self.avgpool = base.avgpool
        # A compact regularized head gives CBAM-refined features room to adapt to
        # dermoscopic classes without materially changing the lightweight backbone.
        # MobileNetV3-Large's feature extractor emits 960 channels.  Its final
        # ImageNet classifier later expands these to 1280 channels, so using the
        # last classifier layer's input size here would not match `features`.
        feature_dim = base.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, classifier_hidden_dim),
            # LayerNorm remains valid for a final short client batch, unlike
            # BatchNorm1d which can fail when a simulated hospital has one sample.
            nn.LayerNorm(classifier_hidden_dim), nn.Hardswish(), nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.attention(self.features(images))
        return self.classifier(torch.flatten(self.avgpool(features), 1))


def build_model(num_classes: int, pretrained: bool = True, use_cbam: bool = True,
                classifier_hidden_dim: int = 256, classifier_dropout: float = 0.30,
                drop_path_rate: float = 0.10,
                model_name: str = "mobilenet_v3_large") -> nn.Module:
    if model_name != "mobilenet_v3_large":
        try:
            import timm
        except ImportError as error:
            raise RuntimeError(
                f"Model '{model_name}' requires timm; install a recent release with `pip install -U timm`."
            ) from error
        available = timm.list_models()
        if model_name not in available:
            raise ValueError(
                f"Model '{model_name}' is unavailable in timm {timm.__version__}. "
                "Upgrade timm and confirm availability with `python -c \"import timm; "
                "print(timm.__version__, 'mobilenetv4_conv_medium' in timm.list_models())\"`."
            )
        if use_cbam:
            warnings.warn("CBAM is specific to the MobileNetV3 wrapper; the timm model uses its native head.", stacklevel=2)
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
        base, num_classes, use_cbam, classifier_hidden_dim, classifier_dropout
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
    raise TypeError(f"Cannot identify backbone stages for {type(model).__name__}")


def freeze_features(model: nn.Module, frozen: bool) -> None:
    for layer in model_backbone_layers(model):
        for parameter in layer.parameters():
            parameter.requires_grad = not frozen
