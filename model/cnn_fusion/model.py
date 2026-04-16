"""CNN Fusion model: per-frame CNN features aggregated across frames, then classified.

Architecture
------------
1. Uniformly sample ``num_frames`` frames from the input video clip (handled upstream
   by the data pipeline via ``UniformTemporalSubsample``).
2. Reshape the (B, C, T, H, W) clip to (B*T, C, H, W) and pass all frames through a
   shared pretrained CNN backbone (e.g. ResNet-50) to get one feature vector per frame.
3. Aggregate the per-frame features along the time axis by summation (or mean).
4. Apply dropout and a linear classification head to produce logits.

Supported backbones: resnet18, resnet34, resnet50, resnet101.
"""

from typing import Literal, Tuple

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
)

FreezeStrategy = Literal["full", "head_only", "backbone_only"]

# Maps backbone name → (factory, pretrained weights, feature dimension)
_BACKBONE_REGISTRY = {
    "resnet18":  (resnet18,  ResNet18_Weights.IMAGENET1K_V1,  512),
    "resnet34":  (resnet34,  ResNet34_Weights.IMAGENET1K_V1,  512),
    "resnet50":  (resnet50,  ResNet50_Weights.IMAGENET1K_V2,  2048),
    "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V2, 2048),
}


def _build_backbone(name: str) -> Tuple[nn.Module, int]:
    """Load a pretrained backbone and remove its classification head.

    Returns:
        ``(backbone, feature_dim)`` — the backbone outputs a ``(N, feature_dim)``
        tensor after global average pooling.
    """
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Available: {list(_BACKBONE_REGISTRY)}"
        )
    factory, weights, feat_dim = _BACKBONE_REGISTRY[name]
    backbone = factory(weights=weights)
    backbone.fc = nn.Identity()   # strip classifier; keep avgpool output
    return backbone, feat_dim


class CNNFusion(nn.Module):
    """Per-frame CNN feature extraction with temporal aggregation and classification.

    Args:
        num_classes:  Number of output classes.
        backbone:     Name of the pretrained CNN backbone (default: ``"resnet50"``).
        num_frames:   Number of frames expected in the input clip (informational;
                      actual subsampling is done by the data transform).
        aggregation:  How to combine per-frame features — ``"sum"`` or ``"mean"``.
        dropout:      Dropout probability applied before the classification head.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        num_frames: int = 8,
        aggregation: str = "sum",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        if aggregation not in ("sum", "mean"):
            raise ValueError(f"aggregation must be 'sum' or 'mean', got '{aggregation}'")
        self.aggregation = aggregation

        self.backbone, self.feat_dim = _build_backbone(backbone)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            video: (B, C, T, H, W) — batch of video clips with T frames.

        Returns:
            logits: (B, num_classes)
        """
        B, C, T, H, W = video.shape

        # Flatten temporal dim into the batch for a single CNN forward pass
        frames = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, C, H, W)

        # Extract per-frame features: (B*T, feat_dim)
        features = self.backbone(frames)

        # Restore shape to (B, T, feat_dim) and aggregate across frames
        features = features.view(B, T, -1)
        if self.aggregation == "sum":
            agg = features.sum(dim=1)   # (B, feat_dim)
        else:
            agg = features.mean(dim=1)  # (B, feat_dim)

        return self.classifier(agg)


# ── Registry entry ────────────────────────────────────────────────────────────

def build_model(
    num_classes: int,
    backbone: str = "resnet50",
    num_frames: int = 8,
    aggregation: str = "sum",
    dropout: float = 0.5,
    **_kwargs,
) -> Tuple[CNNFusion, None]:
    """Instantiate a CNNFusion model.

    Returns:
        ``(model, None)`` — no separate processor needed.
    """
    model = CNNFusion(
        num_classes=num_classes,
        backbone=backbone,
        num_frames=num_frames,
        aggregation=aggregation,
        dropout=dropout,
    )
    return model, None


# ── Freeze helpers ────────────────────────────────────────────────────────────

def apply_freeze_strategy(model: CNNFusion, strategy: FreezeStrategy) -> None:
    """Freeze or unfreeze model parameters for transfer learning.

    Strategies
    ----------
    ``"full"``          All parameters trainable (~25M for ResNet-50).
    ``"head_only"``     Freeze backbone; train only the classification head.
    ``"backbone_only"`` Freeze the classification head; train only the backbone.
    """
    for p in model.parameters():
        p.requires_grad = False

    if strategy == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif strategy == "head_only":
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif strategy == "backbone_only":
        for p in model.backbone.parameters():
            p.requires_grad = True
    else:
        raise ValueError(
            f"Unknown freeze_strategy '{strategy}'. "
            "Choose 'full', 'head_only', or 'backbone_only'."
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(
        f"Freeze strategy: '{strategy}' — "
        f"trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.1f}%)"
    )
