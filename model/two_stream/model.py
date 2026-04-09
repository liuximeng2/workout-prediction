"""Two-Stream Convolutional Network for action recognition.

Reference: Simonyan & Zisserman, "Two-Stream Convolutional Networks for
Action Recognition in Videos", NeurIPS 2014.

Architecture
------------
Spatial stream:  ResNet-50 pretrained on ImageNet.
                 Input: (N, 3, 224, 224) — single RGB frame.

Temporal stream: ResNet-50 pretrained on ImageNet.
                 First conv layer replaced to accept (2*L) input channels
                 (stacked u/v optical flow), initialised by averaging the
                 original 3-channel weights across the new channel dim.
                 Input: (N, 2*L, 224, 224) — stacked flow frames.

Fusion:          Weighted average of softmax logits from both streams.
                 The fusion weight ``flow_weight`` is a learnable scalar
                 (initialised to 0.5) so the model can adapt the balance
                 during training.  Set ``learnable_fusion=False`` to use
                 a fixed 0.5/0.5 average instead.
"""

from typing import Literal, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

FreezeStrategy = Literal["full", "spatial_only", "temporal_only", "head_only"]


# ── Stream builder ────────────────────────────────────────────────────────────

def _build_spatial_stream(num_classes: int) -> nn.Module:
    """ResNet-50 pretrained on ImageNet, classification head replaced."""
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)
    return backbone


def _build_temporal_stream(num_classes: int, num_flow_channels: int) -> nn.Module:
    """ResNet-50 with first conv adapted for stacked optical flow input.

    Weight initialisation: the pretrained 3-channel conv weights are averaged
    across the channel dimension and tiled to fill ``num_flow_channels``
    channels, then scaled by 3 / num_flow_channels so the output magnitude
    stays comparable to the original.
    """
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    old_conv = backbone.conv1
    new_conv = nn.Conv2d(
        num_flow_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    # Initialise: tile the mean of the 3 source channels
    with torch.no_grad():
        mean_weight = old_conv.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
        new_conv.weight.copy_(
            mean_weight.repeat(1, num_flow_channels, 1, 1)
            * (3.0 / num_flow_channels)
        )
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    backbone.conv1 = new_conv
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)
    return backbone


# ── Two-Stream model ──────────────────────────────────────────────────────────

class TwoStreamNet(nn.Module):
    """Two-stream network with late fusion.

    Args:
        num_classes:       Number of output classes.
        num_flow_frames:   Number of stacked flow frames (L); temporal stream
                           receives 2*L input channels.
        learnable_fusion:  If True, fusion weight is a learnable parameter.
                           If False, a fixed 0.5/0.5 average is used.
    """

    def __init__(
        self,
        num_classes: int,
        num_flow_frames: int = 10,
        learnable_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.spatial_stream  = _build_spatial_stream(num_classes)
        self.temporal_stream = _build_temporal_stream(num_classes, 2 * num_flow_frames)

        if learnable_fusion:
            # Sigmoid of this scalar gives the flow weight in (0, 1)
            self._fusion_logit = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("_fusion_logit", None)
        self.learnable_fusion = learnable_fusion

    @property
    def flow_weight(self) -> float:
        """Current weight given to the temporal (flow) stream (0–1)."""
        if self.learnable_fusion:
            return torch.sigmoid(self._fusion_logit).item()
        return 0.5

    def forward(self, video: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            video: (N, 3, H, W)        — RGB frame (spatial stream input).
            flow:  (N, 2*L, H, W)      — stacked optical flow (temporal stream).

        Returns:
            logits: (N, num_classes)
        """
        spatial_logits  = self.spatial_stream(video)    # (N, C)
        temporal_logits = self.temporal_stream(flow)    # (N, C)

        if self.learnable_fusion:
            w = torch.sigmoid(self._fusion_logit)       # scalar in (0, 1)
        else:
            w = 0.5

        return (1 - w) * spatial_logits + w * temporal_logits


# ── Registry entry ────────────────────────────────────────────────────────────

def build_model(
    num_classes: int,
    num_flow_frames: int = 10,
    learnable_fusion: bool = True,
    **_kwargs,
) -> Tuple[TwoStreamNet, None]:
    """Instantiate a TwoStreamNet.  Returns ``(model, None)`` to match the
    registry interface (no separate processor needed for this model).
    """
    model = TwoStreamNet(
        num_classes=num_classes,
        num_flow_frames=num_flow_frames,
        learnable_fusion=learnable_fusion,
    )
    return model, None


# ── Freeze helpers ────────────────────────────────────────────────────────────

def apply_freeze_strategy(model: TwoStreamNet, strategy: FreezeStrategy) -> None:
    """Freeze / unfreeze parameters for transfer learning.

    Strategies
    ----------
    ``"full"``          All parameters trainable (~50M × 2 streams).
    ``"head_only"``     Freeze both backbones; train only the two FC heads
                        and the fusion weight (~2 × num_classes parameters).
    ``"spatial_only"``  Freeze temporal stream; train spatial stream + fusion.
    ``"temporal_only"`` Freeze spatial stream; train temporal stream + fusion.
    """
    # Start: freeze everything
    for p in model.parameters():
        p.requires_grad = False

    if strategy == "full":
        for p in model.parameters():
            p.requires_grad = True

    elif strategy == "head_only":
        for p in model.spatial_stream.fc.parameters():
            p.requires_grad = True
        for p in model.temporal_stream.fc.parameters():
            p.requires_grad = True
        if model.learnable_fusion:
            model._fusion_logit.requires_grad = True

    elif strategy == "spatial_only":
        for p in model.spatial_stream.parameters():
            p.requires_grad = True
        if model.learnable_fusion:
            model._fusion_logit.requires_grad = True

    elif strategy == "temporal_only":
        for p in model.temporal_stream.parameters():
            p.requires_grad = True
        if model.learnable_fusion:
            model._fusion_logit.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(
        f"Freeze strategy: '{strategy}' — "
        f"trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.1f}%)"
    )
