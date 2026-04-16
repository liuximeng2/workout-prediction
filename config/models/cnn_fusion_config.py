"""CNN Fusion model configuration."""

from dataclasses import dataclass

from config.base_config import BaseConfig


@dataclass
class CNNFusionConfig(BaseConfig):
    """Training configuration for the CNN Fusion model.

    Inherits all global settings from :class:`~config.base_config.BaseConfig`.
    """

    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = "cnn_fusion"
    output_dir: str = "checkpoints/cnn-fusion"

    # ── Architecture ─────────────────────────────────────────────────────────
    # Backbone CNN: "resnet18", "resnet34", "resnet50", or "resnet101"
    backbone: str = "resnet50"
    # Number of frames uniformly sampled from each video clip
    num_frames: int = 8
    # Temporal aggregation across per-frame features: "sum" or "mean"
    aggregation: str = "sum"
    dropout: float = 0.5

    # Freeze strategy:
    #   "full"          — train backbone + head end-to-end
    #   "head_only"     — freeze backbone, train only the classification head
    #   "backbone_only" — freeze head, train only the backbone
    freeze_strategy: str = "full"

    # ── Video sampling ───────────────────────────────────────────────────────
    clip_duration: float = 4.0   # seconds of video per sample

    # ── Training hyperparameters ─────────────────────────────────────────────
    num_epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_step_size: int = 10       # StepLR: decay every N epochs
    lr_gamma: float = 0.1

    # ── Logging ───────────────────────────────────────────────────────────────
    log_every_n_steps: int = 20
    metric_for_best_model: str = "val_acc"
