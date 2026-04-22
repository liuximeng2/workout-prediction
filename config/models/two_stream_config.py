"""Two-Stream CNN configuration."""

from dataclasses import dataclass, field
from pathlib import Path

from config.base_config import BaseConfig


@dataclass
class TwoStreamConfig(BaseConfig):
    """Training configuration for the Two-Stream CNN.

    Inherits all global settings from :class:`~config.base_config.BaseConfig`.
    """

    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = "two_stream"
    output_dir: str = "checkpoints/two-stream-workout"

    # ── Flow ─────────────────────────────────────────────────────────────────
    flow_root: Path = Path("data/flow")
    num_flow_frames: int = 20         # L; temporal stream gets 2*L channels

    # ── Architecture ─────────────────────────────────────────────────────────
    learnable_fusion: bool = True      # learnable vs fixed 0.5/0.5 fusion weight

    # Freeze strategy:
    #   "full"          — train both streams end-to-end (~50M params each)
    #   "head_only"     — freeze backbones, train FC heads + fusion only
    #   "spatial_only"  — freeze temporal stream
    #   "temporal_only" — freeze spatial stream
    freeze_strategy: str = "full"

    # ── Video sampling ───────────────────────────────────────────────────────
    clip_duration: float = 4.0         # seconds of video per sample

    # ── Training hyperparameters ─────────────────────────────────────────────
    num_epochs: int = 20
    batch_size: int = 32               # larger batches OK (no video encoder overhead)
    learning_rate: float = 1e-3        # SGD-style LR; ResNet fine-tuning
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_step_size: int = 10             # StepLR: decay every N epochs
    lr_gamma: float = 0.1

    # ── Logging ───────────────────────────────────────────────────────────────
    log_every_n_steps: int = 20
    metric_for_best_model: str = "val_acc"
