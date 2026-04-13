"""Pose-based MLP classifier configuration."""

from dataclasses import dataclass
from pathlib import Path

from config.base_config import BaseConfig


@dataclass
class PoseConfig(BaseConfig):
    """Training configuration for the Pose MLP classifier.

    Inherits all global settings from :class:`~config.base_config.BaseConfig`.
    """

    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = "pose"
    output_dir: str = "checkpoints/pose-workout"

    # ── Pose data ────────────────────────────────────────────────────────────
    pose_root: Path = Path("data/pose")

    # ── Architecture ─────────────────────────────────────────────────────────
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.3

    # ── Training hyperparameters ─────────────────────────────────────────────
    num_epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_step_size: int = 30
    lr_gamma: float = 0.3

    # ── Logging ──────────────────────────────────────────────────────────────
    log_every_n_steps: int = 10
    metric_for_best_model: str = "val_acc"
