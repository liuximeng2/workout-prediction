"""ViViT-specific configuration."""

from dataclasses import dataclass

from config.base_config import BaseConfig


@dataclass
class ViViTConfig(BaseConfig):
    """Fine-tuning configuration for ViViT.

    Inherits all global settings from :class:`~config.base_config.BaseConfig`
    and adds ViViT-specific knobs.
    """

    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = "vivit"
    model_ckpt: str = "google/vivit-b-16x2-kinetics400"

    # Checkpoint directory (relative to repo root)
    output_dir: str = "checkpoints/vivit"

    # ── Video sampling ───────────────────────────────────────────────────────
    # ViViT (google/vivit-b-16x2-kinetics400) expects 32 frames.
    # ``clip_duration = num_frames * sample_rate / fps`` — driven by the
    # architecture's native input budget.
    sample_rate: int = 4
    fps: int = 30

    # ── Training hyperparameters ─────────────────────────────────────────────
    num_epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    metric_for_best_model: str = "accuracy"

    # ── Freeze strategy ──────────────────────────────────────────────────────
    # "full"      — fine-tune all parameters (best accuracy, needs more data)
    # "head_only" — freeze encoder, train only the classification head (fast,
    #               low memory, good when data is scarce)
    freeze_strategy: str = "full"

    # ── HuggingFace Hub ───────────────────────────────────────────────────────
    push_to_hub: bool = False
