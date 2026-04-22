"""VideoPrism visual-encoder configuration."""

from dataclasses import dataclass

from config.base_config import BaseConfig


@dataclass
class VideoPrismConfig(BaseConfig):
    """Configuration for the VideoPrism ViT-B encoder + linear head.

    Embeddings must be pre-computed with precompute_videoprism_embeddings.py
    before training.  The JAX/Flax encoder is never loaded during training.
    """

    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = "videoprism"
    model_ckpt: str = "google/videoprism-base-f16r288"
    output_dir: str = "checkpoints/videoprism"

    # ── Video sampling (used during embedding pre-computation only) ──────────
    num_frames: int = 16
    sample_rate: int = 4
    fps: int = 30

    # ── Training hyperparameters ─────────────────────────────────────────────
    num_epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    metric_for_best_model: str = "accuracy"

    # ── Freeze strategy ──────────────────────────────────────────────────────
    # Always head_only — the visual encoder is never fine-tuned here.
    freeze_strategy: str = "head_only"

    push_to_hub: bool = False
