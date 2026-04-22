"""video-SALMONN-2 visual-encoder configuration."""

from dataclasses import dataclass

from config.base_config import BaseConfig


@dataclass
class VideoSalmonnConfig(BaseConfig):
    """Configuration for the video-SALMONN-2 visual encoder + linear head.

    Only the SigLIP visual encoder and mm_projector are loaded from the
    video-SALMONN-2 safetensors.  The 7B Qwen2 LLM is skipped entirely.
    Embeddings must be pre-computed with precompute_salmonn_embeddings.py
    before training.
    """

    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = "video_salmonn"
    model_ckpt: str = "tsinghua-ee/video-SALMONN-2"
    output_dir: str = "checkpoints/video-salmonn"

    # ── Video sampling (used during embedding pre-computation only) ──────────
    num_frames: int = 8
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

    # ── Text conditioning ────────────────────────────────────────────────────
    # False → visual-only (SigLIP + mm_projector, existing behaviour)
    # True  → full Qwen2-7B forward pass with text prompt (~14 GB RAM)
    text_conditioned: bool = False
    text_prompt: str = "Classify the exercise being performed."
