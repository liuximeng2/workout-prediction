"""LLaVA-OneVision-specific configuration."""

from dataclasses import dataclass

from config.base_config import BaseConfig


@dataclass
class LlavaOnevisionConfig(BaseConfig):
    """Fine-tuning configuration for LLaVA-OneVision (classification head variant).

    The model wraps LlavaOnevisionForConditionalGeneration and adds a linear
    classification head on top of the language model's final hidden state.
    """

    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = "llava_onevision"
    model_ckpt: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"

    output_dir: str = "checkpoints/llava-onevision-workout"

    # ── Video sampling ───────────────────────────────────────────────────────
    # Number of frames passed to the model as an image sequence.
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
    # "head_only" — freeze the entire base model, train only the linear head
    # "full"      — fine-tune all parameters (expensive for a 0.5B model)
    freeze_strategy: str = "head_only"

    # ── Text conditioning ────────────────────────────────────────────────────
    # True  → include task instruction in the chat prompt (existing behaviour)
    # False → pass only the video token with no text instruction
    text_conditioned: bool = True
    text_prompt: str = "Classify the exercise being performed."

    # ── HuggingFace Hub ───────────────────────────────────────────────────────
    push_to_hub: bool = False
