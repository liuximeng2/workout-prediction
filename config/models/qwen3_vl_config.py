"""Qwen3-VL (Ollama) configuration for zero-shot exercise classification."""

from dataclasses import dataclass

from config.base_config import BaseConfig


@dataclass
class Qwen3VLConfig(BaseConfig):
    model_name: str = "qwen3_vl"

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3-vl:latest"

    # ── Frame sampling ────────────────────────────────────────────────────────
    # Frames uniformly sampled from the full video and sent to the model.
    # Qwen3-VL supports up to 768 frames; 16 balances accuracy vs. latency.
    num_frames: int = 16
    # Short-side resize before JPEG encoding (reduces visual token count).
    frame_size: int = 336

    # ── Multi-clip voting ─────────────────────────────────────────────────────
    # Number of independent frame sets to sample; majority vote decides label.
    # Use 1 for speed, 3 for slightly more robust results.
    num_clips: int = 1

    # ── Inference ─────────────────────────────────────────────────────────────
    request_timeout: int = 180  # seconds per Ollama request
