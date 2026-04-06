"""Global configuration shared across all models."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseConfig:
    """Settings that are independent of which model is being used.

    Model-specific hyperparameters (learning rate, checkpoint, num frames, …)
    should live in a subclass under ``config/models/``.
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    data_root: Path = Path("data/verified_data/verified_data/data_btc_10s")

    # Stratified split fractions (test = 1 - train_split - val_split)
    train_split: float = 0.70
    val_split: float = 0.15

    # ── Output ────────────────────────────────────────────────────────────────
    # Base directory; model configs append a model-specific sub-folder.
    output_base: str = "checkpoints"

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42
