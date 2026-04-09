"""Global configuration shared across all models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class BaseConfig:
    """Settings that are independent of which model is being used.

    Model-specific hyperparameters (learning rate, checkpoint, num frames, …)
    should live in a subclass under ``config/models/``.
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    # Each path is a root with one sub-folder per class (videos merged for training).
    data_roots: Tuple[Path, ...] = (
        Path("data/verified_data/verified_data/data_btc_10s"),
        Path("data/verified_data/verified_data/data_crawl_10s"),
    )

    # Hold-out test videos: one sub-folder per class (same names as under ``data_roots``).
    # This repo stores them under ``data/test/test/<class>/`` (not ``data/test/<class>/``).
    # If no videos are found here, eval/analysis fall back to the stratified test split
    # from ``data_roots`` (often a few hundred clips). Set to ``None`` to always use that split.
    test_data_roots: Optional[Tuple[Path, ...]] = (Path("data/test/test"),)

    # Stratified split fractions (test = 1 - train_split - val_split)
    train_split: float = 0.70
    val_split: float = 0.15

    # ── Output ────────────────────────────────────────────────────────────────
    # Base directory; model configs append a model-specific sub-folder.
    output_base: str = "checkpoints"

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42
