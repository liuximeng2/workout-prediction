"""Load and structure Hugging Face Trainer state from checkpoint directories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingHistory:
    """Training log (per logging step) and validation log (per eval)."""

    train: list[dict[str, Any]] = field(default_factory=list)
    eval: list[dict[str, Any]] = field(default_factory=list)
    trainer_meta: dict[str, Any] = field(default_factory=dict)


def resolve_trainer_state_path(run_dir: str | Path) -> Path:
    """Return ``trainer_state.json`` for a fine-tuning run.

    If ``run_dir/trainer_state.json`` exists, use it. Otherwise use the
    checkpoint subdirectory with the largest step (full log history).
    """
    run_dir = Path(run_dir).resolve()
    direct = run_dir / "trainer_state.json"
    if direct.is_file():
        return direct

    candidates = list(run_dir.glob("checkpoint-*/trainer_state.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No trainer_state.json under {run_dir} or checkpoint-*/trainer_state.json"
        )
    return max(candidates, key=lambda p: int(p.parent.name.split("-", 1)[1]))


def load_training_history(
    run_dir: str | Path | None = None,
    *,
    trainer_state_path: str | Path | None = None,
) -> TrainingHistory:
    """Parse ``trainer_state.json`` into training vs validation series."""
    if trainer_state_path is not None:
        path = Path(trainer_state_path).resolve()
    elif run_dir is not None:
        path = resolve_trainer_state_path(run_dir)
    else:
        raise ValueError("Provide run_dir or trainer_state_path")

    with path.open(encoding="utf-8") as f:
        state = json.load(f)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for entry in state.get("log_history", []):
        if "eval_loss" in entry or "eval_accuracy" in entry:
            eval_rows.append(dict(entry))
        elif "loss" in entry:
            train_rows.append(dict(entry))

    meta = {
        "best_global_step": state.get("best_global_step"),
        "best_metric": state.get("best_metric"),
        "best_model_checkpoint": state.get("best_model_checkpoint"),
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "trainer_state_path": str(path),
    }
    return TrainingHistory(train=train_rows, eval=eval_rows, trainer_meta=meta)


def load_train_results_json(run_dir: str | Path) -> dict[str, Any] | None:
    """Load ``train_results.json`` from the run directory root, if present."""
    p = Path(run_dir).resolve() / "train_results.json"
    if not p.is_file():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def moving_average(values: list[float], window: int) -> list[float]:
    """Simple trailing moving average; shorter prefixes averaged over available length."""
    if window <= 1 or not values:
        return list(values)
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out
