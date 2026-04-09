"""Post-training analysis: training curves, metrics, and inference benchmarking.

Heavy deps (torch, transformers) live in ``inference_metrics`` — import that module
when you need test evaluation:

    from analysis.inference_metrics import evaluate_checkpoint_on_test
"""

from analysis.training_history import (
    TrainingHistory,
    load_train_results_json,
    load_training_history,
    moving_average,
    resolve_trainer_state_path,
)

__all__ = [
    "TrainingHistory",
    "load_train_results_json",
    "load_training_history",
    "moving_average",
    "resolve_trainer_state_path",
]
