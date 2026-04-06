"""Model registry.

To add a new model:
1. Create ``model/<your_model>/`` with ``model.py`` exposing a ``build_model`` function.
2. Add an entry to ``MODEL_REGISTRY`` below.
3. Add a config dataclass under ``config/models/<your_model>_config.py``.
"""

from typing import Any, Dict, Tuple

from model.video_mae.model import build_model as _build_video_mae

# Registry maps model_name → callable(model_ckpt, label2id, id2label) → (model, processor)
MODEL_REGISTRY: Dict[str, Any] = {
    "video_mae": _build_video_mae,
}


def build_model(model_name: str, **kwargs) -> Tuple[Any, Any]:
    """Instantiate a model and its associated processor by name.

    Args:
        model_name: Key from :data:`MODEL_REGISTRY` (e.g. ``"video_mae"``).
        **kwargs:   Forwarded to the model's factory function
                    (``model_ckpt``, ``label2id``, ``id2label``).

    Returns:
        ``(model, processor)`` tuple.

    Raises:
        ValueError: If ``model_name`` is not in the registry.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)
