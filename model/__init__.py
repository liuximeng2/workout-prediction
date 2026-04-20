"""Model registry.

To add a new model:
1. Create ``model/<your_model>/`` with ``model.py`` exposing a ``build_model`` function.
2. Add an entry to ``MODEL_REGISTRY`` below.
3. Add a config dataclass under ``config/models/<your_model>_config.py``.
"""

from typing import Any, Dict, Tuple

from model.video_mae.model import build_model as _build_video_mae
from model.two_stream.model import build_model as _build_two_stream
from model.vivit.model import build_model as _build_vivit
from model.pose.model import build_model as _build_pose
from model.cnn_fusion.model import build_model as _build_cnn_fusion
from model.qwen3_vl.model import build_model as _build_qwen3_vl

# Registry maps model_name → callable(**kwargs) → (model, processor)
# Note: qwen3_vl returns (Qwen3VLClassifier, None) — not a torch.nn.Module.
# Use scripts/eval_qwen3_vl.py for evaluation; no training script exists.
MODEL_REGISTRY: Dict[str, Any] = {
    "video_mae":  _build_video_mae,
    "two_stream": _build_two_stream,
    "vivit":      _build_vivit,
    "pose":       _build_pose,
    "cnn_fusion": _build_cnn_fusion,
    "qwen3_vl":   _build_qwen3_vl,
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
