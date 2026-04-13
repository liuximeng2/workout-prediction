"""Pose MLP classifier: a simple feed-forward network on engineered pose features."""

from typing import Any, Tuple

import torch
import torch.nn as nn

from utils.pose_dataset import FEATURE_DIM


class PoseMLP(nn.Module):
    """Multi-layer perceptron for classifying exercise videos from pose features.

    Args:
        num_classes: Number of exercise classes.
        input_dim:   Dimensionality of the input feature vector.
        hidden_dim:  Width of each hidden layer.
        num_layers:  Number of hidden layers.
        dropout:     Dropout probability between hidden layers.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch_size, input_dim) feature tensor.

        Returns:
            (batch_size, num_classes) logits.
        """
        return self.net(x)


def build_model(
    num_classes: int,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.3,
    **kwargs: Any,
) -> Tuple[PoseMLP, None]:
    """Factory function matching the model registry interface.

    Returns:
        ``(model, None)`` — no processor needed for pose features.
    """
    model = PoseMLP(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    return model, None
