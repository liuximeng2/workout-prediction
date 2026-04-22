"""VideoPrism linear classifier trained on cached embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn

_EMBED_DIM = 768  # ViT-B output dimension


class VideoPrismHeadOnly(nn.Module):
    """Linear classifier trained on cached VideoPrism embeddings."""

    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(_EMBED_DIM, num_labels)

    def forward(
        self,
        embedding: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        logits = self.classifier(embedding.to(self.classifier.weight.dtype))
        loss = nn.CrossEntropyLoss()(logits.float(), labels) if labels is not None else None
        return (loss, logits) if loss is not None else logits


def build_model_cached(label2id: dict) -> tuple[VideoPrismHeadOnly, None]:
    """Return only the linear head — no encoder loaded."""
    return VideoPrismHeadOnly(num_labels=len(label2id)), None


def build_model(
    model_ckpt: str,
    label2id: dict,
    id2label: dict,
) -> tuple[VideoPrismHeadOnly, None]:
    return build_model_cached(label2id)


def apply_freeze_strategy(model: VideoPrismHeadOnly, strategy: str) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Freeze strategy: '{strategy}' — "
        f"trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.1f}%)"
    )
