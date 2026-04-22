"""LLaVA-OneVision classification wrapper.

Wraps LlavaOnevisionForConditionalGeneration and adds a linear head on top of
the language model's final hidden state for video classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor


class LlavaOnevisionClassifier(nn.Module):
    """LLaVA-OneVision base model with a linear classification head."""

    def __init__(self, base_model: LlavaOnevisionForConditionalGeneration, num_labels: int):
        super().__init__()
        self.base = base_model
        hidden_size = base_model.config.text_config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        embedding: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        if embedding is not None:
            # Fast path: pre-computed embedding, skip base model entirely.
            pooled = embedding
        else:
            # Full path: run vision encoder + language model.
            pv = pixel_values_videos if pixel_values_videos is not None else pixel_values
            out = self.base(
                pixel_values_videos=pv,
                image_sizes=image_sizes,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = out.hidden_states[-1]  # (B, seq_len, hidden)
            pooled = last_hidden[:, -1, :]       # (B, hidden)

        logits = self.classifier(pooled.to(self.classifier.weight.dtype))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.float(), labels)

        # HF Trainer expects either a loss scalar or a (loss, logits) tuple.
        if loss is not None:
            return loss, logits
        return logits


_HIDDEN_SIZE = 896  # Qwen2-0.5B text hidden size


class LlavaOnevisionHeadOnly(nn.Module):
    """Standalone linear classifier for cached-embedding training.

    No base model — forward expects a pre-computed embedding tensor directly.
    """

    def __init__(self, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(_HIDDEN_SIZE, num_labels)

    def forward(self, embedding: torch.Tensor, labels: torch.Tensor | None = None):
        logits = self.classifier(embedding.to(self.classifier.weight.dtype))
        loss = nn.CrossEntropyLoss()(logits.float(), labels) if labels is not None else None
        return (loss, logits) if loss is not None else logits


def build_model_cached(label2id: dict) -> tuple[LlavaOnevisionHeadOnly, None]:
    """Build only the linear classification head — no base model loaded.

    Use this when pre-computed embeddings are available so the expensive
    ``from_pretrained`` call is skipped entirely.
    """
    return LlavaOnevisionHeadOnly(num_labels=len(label2id)), None


def build_model(
    model_ckpt: str,
    label2id: dict,
    id2label: dict,
) -> tuple[LlavaOnevisionClassifier, LlavaOnevisionProcessor]:
    processor = LlavaOnevisionProcessor.from_pretrained(model_ckpt)
    base = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        ignore_mismatched_sizes=True,
    )
    model = LlavaOnevisionClassifier(base, num_labels=len(label2id))
    return model, processor


def apply_freeze_strategy(model: LlavaOnevisionClassifier, strategy: str) -> None:
    if strategy == "head_only":
        for p in model.base.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
    # "full" — all parameters remain trainable (default)


def get_video_params(processor: LlavaOnevisionProcessor) -> tuple:
    ip = processor.image_processor
    mean = ip.image_mean
    std = ip.image_std
    # Size may be stored under different keys depending on processor version.
    size_cfg = ip.size
    if isinstance(size_cfg, dict):
        h = size_cfg.get("height") or size_cfg.get("shortest_edge", 224)
    else:
        h = 224
    return mean, std, (h, h)
