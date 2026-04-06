"""VideoMAE model construction and image-processor helpers."""

from typing import Dict, Literal, Tuple

from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

FreezeStrategy = Literal["full", "head_only"]


def build_model(
    model_ckpt: str,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> Tuple[VideoMAEForVideoClassification, VideoMAEImageProcessor]:
    """Load a pre-trained VideoMAE checkpoint and attach a new classification head.

    Args:
        model_ckpt: HuggingFace model identifier (e.g. ``"MCG-NJU/videomae-base"``).
        label2id:   Mapping from class name to integer label.
        id2label:   Mapping from integer label to class name.

    Returns:
        ``(model, image_processor)`` ready for fine-tuning.
    """
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    return model, image_processor


def apply_freeze_strategy(
    model: VideoMAEForVideoClassification,
    strategy: FreezeStrategy,
) -> None:
    """Freeze or unfreeze model parameters according to ``strategy``.

    Strategies
    ----------
    ``"full"``
        All parameters are trainable (~86.7M). Best accuracy; needs more data
        and GPU memory.
    ``"head_only"``
        The entire VideoMAE encoder is frozen; only the 17K classification head
        is updated. Trains in seconds, minimal memory, suits small datasets.

    Args:
        model:    The loaded ``VideoMAEForVideoClassification`` instance.
        strategy: One of ``"full"`` or ``"head_only"``.
    """
    if strategy == "full":
        for param in model.parameters():
            param.requires_grad = True

    elif strategy == "head_only":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    else:
        raise ValueError(
            f"Unknown freeze_strategy '{strategy}'. Choose 'full' or 'head_only'."
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Freeze strategy: '{strategy}' — "
        f"trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.1f}%)"
    )


def get_video_params(
    image_processor: VideoMAEImageProcessor,
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[int, int]]:
    """Extract normalisation stats and spatial resolution from the image processor.

    Returns:
        ``(mean, std, (height, width))``
    """
    mean = tuple(image_processor.image_mean)
    std = tuple(image_processor.image_std)
    if "shortest_edge" in image_processor.size:
        side = image_processor.size["shortest_edge"]
        resize_to = (side, side)
    else:
        resize_to = (
            image_processor.size["height"],
            image_processor.size["width"],
        )
    return mean, std, resize_to
