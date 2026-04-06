"""Video pre-processing transforms for training and validation/test splits."""

from typing import Tuple

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)


def make_train_transform(
    num_frames: int,
    resize_to: Tuple[int, int],
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> Compose:
    """Augmented transform pipeline used during training.

    Pipeline:
        1. Temporally subsample ``num_frames`` frames uniformly.
        2. Normalise pixel values to [0, 1].
        3. Normalise with ImageNet-style mean/std.
        4. Random short-side scale in [256, 320].
        5. Random spatial crop to ``resize_to``.
        6. Random horizontal flip (p=0.5).
    """
    return Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )


def make_val_transform(
    num_frames: int,
    resize_to: Tuple[int, int],
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> Compose:
    """Deterministic transform pipeline used during validation and testing.

    Pipeline:
        1. Temporally subsample ``num_frames`` frames uniformly.
        2. Normalise pixel values to [0, 1].
        3. Normalise with ImageNet-style mean/std.
        4. Resize to ``resize_to`` (centre crop after short-side resize).
    """
    height, _ = resize_to
    return Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        ShortSideScale(size=height),
                        CenterCrop(resize_to),
                    ]
                ),
            ),
        ]
    )
