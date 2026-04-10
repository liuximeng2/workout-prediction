"""Video pre-processing transforms for training and validation/test splits."""

import random
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)


class UniformTemporalSubsample:
    """Uniformly subsample ``num_frames`` frames from a (C, T, H, W) tensor."""

    def __init__(self, num_frames: int) -> None:
        self.num_frames = num_frames

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        indices = torch.linspace(0, t - 1, self.num_frames).long()
        return x[:, indices]


class Normalize:
    """Normalize a (C, T, H, W) float tensor with per-channel mean and std."""

    def __init__(self, mean: Tuple[float, ...], std: Tuple[float, ...]) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


class ShortSideScale:
    """Resize the short spatial side of a (C, T, H, W) tensor to ``size`` pixels."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h <= w:
            new_h, new_w = self.size, max(1, int(w * self.size / h))
        else:
            new_h, new_w = max(1, int(h * self.size / w)), self.size
        # interpolate expects (N, C, H, W); treat T as the batch dim.
        return F.interpolate(
            x.permute(1, 0, 2, 3),  # (T, C, H, W)
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3)  # (C, T, H, W)


class RandomShortSideScale:
    """Randomly resize the short side to a value in [``min_size``, ``max_size``]."""

    def __init__(self, min_size: int, max_size: int) -> None:
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        size = random.randint(self.min_size, self.max_size)
        return ShortSideScale(size)(x)


def uint8_to_float01(x: torch.Tensor) -> torch.Tensor:
    """Scale values from [0, 255] to [0, 1]. Module-level for DataLoader pickling."""

    return x / 255.0


class ApplyTransformToKey:
    """Apply ``transform`` to ``sample[key]`` and return the updated dict."""

    def __init__(self, key: str, transform: Callable) -> None:
        self.key = key
        self.transform = transform

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[self.key] = self.transform(sample[self.key])
        return sample


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
                        Lambda(uint8_to_float01),
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
                        Lambda(uint8_to_float01),
                        Normalize(mean, std),
                        ShortSideScale(size=height),
                        CenterCrop(resize_to),
                    ]
                ),
            ),
        ]
    )
