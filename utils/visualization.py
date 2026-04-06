"""Utilities for visualising video clips as animated GIFs."""

from pathlib import Path
from typing import Sequence, Tuple, Union

import imageio
import numpy as np
import torch


def _unnormalize_frame(
    frame: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """Reverse ImageNet-style normalisation and return uint8 HWC image."""
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    frame = (frame * std_arr) + mean_arr
    frame = (frame * 255).astype(np.uint8)
    return frame.clip(0, 255)


def save_gif(
    video_tensor: torch.Tensor,
    filename: Union[str, Path],
    mean: Sequence[float] = (0.45, 0.45, 0.45),
    std: Sequence[float] = (0.225, 0.225, 0.225),
    fps: float = 4.0,
) -> Path:
    """Save a video tensor as an animated GIF.

    Args:
        video_tensor: Shape ``(C, T, H, W)`` or ``(T, C, H, W)``.
        filename:     Output path (will be created).
        mean:         Per-channel mean used during normalisation.
        std:          Per-channel std used during normalisation.
        fps:          Frame rate of the output GIF.

    Returns:
        Resolved path to the saved GIF.
    """
    # Accept both (C, T, H, W) and (T, C, H, W)
    if video_tensor.shape[0] in (1, 3):
        # Assume (C, T, H, W) → (T, C, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)

    frames = []
    for frame_tensor in video_tensor:  # (C, H, W)
        frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        frames.append(_unnormalize_frame(frame_np, mean, std))

    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, format="GIF", duration=1.0 / fps)
    return out_path.resolve()


def display_gif(
    video_tensor: torch.Tensor,
    gif_path: Union[str, Path] = "sample.gif",
    mean: Sequence[float] = (0.45, 0.45, 0.45),
    std: Sequence[float] = (0.225, 0.225, 0.225),
    fps: float = 4.0,
):
    """Save and display a GIF inline (Jupyter-compatible).

    Returns an ``IPython.display.Image`` object when called in a notebook,
    or ``None`` when called outside a notebook environment.
    """
    path = save_gif(video_tensor, gif_path, mean=mean, std=std, fps=fps)
    try:
        from IPython.display import Image

        return Image(filename=str(path))
    except ImportError:
        print(f"GIF saved to {path}")
        return None
