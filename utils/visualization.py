"""Utilities for visualising video clips as animated GIFs and flow grids."""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import imageio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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


def _flow_direction_wheel_rgb(size: int = 160) -> np.ndarray:
    """HSV hue wheel (direction) for legend; RGB uint8 (size, size, 3)."""
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    r = np.sqrt(x**2 + y**2)
    mask = r <= 1.0
    angle = np.arctan2(y, x)
    hue = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    val = np.clip(r * 255, 0, 255).astype(np.uint8)
    hsv = np.stack([hue, np.full_like(hue, 255), val], axis=-1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb[~mask] = 255
    return rgb


def build_flow_grid_figure(
    rgb_frames: Sequence[np.ndarray],
    magnitude_maps: Sequence[Optional[np.ndarray]],
    direction_rgb: Sequence[Optional[np.ndarray]],
    frame_indices: Sequence[int],
    *,
    title: str = "barbell bicep curl",
) -> plt.Figure:
    """3×N grid: RGB | magnitude | direction (HSV as RGB). Compact layout.

    Args:
        rgb_frames: One (H, W, 3) uint8 RGB image per column.
        magnitude_maps: Grayscale magnitude in [0, 1] or None for N/A.
        direction_rgb: Flow direction as RGB uint8 or None.
        frame_indices: Shown above each column (e.g. frame numbers).
        title: Figure title (default human-readable exercise name).
    """
    n = len(frame_indices)
    if not (len(rgb_frames) == len(magnitude_maps) == len(direction_rgb) == n):
        raise ValueError("rgb_frames, magnitude_maps, direction_rgb, and frame_indices must match in length.")

    row_labels = ["RGB", "Magnitude", "Direction"]

    # Tight geometry: slim label column + equal image columns
    label_w = 0.11
    cell_w = 1.0
    fig_w = max(4.0, label_w + n * cell_w)
    fig_h = 3.0 * 0.92 + 0.55
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#d0d0d0",
        "axes.linewidth": 0.6,
        "figure.facecolor": "#fafafa",
        "axes.facecolor": "#fafafa",
    }

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
        width_ratios = [label_w] + [cell_w] * n
        gs = gridspec.GridSpec(
            3,
            n + 1,
            figure=fig,
            width_ratios=width_ratios,
            height_ratios=[1, 1, 1],
            wspace=0.06,
            hspace=0.12,
            left=0.02,
            right=0.99,
            top=0.91,
            bottom=0.06,
        )

        fig.suptitle(
            title,
            fontsize=13,
            fontweight="semibold",
            color="#1a1a1a",
            y=0.97,
        )

        axes_grid: List[List[plt.Axes]] = []

        for row in range(3):
            row_axes: List[plt.Axes] = []
            ax_lab = fig.add_subplot(gs[row, 0])
            ax_lab.set_axis_off()
            ax_lab.text(
                0.96,
                0.5,
                row_labels[row],
                ha="right",
                va="center",
                fontsize=9,
                color="#444444",
                transform=ax_lab.transAxes,
            )
            for col in range(n):
                ax = fig.add_subplot(gs[row, col + 1])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                row_axes.append(ax)
            axes_grid.append(row_axes)

        for col, idx in enumerate(frame_indices):
            rgb = rgb_frames[col]
            mag = magnitude_maps[col]
            hsv_rgb = direction_rgb[col]

            axes_grid[0][col].imshow(rgb)
            axes_grid[0][col].set_title(
                f"{idx}",
                fontsize=8,
                color="#333333",
                pad=3,
            )

            if mag is not None:
                axes_grid[1][col].imshow(mag, cmap="gray", vmin=0, vmax=1)
            else:
                axes_grid[1][col].text(
                    0.5, 0.5, "N/A", ha="center", va="center", transform=axes_grid[1][col].transAxes, fontsize=8, color="#888888"
                )

            if hsv_rgb is not None:
                axes_grid[2][col].imshow(hsv_rgb)
            else:
                axes_grid[2][col].text(
                    0.5, 0.5, "N/A", ha="center", va="center", transform=axes_grid[2][col].transAxes, fontsize=8, color="#888888"
                )

        # Direction legend inset on bottom-right panel
        br_ax = axes_grid[2][n - 1]
        inset = inset_axes(br_ax, width="20%", height="20%", loc="lower right", borderpad=0.35)
        inset.imshow(_flow_direction_wheel_rgb(140), origin="upper")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title("dir.", fontsize=5, color="#555555", pad=1)
        for spine in inset.spines.values():
            spine.set_visible(False)

    return fig
