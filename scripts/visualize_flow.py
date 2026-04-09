"""Visualize a video clip alongside its pre-computed optical flow.

For a given video file, this script:
  1. Decodes RGB frames from the raw video.
  2. Loads the corresponding pre-computed flow (.npy files).
  3. Renders a side-by-side figure: RGB | flow-magnitude | flow-HSV color wheel.
  4. Saves to a PNG under analysis/visualization/ (default) or displays interactively.
  5. Optionally exports an animated GIF cycling through all frames.

Flow is visualized two ways:
  - Magnitude map (grayscale): brighter = faster motion.
  - HSV color wheel: hue = direction, saturation = magnitude.

Usage:
    python scripts/visualize_flow.py <path/to/video.mp4>
    python scripts/visualize_flow.py <path/to/video.mp4> --flow_root data/flow
    python scripts/visualize_flow.py <path/to/video.mp4> --num_frames 8 --show
    python scripts/visualize_flow.py <path/to/video.mp4> --out my_viz.png   # custom path
    python scripts/visualize_flow.py <path/to/video.mp4> --gif
    python scripts/visualize_flow.py "data/verified_data/verified_data/data_btc_10s/barbell biceps curl/0b43a151-8995-4f7e-8568-45d65996a19c.mp4" --gif --gif_fps 10 --num_frames 30
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import av
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scripts.precompute_flow import FLOW_CLIP
from utils.dataset import flow_dir_for_video

# ── Helpers ───────────────────────────────────────────────────────────────────


def _figure_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    """Rasterize the current figure to (H, W, 3) uint8 RGB.

    Uses buffer_rgba() when available (macOS / newer matplotlib); avoids
    FigureCanvasMac lacking tostring_rgb().
    """
    fig.canvas.draw()
    canvas = fig.canvas
    if hasattr(canvas, "buffer_rgba"):
        buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        return np.ascontiguousarray(buf[..., :3])
    w, h = canvas.get_width_height()
    raw = canvas.tostring_rgb()
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)

def decode_rgb_frames(video_path: Path, max_short_side: int = 320) -> list[np.ndarray]:
    """Decode all frames from a video, resize to max_short_side, return list of (H,W,3) uint8."""
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            rgb = frame.to_ndarray(format="rgb24")
            h, w = rgb.shape[:2]
            short = min(h, w)
            if short > max_short_side:
                scale = max_short_side / short
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            frames.append(rgb)
    return frames


def load_flow_frame(flow_dir: Path, idx: int) -> np.ndarray:
    """Load and decode a single flow frame → float32 (H, W, 2) in pixel units."""
    path = flow_dir / f"frame_{idx:04d}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Flow frame not found: {path}")
    encoded = np.load(path)                                  # (2, H, W) uint8
    flow = encoded.astype(np.float32) / 255.0 * (2 * FLOW_CLIP) - FLOW_CLIP
    return flow.transpose(1, 2, 0)                          # (H, W, 2)


def flow_to_magnitude(flow: np.ndarray) -> np.ndarray:
    """Compute magnitude map, normalized to [0, 1]. Shape: (H, W)."""
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return np.clip(mag / FLOW_CLIP, 0.0, 1.0)


def flow_to_hsv(flow: np.ndarray) -> np.ndarray:
    """Convert flow (H,W,2) to an HSV color image (H,W,3) uint8.

    Hue encodes direction (angle), saturation=1, value encodes magnitude.
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    u, v = flow[..., 0], flow[..., 1]
    angle = np.arctan2(v, u)                       # radians in [-π, π]
    hue = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)   # OpenCV hue [0,179]
    mag = np.sqrt(u ** 2 + v ** 2)
    val = np.clip(mag / FLOW_CLIP * 255, 0, 255).astype(np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = 255                              # full saturation
    hsv[..., 2] = val
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def find_flow_dir(video_path: Path, flow_root: Path) -> Path:
    return flow_dir_for_video(video_path, flow_root)


def select_frame_indices(n_rgb: int, n_flow: int, num_frames: int) -> list[int]:
    """Pick num_frames evenly-spaced frame indices valid for both RGB and flow."""
    # flow index i = transition i→i+1, so valid rgb indices are 0..n_flow-1
    valid = min(n_rgb - 1, n_flow)
    indices = np.linspace(0, valid - 1, num_frames, dtype=int).tolist()
    return indices


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize video frames and optical flow side by side.")
    parser.add_argument("video", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "--flow_root", type=Path, default=Path("data/flow"),
        help="Root directory containing pre-computed flow (default: data/flow).",
    )
    parser.add_argument(
        "--num_frames", type=int, default=8,
        help="Number of frames to display (default: 8).",
    )
    parser.add_argument(
        "--max_short_side", type=int, default=320,
        help="Short-side resize for display (default: 320, matches precompute).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Save figure to this path (e.g. viz.png). If omitted, saves to "
             "analysis/visualization/<video_stem>_flow_viz.png (under the repo root).",
    )
    parser.add_argument(
        "--show", action="store_false",
        help="Display the figure interactively (in addition to saving).",
    )
    parser.add_argument(
        "--gif", action="store_true",
        help="Also export an animated GIF cycling through frames (3 panels per frame).",
    )
    parser.add_argument(
        "--gif_fps", type=float, default=8.0,
        help="Frame rate for the output GIF (default: 8).",
    )
    args = parser.parse_args()

    video_path = args.video.resolve()
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    flow_root = (Path(__file__).resolve().parent.parent / args.flow_root
                 if not args.flow_root.is_absolute() else args.flow_root)
    flow_dir = find_flow_dir(video_path, flow_root)

    if not flow_dir.exists():
        print(f"Error: flow directory not found: {flow_dir}")
        print("Run scripts/precompute_flow.py first.")
        sys.exit(1)

    meta_path = flow_dir / "meta.npy"
    if not meta_path.exists():
        print(f"Error: meta.npy missing in {flow_dir} — flow may be incomplete.")
        sys.exit(1)

    meta = np.load(meta_path)
    n_flow = int(meta[0])
    print(f"Video     : {video_path.name}")
    print(f"Flow dir  : {flow_dir}")
    print(f"Flow frames available: {n_flow}")

    # Decode RGB
    print("Decoding video frames...")
    rgb_frames = decode_rgb_frames(video_path, args.max_short_side)
    print(f"RGB frames decoded : {len(rgb_frames)}")

    # Select frame indices
    frame_indices = select_frame_indices(len(rgb_frames), n_flow, args.num_frames)
    n = len(frame_indices)

    # ── Build figure ──────────────────────────────────────────────────────────
    # Rows: RGB | Flow magnitude | Flow HSV
    fig = plt.figure(figsize=(n * 2.5, 3 * 2.8))
    fig.suptitle(
        f"{video_path.stem}  —  RGB / Flow magnitude / Flow direction",
        fontsize=13, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(3, n, figure=fig, hspace=0.08, wspace=0.04)

    row_labels = ["RGB", "Magnitude", "Direction (HSV)"]
    for col, idx in enumerate(frame_indices):
        rgb = rgb_frames[idx]

        try:
            flow = load_flow_frame(flow_dir, idx)
        except FileNotFoundError:
            flow = None

        for row in range(3):
            ax = fig.add_subplot(gs[row, col])
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.imshow(rgb)
                if col == 0:
                    ax.set_ylabel(row_labels[0], fontsize=10, labelpad=4)
                ax.set_title(f"frame {idx}", fontsize=8)

            elif row == 1:
                if flow is not None:
                    mag = flow_to_magnitude(flow)
                    ax.imshow(mag, cmap="gray", vmin=0, vmax=1)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                if col == 0:
                    ax.set_ylabel(row_labels[1], fontsize=10, labelpad=4)

            else:
                if flow is not None:
                    hsv_img = flow_to_hsv(flow)
                    ax.imshow(hsv_img)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                if col == 0:
                    ax.set_ylabel(row_labels[2], fontsize=10, labelpad=4)

    # Add a small HSV color wheel legend
    _add_hsv_legend(fig)

    plt.tight_layout()

    repo_root = Path(__file__).resolve().parent.parent
    out_path = args.out
    if out_path is None:
        vis_dir = repo_root / "analysis" / "visualization"
        vis_dir.mkdir(parents=True, exist_ok=True)
        out_path = vis_dir / f"{video_path.stem}_flow_viz.png"
    else:
        out_path = out_path.expanduser()
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved → {out_path}")

    if args.gif:
        gif_path = out_path.with_suffix(".gif")
        export_gif(video_path, flow_dir, rgb_frames, n_flow, args.gif_fps, gif_path)

    # if args.show:
    #     plt.show()


def _render_gif_frame(rgb: np.ndarray, flow: np.ndarray | None, frame_idx: int) -> np.ndarray:
    """Render a single GIF frame: [RGB | magnitude | HSV] side by side as an RGB array."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    fig.suptitle(f"frame {frame_idx}", fontsize=9, y=1.0)

    axes[0].imshow(rgb)
    axes[0].set_title("RGB", fontsize=8)

    if flow is not None:
        axes[1].imshow(flow_to_magnitude(flow), cmap="gray", vmin=0, vmax=1)
        axes[2].imshow(flow_to_hsv(flow))
    else:
        for ax in axes[1:]:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
    axes[1].set_title("Magnitude", fontsize=8)
    axes[2].set_title("Direction (HSV)", fontsize=8)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    img = _figure_to_rgb_array(fig)
    plt.close(fig)
    return img


def export_gif(
    video_path: Path,
    flow_dir: Path,
    rgb_frames: list[np.ndarray],
    n_flow: int,
    fps: float,
    out_path: Path,
) -> None:
    """Export an animated GIF with all available frames."""
    n_frames = min(len(rgb_frames) - 1, n_flow)
    duration = 1.0 / fps  # seconds per frame for imageio

    print(f"Rendering {n_frames} GIF frames at {fps} fps...")
    gif_frames = []
    for i in range(n_frames):
        rgb = rgb_frames[i]
        try:
            flow = load_flow_frame(flow_dir, i)
        except FileNotFoundError:
            flow = None
        gif_frames.append(_render_gif_frame(rgb, flow, i))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_frames}")

    imageio.mimsave(out_path, gif_frames, duration=duration, loop=0)
    print(f"GIF saved → {out_path}  ({n_frames} frames, {fps} fps)")


def _add_hsv_legend(fig: plt.Figure, size: float = 0.08):
    """Draw a tiny HSV color wheel in the bottom-right corner as a direction legend."""
    ax = fig.add_axes([0.92, 0.01, size, size * fig.get_figwidth() / fig.get_figheight()])
    N = 200
    y, x = np.mgrid[-1:1:N*1j, -1:1:N*1j]
    r = np.sqrt(x**2 + y**2)
    mask = r <= 1.0
    angle = np.arctan2(y, x)
    hue = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    val = np.clip(r * 255, 0, 255).astype(np.uint8)
    hsv = np.stack([hue, np.full_like(hue, 255), val], axis=-1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb[~mask] = 255   # white outside circle
    ax.imshow(rgb, origin="upper")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("direction", fontsize=6, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


if __name__ == "__main__":
    main()
