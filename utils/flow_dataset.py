"""Two-stream dataset: returns RGB frames and stacked optical flow for the same clip.

Design notes
------------
- The same random clip start (temporal) is used for both RGB and flow.
- The same random spatial transform parameters (crop box, flip) are applied to
  both streams so they stay spatially aligned.
- Horizontal flip negates the u (horizontal) flow component.
- Flow is loaded from pre-computed .npy files written by scripts/precompute_flow.py.
- Each flow .npy is uint8 (2, H, W); decoded back to float32 in pixel units.
"""

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import av
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.dataset import FLOW_CLIP, flow_dir_for_video


# ── Flow decode ───────────────────────────────────────────────────────────────

def _load_flow_frames(flow_dir: Path, frame_indices: List[int]) -> torch.Tensor:
    """Load and decode flow frames at given indices → float32 tensor (2*L, H, W).

    ``frame_indices`` selects which flow transitions to load (flow index i
    corresponds to the transition from raw frame i → frame i+1).

    Returns a tensor of shape (2*L, H, W) where L = len(frame_indices),
    channels ordered as [u0, v0, u1, v1, ...].
    """
    channels = []
    for idx in frame_indices:
        path = flow_dir / f"frame_{idx:04d}.npy"
        encoded = np.load(path)                                      # (2, H, W) uint8
        flow = encoded.astype(np.float32) / 255.0 * (2 * FLOW_CLIP) - FLOW_CLIP
        channels.append(torch.from_numpy(flow))                      # (2, H, W)
    return torch.cat(channels, dim=0)                                # (2*L, H, W)


def _uniform_indices(n_available: int, n_select: int) -> List[int]:
    """Pick n_select evenly-spaced indices from [0, n_available)."""
    return torch.linspace(0, n_available - 1, n_select).long().tolist()


# ── Spatial transforms applied identically to both streams ────────────────────

def _short_side_scale(x: torch.Tensor, size: int) -> torch.Tensor:
    """Resize so the short side == size. x: (C, H, W)."""
    c, h, w = x.shape
    if h <= w:
        new_h, new_w = size, max(1, int(w * size / h))
    else:
        new_h, new_w = max(1, int(h * size / w)), size
    return F.interpolate(
        x.unsqueeze(0).float(), size=(new_h, new_w),
        mode="bilinear", align_corners=False,
    ).squeeze(0)


def _apply_spatial_train(
    rgb: torch.Tensor,       # (C, H, W)
    flow: torch.Tensor,      # (2*L, H, W)
    min_size: int = 256,
    max_size: int = 320,
    crop_size: int = 224,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random short-side scale → random crop → random flip, shared params."""
    scale = random.randint(min_size, max_size)
    rgb = _short_side_scale(rgb, scale)
    flow = _short_side_scale(flow, scale)

    _, h, w = rgb.shape
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    rgb = rgb[:, top:top + crop_size, left:left + crop_size]
    flow = flow[:, top:top + crop_size, left:left + crop_size]

    if random.random() < 0.5:
        rgb = rgb.flip(-1)
        flow = flow.flip(-1)
        # Negate horizontal (u) channels: indices 0, 2, 4, ...
        flow[0::2] = -flow[0::2]

    return rgb, flow


def _apply_spatial_val(
    rgb: torch.Tensor,
    flow: torch.Tensor,
    size: int = 224,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic short-side scale → center crop."""
    rgb = _short_side_scale(rgb, size)
    flow = _short_side_scale(flow, size)

    _, h, w = rgb.shape
    top = (h - size) // 2
    left = (w - size) // 2
    rgb = rgb[:, top:top + size, left:left + size]
    flow = flow[:, top:top + size, left:left + size]
    return rgb, flow


# ── Dataset ───────────────────────────────────────────────────────────────────

class TwoStreamDataset(Dataset):
    """Dataset that returns RGB clip + stacked optical flow for the same clip.

    Each sample dict contains:
        "video"  : float32 tensor (3, H, W)  — mean-pooled RGB frame (spatial stream input)
        "flow"   : float32 tensor (2*L, H, W) — stacked u/v flow (temporal stream input)
        "label"  : int

    The spatial stream receives a single representative RGB frame (mean of the
    clip) rather than the full (C, T, H, W) volume, matching the original
    Two-Stream paper which processes individual frames through the spatial CNN.
    The temporal stream receives L=``num_flow_frames`` stacked flow frames.

    Args:
        labeled_paths:   List of ``(video_path_str, {"label": int})`` tuples.
                         Should already be filtered by ``filter_valid_flow``.
        flow_root:       Root directory of pre-computed flow files.
        clip_duration:   Clip length in seconds (used for temporal clip sampling).
        num_flow_frames: Number of consecutive flow frames to stack (default 10).
        mode:            ``"train"`` (random augmentation) or ``"val"`` (center crop).
        mean:            RGB normalisation mean (ImageNet default).
        std:             RGB normalisation std (ImageNet default).
        repo_root:       Repo root for reconstructing flow paths (auto-detected).
    """

    RGB_MEAN = (0.485, 0.456, 0.406)
    RGB_STD  = (0.229, 0.224, 0.225)
    FLOW_MEAN = 0.5   # flow is normalised to [0,1] range after decoding
    FLOW_STD  = 0.5

    def __init__(
        self,
        labeled_paths: List[Tuple[str, Dict]],
        flow_root: Path,
        clip_duration: float,
        num_flow_frames: int = 10,
        mode: str = "train",
        mean: Tuple[float, ...] = RGB_MEAN,
        std: Tuple[float, ...] = RGB_STD,
        repo_root: Optional[Path] = None,
        clip_position: Optional[float] = None,
    ) -> None:
        self.labeled_paths = labeled_paths
        self.flow_root = Path(flow_root)
        self.clip_duration = clip_duration
        self.num_flow_frames = num_flow_frames
        self.mode = mode
        self.repo_root = repo_root
        self.clip_position = clip_position

        self._rgb_mean = torch.tensor(mean).view(3, 1, 1)
        self._rgb_std  = torch.tensor(std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.labeled_paths)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _decode_rgb_clip(self, path: str) -> Tuple[torch.Tensor, float, float]:
        """Decode raw RGB frames in the clip window.

        Returns:
            frames: (C, T, H, W) float32 tensor in [0, 1]
            clip_start: actual start time used
            fps: video frame rate
        """
        with av.open(path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 30.0
            if container.duration:
                # container.duration is in AV_TIME_BASE units (microseconds).
                duration = float(container.duration) * float(av.time_base)
            elif stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            else:
                duration = self.clip_duration

            max_start = max(0.0, duration - self.clip_duration)
            if self.mode == "train":
                start = random.uniform(0.0, max_start)
            elif self.clip_position is not None:
                start = max_start * self.clip_position
            else:
                start = max_start / 2.0

            seek_pts = int(start / stream.time_base)
            container.seek(seek_pts, stream=stream)

            frames = []
            for frame in container.decode(stream):
                t = float(frame.pts * stream.time_base)
                if t < start:
                    continue
                if t > start + self.clip_duration:
                    break
                frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))

        if not frames:
            with av.open(path) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate) if stream.average_rate else 30.0
                start = 0.0
                for frame in container.decode(stream):
                    frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
                    if len(frames) >= max(1, int(fps * self.clip_duration)):
                        break

        # (T, H, W, C) → (C, T, H, W), normalise to [0, 1]
        video = torch.stack(frames).permute(3, 0, 1, 2).float() / 255.0
        return video, start, fps

    def _pick_flow_indices(
        self, clip_start: float, fps: float, flow_dir: Path
    ) -> List[int]:
        """Select num_flow_frames flow indices that align with the clip window."""
        meta = np.load(flow_dir / "meta.npy")
        n_flow = int(meta[0])

        # First flow index corresponding to clip_start
        start_idx = max(0, int(clip_start * fps))
        # Clip window spans roughly clip_duration * fps frames
        end_idx = min(n_flow - 1, start_idx + int(self.clip_duration * fps))

        available = max(1, end_idx - start_idx)
        offsets = torch.linspace(0, available - 1, self.num_flow_frames).long().tolist()
        return [min(start_idx + o, n_flow - 1) for o in offsets]

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def _get_sample(self, idx: int) -> Dict[str, Any]:
        path, info = self.labeled_paths[idx]
        label = info["label"]

        flow_dir = flow_dir_for_video(Path(path), self.flow_root, self.repo_root)

        # 1. Decode RGB clip
        rgb_clip, clip_start, fps = self._decode_rgb_clip(path)  # (C, T, H, W)

        # 2. Pick a single representative RGB frame (middle of clip) → (C, H, W)
        t = rgb_clip.shape[1]
        rgb_frame = rgb_clip[:, t // 2]   # (C, H, W)

        # 3. Load flow frames aligned to the same clip window
        flow_indices = self._pick_flow_indices(clip_start, fps, flow_dir)
        flow = _load_flow_frames(flow_dir, flow_indices)    # (2*L, H, W)

        # 4. Shared spatial transforms
        if self.mode == "train":
            rgb_frame, flow = _apply_spatial_train(rgb_frame, flow)
        else:
            rgb_frame, flow = _apply_spatial_val(rgb_frame, flow)

        # 5. Normalise RGB with ImageNet stats
        rgb_frame = (rgb_frame - self._rgb_mean) / self._rgb_std

        # 6. Normalise flow to [-1, 1] (already in pixel units, clip to FLOW_CLIP)
        flow = torch.clamp(flow / FLOW_CLIP, -1.0, 1.0)

        return {"video": rgb_frame, "flow": flow, "label": label}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load sample; on corrupt/unreadable video, try other indices (PyAV FFmpeg errors)."""
        n = len(self.labeled_paths)
        if n == 0:
            raise IndexError("empty dataset")
        max_attempts = min(n, 64)
        last: Optional[BaseException] = None
        for k in range(max_attempts):
            j = (idx + k) % n
            try:
                return self._get_sample(j)
            except av.error.FFmpegError as e:
                last = e
                continue
        raise RuntimeError(
            f"Could not decode any of {max_attempts} consecutive videos starting at index {idx}"
        ) from last
