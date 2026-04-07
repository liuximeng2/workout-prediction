"""Dataset utilities: label maps, train/val/test splitting, dataset construction."""

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import av
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


def build_label_maps(data_root: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Derive label↔id mappings from the class subdirectories.

    Args:
        data_root: Path to the directory that contains one sub-folder per class.

    Returns:
        ``(label2id, id2label)`` dictionaries.
    """
    class_names = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    label2id = {name: idx for idx, name in enumerate(class_names)}
    id2label = {idx: name for name, idx in label2id.items()}
    return label2id, id2label


def _collect_labeled_paths(
    data_root: Path,
    label2id: Dict[str, int],
    extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov"),
) -> List[Tuple[str, Dict]]:
    """Return a list of ``(video_path_str, {"label": int})`` tuples."""
    labeled: List[Tuple[str, Dict]] = []
    for class_dir in sorted(data_root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = label2id.get(class_dir.name)
        if label is None:
            continue
        for video_path in sorted(class_dir.iterdir()):
            if video_path.suffix.lower() in extensions:
                labeled.append((str(video_path), {"label": label}))
    return labeled


def _split_paths(
    labeled_paths: List[Tuple[str, Dict]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[
    List[Tuple[str, Dict]],
    List[Tuple[str, Dict]],
    List[Tuple[str, Dict]],
]:
    """Stratified random split into train / val / test."""
    by_label: Dict[int, List[Tuple[str, Dict]]] = {}
    for path, info in labeled_paths:
        lbl = info["label"]
        by_label.setdefault(lbl, []).append((path, info))

    rng = random.Random(seed)
    train, val, test = [], [], []
    for items in by_label.values():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    return train, val, test


class VideoClipDataset(Dataset):
    """A simple labeled video clip dataset backed by ``torchvision.io``.

    Each ``__getitem__`` call reads a clip of ``clip_duration`` seconds from a
    video file, converts it to a ``(C, T, H, W)`` float tensor, and applies
    ``transform`` to a dict ``{"video": tensor, "label": int}``.

    Args:
        labeled_paths:  List of ``(path_str, {"label": int})`` tuples.
        clip_duration:  Clip length in seconds.
        transform:      Optional transform applied to the sample dict.
        mode:           ``"random"`` samples a random start time (training);
                        ``"uniform"`` samples from the center (val/test).
    """

    def __init__(
        self,
        labeled_paths: List[Tuple[str, Dict]],
        clip_duration: float,
        transform: Optional[Callable] = None,
        mode: str = "random",
    ) -> None:
        self.labeled_paths = labeled_paths
        self.clip_duration = clip_duration
        self.transform = transform
        self.mode = mode

    def __len__(self) -> int:
        return len(self.labeled_paths)

    @property
    def num_videos(self) -> int:
        return len(self.labeled_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, info = self.labeled_paths[idx]
        label = info["label"]

        with av.open(path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 30.0
            # duration in seconds from container or stream metadata
            if container.duration:
                duration = float(container.duration) / av.time_base
            elif stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            else:
                duration = self.clip_duration

            max_start = max(0.0, duration - self.clip_duration)
            if self.mode == "random":
                start = random.uniform(0.0, max_start)
            else:
                start = max_start / 2.0

            # Seek and decode frames in the clip window
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
            # Fallback: re-open and grab whatever frames exist
            with av.open(path) as container:
                stream = container.streams.video[0]
                for frame in container.decode(stream):
                    frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
                    if len(frames) >= max(1, int(fps * self.clip_duration)):
                        break

        # Stack to (T, H, W, C) then convert to (C, T, H, W)
        video = torch.stack(frames).permute(3, 0, 1, 2).float()

        sample: Dict[str, Any] = {"video": video, "label": label}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def build_datasets(
    data_root: Path,
    label2id: Dict[str, int],
    clip_duration: float,
    train_transform: Compose,
    val_transform: Compose,
    train_split: float = 0.70,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[VideoClipDataset, VideoClipDataset, VideoClipDataset]:
    """Build train, val, and test ``VideoClipDataset`` objects.

    Args:
        data_root:       Root directory with one sub-folder per class.
        label2id:        Mapping from class name to integer id.
        clip_duration:   Duration (seconds) of each sampled clip.
        train_transform: Transform applied to training clips.
        val_transform:   Transform applied to val/test clips.
        train_split:     Fraction of videos assigned to training.
        val_split:       Fraction of videos assigned to validation.
        seed:            Random seed for the split.

    Returns:
        ``(train_dataset, val_dataset, test_dataset)``
    """
    all_paths = _collect_labeled_paths(data_root, label2id)
    train_paths, val_paths, test_paths = _split_paths(
        all_paths, train_split, val_split, seed
    )

    train_dataset = VideoClipDataset(
        labeled_paths=train_paths,
        clip_duration=clip_duration,
        transform=train_transform,
        mode="random",
    )
    val_dataset = VideoClipDataset(
        labeled_paths=val_paths,
        clip_duration=clip_duration,
        transform=val_transform,
        mode="uniform",
    )
    test_dataset = VideoClipDataset(
        labeled_paths=test_paths,
        clip_duration=clip_duration,
        transform=val_transform,
        mode="uniform",
    )

    return train_dataset, val_dataset, test_dataset
