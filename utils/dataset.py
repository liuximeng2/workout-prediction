"""Dataset utilities: label maps, train/val/test splitting, dataset construction."""

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import av
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


def resolve_data_roots(
    data_roots: Sequence[Path],
    *,
    repo_root: Optional[Path] = None,
) -> Tuple[Path, ...]:
    """Make ``BaseConfig.data_roots`` absolute under the project root.

    Config paths are relative to the repo root. Resolving them against
    ``Path.cwd()`` fails when the process cwd is elsewhere (e.g. a Jupyter
    notebook opened from ``analysis/``).
    """
    root = repo_root if repo_root is not None else Path(__file__).resolve().parent.parent
    resolved: list[Path] = []
    for p in data_roots:
        p = Path(p)
        resolved.append(p.resolve() if p.is_absolute() else (root / p).resolve())
    return tuple(resolved)


def resolve_roots_for_label_maps(
    data_roots: Sequence[Path],
    test_data_roots: Optional[Sequence[Path]],
    *,
    repo_root: Optional[Path] = None,
) -> Tuple[Path, ...]:
    """Directories scanned for class-folder names: training roots plus optional ``data/test``-style tree."""
    roots = list(resolve_data_roots(data_roots, repo_root=repo_root))
    if test_data_roots:
        extra = resolve_data_roots(test_data_roots, repo_root=repo_root)
        if extra and extra[0].is_dir():
            roots.extend(extra)
    return tuple(roots)


def build_label_maps(data_roots: Sequence[Path]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Derive label↔id mappings from class subdirectories across one or more roots.

    Args:
        data_roots: Each path is a directory that contains one sub-folder per class.
            Class names are the union of folder names across all roots (sorted).

    Returns:
        ``(label2id, id2label)`` dictionaries.
    """
    names: Set[str] = set()
    for data_root in data_roots:
        for p in data_root.iterdir():
            if p.is_dir():
                names.add(p.name)
    class_names = sorted(names)
    label2id = {name: idx for idx, name in enumerate(class_names)}
    id2label = {idx: name for name, idx in label2id.items()}
    return label2id, id2label


def _collect_labeled_paths(
    data_roots: Sequence[Path],
    label2id: Dict[str, int],
    extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov"),
) -> List[Tuple[str, Dict]]:
    """Return a list of ``(video_path_str, {"label": int})`` tuples from all roots."""
    labeled: List[Tuple[str, Dict]] = []
    for data_root in data_roots:
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
    data_roots: Sequence[Path],
    label2id: Dict[str, int],
    clip_duration: float,
    train_transform: Compose,
    val_transform: Compose,
    train_split: float = 0.70,
    val_split: float = 0.15,
    seed: int = 42,
    test_data_roots: Optional[Sequence[Path]] = None,
) -> Tuple[VideoClipDataset, VideoClipDataset, VideoClipDataset]:
    """Build train, val, and test ``VideoClipDataset`` objects.

    Args:
        data_roots:      One or more roots, each with one sub-folder per class.
        label2id:        Mapping from class name to integer id.
        clip_duration:   Duration (seconds) of each sampled clip.
        train_transform: Transform applied to training clips.
        val_transform:   Transform applied to val/test clips.
        train_split:     Fraction of videos assigned to training.
        val_split:       Fraction of videos assigned to validation.
        seed:            Random seed for the split.
        test_data_roots: If set, resolved absolute paths to a dedicated test tree
            (``…/test/<class>/*.mp4``). When the first path exists and yields at least
            one video, that list is the test set; otherwise the stratified test split
            from ``data_roots`` is used.

    Returns:
        ``(train_dataset, val_dataset, test_dataset)``
    """
    all_paths = _collect_labeled_paths(data_roots, label2id)
    use_explicit_test = False
    if test_data_roots and test_data_roots[0].is_dir():
        explicit_test = _collect_labeled_paths(tuple(test_data_roots), label2id)
        if len(explicit_test) > 0:
            use_explicit_test = True
            train_paths, val_paths, _ = _split_paths(all_paths, train_split, val_split, seed)
            test_paths = explicit_test

    if not use_explicit_test:
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
