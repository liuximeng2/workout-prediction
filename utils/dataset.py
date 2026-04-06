"""Dataset utilities: label maps, train/val/test splitting, dataset construction."""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import pytorchvideo.data
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
    # Group by label
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


def build_datasets(
    data_root: Path,
    label2id: Dict[str, int],
    clip_duration: float,
    train_transform: Compose,
    val_transform: Compose,
    train_split: float = 0.70,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[
    pytorchvideo.data.LabeledVideoDataset,
    pytorchvideo.data.LabeledVideoDataset,
    pytorchvideo.data.LabeledVideoDataset,
]:
    """Build train, val, and test ``LabeledVideoDataset`` objects.

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

    train_dataset = pytorchvideo.data.LabeledVideoDataset(
        labeled_video_paths=train_paths,
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )
    val_dataset = pytorchvideo.data.LabeledVideoDataset(
        labeled_video_paths=val_paths,
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )
    test_dataset = pytorchvideo.data.LabeledVideoDataset(
        labeled_video_paths=test_paths,
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    return train_dataset, val_dataset, test_dataset
