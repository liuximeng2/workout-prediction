"""Canonical train/val/test split shared across all models.

Every model (VideoMAE, ViViT, CNN Fusion, Two-Stream, Pose) must train and
evaluate on the *same* partition of videos to make accuracy numbers
comparable. This module builds those lists once from ``BaseConfig`` settings.

Design:
    - Splits are computed from the *unfiltered* video set; auxiliary-data
      filters (pose / flow availability) are applied **per split** afterwards
      via ``filter_split``. That keeps the split composition model-agnostic —
      dropping a video because flow is missing cannot shuffle which videos
      land in train vs. val.
    - If ``test_data_roots`` points at a real directory with videos, that
      directory is the canonical test set. Otherwise the stratified hold-out
      portion of ``data_roots`` is used.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from utils.dataset import (
    _collect_labeled_paths,
    _split_paths,
    resolve_data_roots,
)

LabeledPath = Tuple[str, Dict]


def canonical_splits(
    data_roots: Sequence[Path],
    test_data_roots: Optional[Sequence[Path]],
    label2id: Dict[str, int],
    train_split: float,
    val_split: float,
    seed: int,
    *,
    repo_root: Optional[Path] = None,
) -> Tuple[List[LabeledPath], List[LabeledPath], List[LabeledPath]]:
    """Return (train, val, test) labeled path lists — identical for every model.

    The split is deterministic given ``seed`` and the contents of
    ``data_roots`` / ``test_data_roots``. No auxiliary-data filtering is
    applied here.
    """
    abs_data_roots = resolve_data_roots(data_roots, repo_root=repo_root)
    all_paths = _collect_labeled_paths(tuple(abs_data_roots), label2id)

    explicit_test: Optional[List[LabeledPath]] = None
    if test_data_roots:
        abs_test_roots = resolve_data_roots(test_data_roots, repo_root=repo_root)
        if abs_test_roots and abs_test_roots[0].is_dir():
            candidates = _collect_labeled_paths(tuple(abs_test_roots), label2id)
            if candidates:
                explicit_test = candidates

    if explicit_test is not None:
        train, val, _ = _split_paths(all_paths, train_split, val_split, seed)
        test = explicit_test
    else:
        train, val, test = _split_paths(all_paths, train_split, val_split, seed)

    return train, val, test


def filter_split(
    paths: List[LabeledPath],
    predicate: Callable[[str], bool],
    *,
    name: str = "split",
    strict: bool = False,
    verbose: bool = True,
) -> List[LabeledPath]:
    """Keep only entries whose video path satisfies ``predicate``.

    Applied *per split* (after the canonical split has been computed) so that
    models needing auxiliary data (pose keypoints, optical flow) keep the same
    split partition as every other model — they just drop samples that their
    backbone can't ingest.

    Args:
        paths: Labeled-path list for one split.
        predicate: ``path_str -> bool``; returns True if the auxiliary file exists.
        name: Split label used in log messages ("train" / "val" / "test").
        strict: If True, raise instead of silently dropping missing entries.
        verbose: Print a one-line summary.

    Returns:
        Filtered list.
    """
    kept, dropped = [], []
    for item in paths:
        path_str, _ = item
        (kept if predicate(path_str) else dropped).append(item)

    if dropped and strict:
        head = "\n  ".join(p for p, _ in dropped[:5])
        more = "" if len(dropped) <= 5 else f"\n  …and {len(dropped) - 5} more"
        raise RuntimeError(
            f"[splits] {name}: {len(dropped)} video(s) missing auxiliary data:\n  {head}{more}"
        )

    if verbose:
        print(
            f"[splits] {name}: kept {len(kept)}, dropped {len(dropped)} "
            f"(missing aux data) out of {len(paths)}."
        )
    return kept
