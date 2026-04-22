"""Cached embedding dataset for frozen-encoder models (e.g. LLaVA-OneVision).

Instead of decoding video and running a heavy encoder every step, load
pre-computed embeddings from disk.  Use ``scripts/precompute_llava_embeddings.py``
to populate the cache before training.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def _cache_path(video_path: str, cache_dir: Path) -> Path:
    """Return the .pt cache file path for a given video."""
    p = Path(video_path)
    class_name = p.parent.name
    return cache_dir / class_name / f"{p.stem}.pt"


class EmbeddingDataset(Dataset):
    """Dataset that serves pre-computed embeddings instead of raw video.

    Args:
        labeled_paths: List of ``(video_path_str, {"label": int})`` — same
            format used by :class:`~utils.dataset.VideoClipDataset`.
        cache_dir:     Root directory written by
            ``scripts/precompute_llava_embeddings.py``.
    """

    def __init__(
        self,
        labeled_paths: List[Tuple[str, Dict]],
        cache_dir: Path,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        missing = []
        self.labeled_paths: List[Tuple[str, Dict]] = []
        for path_str, info in labeled_paths:
            cp = _cache_path(path_str, self.cache_dir)
            if cp.exists():
                self.labeled_paths.append((path_str, info))
            else:
                missing.append(path_str)

        if missing:
            print(
                f"[EmbeddingDataset] WARNING: {len(missing)} video(s) have no cache file "
                f"and will be skipped. Run scripts/precompute_llava_embeddings.py first."
            )
        if not self.labeled_paths:
            raise RuntimeError(
                f"No cached embeddings found under {self.cache_dir}. "
                "Run scripts/precompute_llava_embeddings.py before training."
            )

    def __len__(self) -> int:
        return len(self.labeled_paths)

    @property
    def num_videos(self) -> int:
        return len(self.labeled_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path_str, info = self.labeled_paths[idx]
        cp = _cache_path(path_str, self.cache_dir)
        data = torch.load(cp, weights_only=True)
        return {"embedding": data["embedding"], "label": data["label"]}
