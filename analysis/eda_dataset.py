"""Exploratory helpers for the re-encoded video trees under `data/reencoded`.

Paths align with :class:`config.base_config.BaseConfig` (``data_btc_10s``,
``data_crawl_10s``, ``test`` — all re-encoded to 256 px short-side, 15 fps).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, TypeVar

import av
import numpy as np
from av.error import FFmpegError
from PIL import Image

# Smaller probe / analyze windows → faster demux on large dirs (FFmpeg input options).
# See https://ffmpeg.org/ffmpeg-formats.html#Format-options — values are strings.
FAST_PROBE_OPTIONS: Dict[str, str] = {
    "probesize": str(512 * 1024),  # 512 KiB
    "analyzeduration": "1000000",  # 1 s (microseconds)
}

T = TypeVar("T")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Mirrors config/base_config.py defaults (reencoded at 256px short-side, 15 fps)
DEFAULT_PATHS: Dict[str, Path] = {
    "verified_btc": REPO_ROOT / "data/reencoded/data_btc_10s",
    "verified_crawl": REPO_ROOT / "data/reencoded/data_crawl_10s",
    "test": REPO_ROOT / "data/reencoded/test",
}


@dataclass(frozen=True)
class VideoRecord:
    path: Path
    class_name: str
    source: str


def iter_video_records(
    paths: Optional[Dict[str, Path]] = None,
    *,
    extensions: tuple[str, ...] = (".mp4", ".avi", ".mov"),
) -> Iterator[VideoRecord]:
    roots = paths or DEFAULT_PATHS
    for source_key, root in roots.items():
        if not root.is_dir():
            continue
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            for p in sorted(class_dir.iterdir()):
                if p.suffix.lower() in extensions:
                    yield VideoRecord(
                        path=p.resolve(),
                        class_name=class_dir.name,
                        source=source_key,
                    )


def video_metadata(
    path: Path,
    *,
    fast_probe: bool = True,
    open_options: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Return duration (seconds), fps, native width/height; duration may be None.

    Corrupt or unreadable files return null fields and ``probe_error`` set instead of raising.

    Args:
        path: Video file path.
        fast_probe: If True (default), pass ``FAST_PROBE_OPTIONS`` to ``av.open`` for quicker probing.
        open_options: Optional FFmpeg demuxer options; overrides ``fast_probe`` when set.
    """
    out: Dict[str, Any] = {
        "duration_sec": None,
        "fps": None,
        "width": None,
        "height": None,
        "probe_error": None,
    }
    opts: Optional[Dict[str, str]] = None
    if open_options is not None:
        opts = dict(open_options)
    elif fast_probe:
        opts = dict(FAST_PROBE_OPTIONS)

    try:
        open_kw: Dict[str, Any] = {}
        if opts:
            open_kw["options"] = opts
        with av.open(str(path), **open_kw) as container:
            if not container.streams.video:
                out["probe_error"] = "no_video_stream"
                return out
            stream = container.streams.video[0]
            out["width"] = stream.width
            out["height"] = stream.height
            if stream.average_rate:
                out["fps"] = float(stream.average_rate)
            else:
                out["fps"] = 30.0
            if container.duration:
                out["duration_sec"] = float(container.duration) / av.time_base
            elif stream.duration is not None and stream.time_base is not None:
                out["duration_sec"] = float(stream.duration * stream.time_base)
    except FFmpegError as exc:
        out["probe_error"] = str(exc)
    except OSError as exc:
        out["probe_error"] = str(exc)
    return out


def dataframe_row_from_record(
    r: VideoRecord,
    *,
    fast_probe: bool = True,
    open_options: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """One flat dict per video for :class:`pandas.DataFrame` rows (includes ``probe_error``)."""
    meta = video_metadata(r.path, fast_probe=fast_probe, open_options=open_options)
    w, h = meta["width"], meta["height"]
    if w is not None and h is not None and w > 0 and h > 0:
        short = float(min(w, h))
    else:
        short = float("nan")
    return {
        "path": str(r.path),
        "class": r.class_name,
        "source": r.source,
        "duration_sec": meta["duration_sec"],
        "fps": meta["fps"],
        "width": meta["width"],
        "height": meta["height"],
        "short_side": short,
        "probe_error": meta["probe_error"],
    }


def map_parallel(
    items: Sequence[T],
    fn: Callable[[T], Any],
    *,
    max_workers: Optional[int] = None,
    chunksize: int = 1,
) -> List[Any]:
    """Run ``fn`` over ``items`` in a thread pool (good for I/O-bound PyAV probes)."""
    if not items:
        return []
    n = len(items)
    workers = max_workers if max_workers is not None else min(32, max(4, (os.cpu_count() or 4) * 2))
    workers = max(1, min(workers, n))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items, chunksize=max(chunksize, 1)))


def _resize_short_side(rgb: np.ndarray, max_short_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    short = min(h, w)
    if short <= max_short_side:
        return rgb
    scale = max_short_side / short
    nw, nh = int(w * scale), int(h * scale)
    pil = Image.fromarray(rgb)
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    pil = pil.resize((nw, nh), resample)
    return np.asarray(pil)


def decode_sample_frames(
    path: Path,
    *,
    max_short_side: int = 320,
    max_frames: int = 8,
    max_decode: int = 400,
) -> List[np.ndarray]:
    """Decode up to ``max_decode`` RGB frames, resize so short side ≤ ``max_short_side``, then subsample to ``max_frames``."""
    raw: List[np.ndarray] = []
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return []
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                rgb = frame.to_ndarray(format="rgb24")
                rgb = _resize_short_side(rgb, max_short_side)
                raw.append(rgb)
                if len(raw) >= max_decode:
                    break
    except FFmpegError:
        return []
    except OSError:
        return []
    if not raw:
        return []
    if len(raw) <= max_frames:
        return raw
    idx = np.linspace(0, len(raw) - 1, max_frames).astype(int)
    return [raw[i] for i in idx]
