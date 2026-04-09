"""Pre-compute Farneback optical flow for all videos and save to disk.

For each video we extract every frame at its native resolution, compute
dense optical flow between consecutive frames, and store the results as
uint8-encoded .npy files (one file per frame transition).

Storage layout mirrors the video directory structure:

    data/flow/<data_root_stem>/<class>/<video_stem>/
        frame_0000.npy   # flow from frame 0 → frame 1, shape (2, H, W), uint8
        frame_0001.npy   # flow from frame 1 → frame 2
        ...
        meta.npy         # scalar array: [n_flow_frames, orig_H, orig_W, flow_min, flow_max]

Each .npy stores two uint8 channels (u=horizontal, v=vertical).  Flow values
are clipped to [-FLOW_CLIP, +FLOW_CLIP] and linearly mapped to [0, 255].
To recover float flow at training time:
    flow_f32 = arr.astype(np.float32) / 255.0 * (2 * FLOW_CLIP) - FLOW_CLIP

Usage:
    python scripts/precompute_flow.py                   # all data roots
    python scripts/precompute_flow.py --max_short_side 320   # resize first
    python scripts/precompute_flow.py --workers 4       # parallel jobs
    python scripts/precompute_flow.py --dry_run         # print plan only
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import av
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from config.base_config import BaseConfig
from utils.dataset import resolve_data_roots

# ── Constants ─────────────────────────────────────────────────────────────────
FLOW_CLIP = 20.0          # pixel/frame displacement clipped to this magnitude
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_videos(data_roots: List[Path]) -> List[Tuple[Path, str]]:
    """Return list of (video_path, class_name) from all roots."""
    videos = []
    for root in data_roots:
        if not root.is_dir():
            continue
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            for vp in sorted(class_dir.iterdir()):
                if vp.suffix.lower() in VIDEO_EXTS:
                    videos.append((vp, class_dir.name))
    return videos


def _flow_to_uint8(flow: np.ndarray) -> np.ndarray:
    """Clip and encode float32 flow (H, W, 2) → uint8 (2, H, W)."""
    flow = np.clip(flow, -FLOW_CLIP, FLOW_CLIP)
    encoded = ((flow + FLOW_CLIP) / (2 * FLOW_CLIP) * 255.0).astype(np.uint8)
    return encoded.transpose(2, 0, 1)   # (2, H, W)


def _resize_frame(frame: np.ndarray, max_short_side: Optional[int]) -> np.ndarray:
    """Resize so the short side ≤ max_short_side, preserving aspect ratio."""
    if max_short_side is None:
        return frame
    h, w = frame.shape[:2]
    short = min(h, w)
    if short <= max_short_side:
        return frame
    scale = max_short_side / short
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _decode_gray_frames(video_path: Path, max_short_side: Optional[int]) -> List[np.ndarray]:
    """Decode all frames from a video as grayscale numpy arrays."""
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            rgb = frame.to_ndarray(format="rgb24")
            rgb = _resize_frame(rgb, max_short_side)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            frames.append(gray)
    return frames


def _farneback_params() -> dict:
    """Standard Farneback parameters (good balance of speed vs quality)."""
    return dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def process_video(
    video_path: Path,
    out_dir: Path,
    max_short_side: Optional[int],
    skip_existing: bool,
) -> Tuple[str, int, float]:
    """Compute and save flow for a single video.

    Returns (status, n_flow_frames, elapsed_seconds).
    """
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.npy"

    if skip_existing and meta_path.exists():
        meta = np.load(meta_path)
        return "skipped", int(meta[0]), 0.0

    frames = _decode_gray_frames(video_path, max_short_side)
    if len(frames) < 2:
        return "too_short", 0, time.perf_counter() - t0

    fb_params = _farneback_params()
    h, w = frames[0].shape
    n_flow = len(frames) - 1

    flow_min, flow_max = np.inf, -np.inf
    for i in range(n_flow):
        flow = cv2.calcOpticalFlowFarneback(frames[i], frames[i + 1], None, **fb_params)
        encoded = _flow_to_uint8(flow)  # (2, H, W) uint8
        np.save(out_dir / f"frame_{i:04d}.npy", encoded)
        flow_min = min(flow_min, flow.min())
        flow_max = max(flow_max, flow.max())

    # meta: [n_flow_frames, H, W, observed_flow_min, observed_flow_max]
    np.save(meta_path, np.array([n_flow, h, w, flow_min, flow_max], dtype=np.float32))

    return "ok", n_flow, time.perf_counter() - t0


def _worker(args):
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    video_path, out_dir, max_short_side, skip_existing = args
    try:
        status, n, elapsed = process_video(video_path, out_dir, max_short_side, skip_existing)
        return str(video_path), status, n, elapsed
    except Exception as exc:
        return str(video_path), f"error: {exc}", 0, 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-compute Farneback optical flow.")
    parser.add_argument(
        "--flow_root",
        type=Path,
        default=Path("data/flow"),
        help="Root directory for output flow files (default: data/flow).",
    )
    parser.add_argument(
        "--max_short_side",
        type=int,
        default=320,
        help="Resize frames so the short side ≤ this value before computing flow. "
             "Set 0 to disable resizing (default: 320).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1). "
             "Increase to speed up on multi-core machines.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip videos whose output directory already contains meta.npy (default: True).",
    )
    parser.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        help="Recompute flow even if output already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the list of videos and output paths without computing anything.",
    )
    args = parser.parse_args()

    max_short_side = args.max_short_side if args.max_short_side > 0 else None

    # Resolve data roots from the base config
    cfg = BaseConfig()
    data_roots = list(resolve_data_roots(cfg.data_roots))

    # Collect all videos
    all_videos = _collect_videos(data_roots)
    if not all_videos:
        print("No videos found. Check BaseConfig.data_roots.")
        sys.exit(1)

    print(f"Found {len(all_videos)} videos across {len(data_roots)} data root(s).")
    print(f"Flow root : {args.flow_root.resolve()}")
    print(f"Short side: {max_short_side or 'native'}")
    print(f"Workers   : {args.workers}")
    print(f"Skip existing: {args.skip_existing}")

    # Build work list: (video_path, out_dir, max_short_side, skip_existing)
    repo_root = Path(__file__).resolve().parent.parent
    work = []
    for video_path, class_name in all_videos:
        # Mirror directory structure: flow/<data_root_stem>/<class>/<video_stem>/
        try:
            rel = video_path.relative_to(repo_root)
            # Use parts[1] onward (strip leading "data/")
            stem_parts = rel.parts[1:-1]  # e.g. ("verified_data", "verified_data", "data_btc_10s", "pushups")
        except ValueError:
            stem_parts = (video_path.parent.parent.name, class_name)

        out_dir = args.flow_root / Path(*stem_parts) / video_path.stem
        work.append((video_path, out_dir, max_short_side, args.skip_existing))

    if args.dry_run:
        print("\n-- Dry run (first 10 entries) --")
        for vp, od, *_ in work[:10]:
            print(f"  {vp}\n    → {od}")
        if len(work) > 10:
            print(f"  ... and {len(work) - 10} more")
        return

    # ── Process ───────────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    counts = {"ok": 0, "skipped": 0, "too_short": 0, "error": 0}

    if args.workers <= 1:
        for i, (vp, od, mss, se) in enumerate(work, 1):
            vid_str, status, n_flow, elapsed = _worker((vp, od, mss, se))
            key = "error" if status.startswith("error") else status
            counts[key] += 1
            tag = f"[{i:4d}/{len(work)}]"
            if status == "ok":
                print(f"{tag} {Path(vid_str).name}: {n_flow} flow frames ({elapsed:.1f}s)")
            elif status == "skipped":
                print(f"{tag} {Path(vid_str).name}: skipped (already exists)")
            else:
                print(f"{tag} {Path(vid_str).name}: {status}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, w): w[0] for w in work}
            done = 0
            for fut in as_completed(futures):
                done += 1
                vid_str, status, n_flow, elapsed = fut.result()
                key = "error" if status.startswith("error") else status
                counts[key] += 1
                tag = f"[{done:4d}/{len(work)}]"
                if status == "ok":
                    print(f"{tag} {Path(vid_str).name}: {n_flow} flow frames ({elapsed:.1f}s)")
                elif status == "skipped":
                    print(f"{tag} {Path(vid_str).name}: skipped")
                else:
                    print(f"{tag} {Path(vid_str).name}: {status}")

    total = time.perf_counter() - t_start
    print(f"\nDone in {total:.1f}s — ok={counts['ok']}  skipped={counts['skipped']}  "
          f"too_short={counts['too_short']}  errors={counts['error']}")


if __name__ == "__main__":
    main()
