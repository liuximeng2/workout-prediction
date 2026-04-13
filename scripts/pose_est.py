"""Pre-compute YOLO pose keypoints for all videos and save to disk.

For each video we run YOLO26l-pose on every Nth frame, extract the dominant
person's 17 COCO keypoints (x, y, confidence), and store the result as a
single .npy file per video.

Storage layout mirrors the video directory structure:

    data/pose/<data_root_stem>/<class>/<video_stem>.npy
        # shape (T, 17, 3) — float32, where T = number of sampled frames
        # channels: (x_normalized, y_normalized, confidence)

Usage:
    python scripts/pose_est.py
    python scripts/pose_est.py --input_dirs data/reencoded/data_btc_10s data/reencoded/data_crawl_10s
    python scripts/pose_est.py --sample_every 2        # every 2nd frame
    python scripts/pose_est.py --yolo_model yolo26l-pose.pt
    python scripts/pose_est.py --dry_run
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config.base_config import BaseConfig
from utils.dataset import resolve_data_roots

# ── Constants ─────────────────────────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


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


def _out_path_for_video(
    video_path: Path,
    pose_root: Path,
    repo_root: Path,
) -> Path:
    """Build the output .npy path for a video, mirroring the data dir structure."""
    try:
        rel = video_path.relative_to(repo_root)
        # Strip leading "data/" segment, keep the rest minus the video filename
        stem_parts = rel.parts[1:-1]
    except ValueError:
        stem_parts = (video_path.parent.parent.name, video_path.parent.name)
    return pose_root / Path(*stem_parts) / f"{video_path.stem}.npy"


def process_video(
    video_path: Path,
    out_path: Path,
    yolo_model,
    sample_every: int,
    img_size: int,
    skip_existing: bool,
) -> Tuple[str, int, float]:
    """Run pose estimation on a single video and save keypoints.

    Returns (status, n_frames_processed, elapsed_seconds).
    """
    t0 = time.perf_counter()

    if skip_existing and out_path.exists():
        return "skipped", 0, 0.0

    import av

    # Decode frames
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i % sample_every != 0:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))

    if len(frames) < 2:
        return "too_short", len(frames), time.perf_counter() - t0

    # Run YOLO pose on all sampled frames
    results = yolo_model(frames, imgsz=img_size, verbose=False)

    all_keypoints = []
    for result in results:
        if result.keypoints is None or len(result.keypoints) == 0:
            # No person detected — fill with zeros
            all_keypoints.append(np.zeros((17, 3), dtype=np.float32))
            continue

        kpts = result.keypoints.data.cpu().numpy()  # (n_persons, 17, 3)

        # Pick the dominant person: highest mean confidence
        if kpts.shape[0] == 1:
            best = kpts[0]
        else:
            mean_conf = kpts[:, :, 2].mean(axis=1)
            best = kpts[mean_conf.argmax()]

        all_keypoints.append(best.astype(np.float32))

    keypoints = np.stack(all_keypoints)  # (T, 17, 3)

    # Normalize x, y to [0, 1] using the image dimensions from the first result
    orig_shape = results[0].orig_img.shape[:2]  # (H, W)
    h, w = orig_shape
    keypoints[:, :, 0] /= w  # normalize x
    keypoints[:, :, 1] /= h  # normalize y

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, keypoints)

    return "ok", len(frames), time.perf_counter() - t0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-compute YOLO pose keypoints.")
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        default=None,
        metavar="DIR",
        help="One or more video root directories (each with <class>/<video> layout). "
             "Overrides BaseConfig.data_roots when provided.",
    )
    parser.add_argument(
        "--pose_root",
        type=Path,
        default=Path("data/pose"),
        help="Root directory for output pose .npy files (default: data/pose).",
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="yolo26n-pose.pt",
        help="YOLO pose model name or path (default: yolo26l-pose.pt).",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=2,
        help="Process every Nth frame (default: 2). Set to 1 to process all frames.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="YOLO input image size (default: 640).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip videos whose output .npy already exists (default: True).",
    )
    parser.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        help="Recompute pose even if output already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the list of videos and output paths without computing anything.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    # Resolve data roots
    if args.input_dirs:
        data_roots = [Path(d).expanduser().resolve() for d in args.input_dirs]
    else:
        cfg = BaseConfig()
        data_roots = list(resolve_data_roots(cfg.data_roots, repo_root=repo_root))

    all_videos = _collect_videos(data_roots)
    if not all_videos:
        print("No videos found. Check BaseConfig.data_roots or --input_dirs.")
        sys.exit(1)

    print(f"Found {len(all_videos)} videos across {len(data_roots)} data root(s).")
    print(f"Pose root    : {args.pose_root.resolve()}")
    print(f"YOLO model   : {args.yolo_model}")
    print(f"Sample every : {args.sample_every}")
    print(f"Image size   : {args.img_size}")
    print(f"Skip existing: {args.skip_existing}")

    # Build work list
    work = []
    for video_path, _class_name in all_videos:
        out_path = _out_path_for_video(video_path, args.pose_root, repo_root)
        work.append((video_path, out_path))

    if args.dry_run:
        print("\n-- Dry run (first 10 entries) --")
        for vp, op in work[:10]:
            print(f"  {vp}\n    → {op}")
        if len(work) > 10:
            print(f"  ... and {len(work) - 10} more")
        return

    # Load YOLO model once
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_model)

    counts = {"ok": 0, "skipped": 0, "too_short": 0, "error": 0}
    t_start = time.perf_counter()

    for i, (vp, op) in enumerate(work, 1):
        try:
            status, n_frames, elapsed = process_video(
                vp, op, yolo, args.sample_every, args.img_size, args.skip_existing,
            )
        except Exception as exc:
            status, n_frames, elapsed = f"error: {exc}", 0, 0.0

        key = "error" if str(status).startswith("error") else status
        counts[key] += 1

        tag = f"[{i:4d}/{len(work)}]"
        if status == "ok":
            print(f"{tag} {vp.name}: {n_frames} frames ({elapsed:.1f}s)")
        elif status == "skipped":
            print(f"{tag} {vp.name}: skipped (already exists)")
        else:
            print(f"{tag} {vp.name}: {status}")

    total = time.perf_counter() - t_start
    print(f"\nDone in {total:.1f}s — ok={counts['ok']}  skipped={counts['skipped']}  "
          f"too_short={counts['too_short']}  errors={counts['error']}")


if __name__ == "__main__":
    main()
