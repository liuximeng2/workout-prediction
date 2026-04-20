"""Zero-shot exercise classification with Qwen3-VL via Ollama.

Frames are sampled uniformly from each test video and sent to the model as
base64-encoded JPEGs in a single chat message.  No fine-tuning is performed.

Usage:
    python scripts/eval_qwen3_vl.py
    python scripts/eval_qwen3_vl.py --num_frames 16 --num_clips 3
    python scripts/eval_qwen3_vl.py --ollama_model qwen3-vl:latest --num_frames 8
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import av
from PIL import Image

from config.models.qwen3_vl_config import Qwen3VLConfig

# Import directly from the submodule file to avoid loading the full model registry
# (which eagerly imports VideoMAE/ViViT and requires transformers/torch).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "qwen3_vl_model",
    Path(__file__).resolve().parent.parent / "model" / "qwen3_vl" / "model.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Qwen3VLClassifier = _mod.Qwen3VLClassifier
from utils.dataset import (
    _collect_labeled_paths,
    _split_paths,
    build_label_maps,
    resolve_data_roots,
    resolve_roots_for_label_maps,
)


# ── Frame extraction ──────────────────────────────────────────────────────────

def _extract_frames(
    video_path: str,
    num_frames: int,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
) -> List[Image.Image]:
    """Decode ``num_frames`` frames uniformly from [start_frac, end_frac] of the video.

    Args:
        video_path:  Absolute path to a video file.
        num_frames:  How many frames to return.
        start_frac:  Start position as fraction of total duration [0, 1].
        end_frac:    End position as fraction of total duration [0, 1].

    Returns:
        List of PIL RGB images in temporal order.  May be shorter than
        ``num_frames`` for very short videos.
    """
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        if container.duration:
            duration = float(container.duration) / av.time_base
        elif stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = 10.0

        t_start = duration * start_frac
        t_end = duration * end_frac

        # Collect all frame timestamps in [t_start, t_end]
        seek_pts = int(t_start / float(stream.time_base))
        container.seek(seek_pts, stream=stream)

        all_frames: List[Tuple[float, Image.Image]] = []
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < t_start:
                continue
            if t > t_end:
                break
            img = Image.fromarray(frame.to_ndarray(format="rgb24"))
            all_frames.append((t, img))

    if not all_frames:
        return []

    if len(all_frames) <= num_frames:
        return [img for _, img in all_frames]

    # Uniform subsample: pick indices spread over the collected frames
    n = len(all_frames)
    indices = [int(i * (n - 1) / (num_frames - 1)) for i in range(num_frames)]
    return [all_frames[i][1] for i in indices]


# ── Clip positions ────────────────────────────────────────────────────────────

def _clip_windows(num_clips: int) -> List[Tuple[float, float]]:
    """Return ``num_clips`` non-overlapping (start_frac, end_frac) windows."""
    if num_clips == 1:
        return [(0.0, 1.0)]
    step = 1.0 / num_clips
    return [(i * step, (i + 1) * step) for i in range(num_clips)]


# ── Per-class printer (mirrors eval.py) ──────────────────────────────────────

def _print_per_class(preds: list, labels: list, id2label: Dict[int, str]) -> None:
    correct: Dict = defaultdict(int)
    total: Dict = defaultdict(int)
    for pred, label in zip(preds, labels):
        cls = id2label[label]
        total[cls] += 1
        if pred == label:
            correct[cls] += 1

    print("\nPer-class accuracy:")
    for cls in sorted(total):
        acc = correct[cls] / total[cls]
        print(f"  {cls:<38s}  {acc:.2%}  ({correct[cls]}/{total[cls]})")


# ── Main evaluation ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL zero-shot on the exercise test split."
    )
    parser.add_argument(
        "--num_frames", type=int, default=None,
        help="Frames per clip sent to Qwen3-VL (default: from config).",
    )
    parser.add_argument(
        "--num_clips", type=int, default=None,
        help="Clips sampled per video; majority vote decides label (default: from config).",
    )
    parser.add_argument(
        "--frame_size", type=int, default=None,
        help="Short-side resize before encoding (default: from config).",
    )
    parser.add_argument(
        "--ollama_model", type=str, default=None,
        help="Ollama model tag (default: from config).",
    )
    parser.add_argument(
        "--ollama_url", type=str, default=None,
        help="Ollama base URL (default: from config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Qwen3VLConfig()

    num_frames  = args.num_frames  or cfg.num_frames
    num_clips   = args.num_clips   or cfg.num_clips
    frame_size  = args.frame_size  or cfg.frame_size
    ollama_model = args.ollama_model or cfg.ollama_model
    ollama_url   = args.ollama_url   or cfg.ollama_base_url

    repo_root = Path(__file__).resolve().parent.parent

    # ── Label maps ────────────────────────────────────────────────────────────
    label2id, id2label = build_label_maps(
        resolve_roots_for_label_maps(
            cfg.data_roots, cfg.test_data_roots, repo_root=repo_root
        )
    )

    # ── Test paths ────────────────────────────────────────────────────────────
    test_paths = None
    if cfg.test_data_roots:
        test_roots = resolve_data_roots(cfg.test_data_roots, repo_root=repo_root)
        if test_roots and test_roots[0].is_dir():
            explicit = _collect_labeled_paths(test_roots, label2id)
            if explicit:
                test_paths = explicit

    if test_paths is None:
        data_roots = resolve_data_roots(cfg.data_roots, repo_root=repo_root)
        all_paths = _collect_labeled_paths(data_roots, label2id)
        _, _, test_paths = _split_paths(all_paths, cfg.train_split, cfg.val_split, cfg.seed)

    n_videos = len(test_paths)

    # ── Build classifier ──────────────────────────────────────────────────────
    clf = Qwen3VLClassifier(
        ollama_base_url=ollama_url,
        ollama_model=ollama_model,
        label2id=label2id,
        id2label=id2label,
        request_timeout=cfg.request_timeout,
    )

    print(f"Model       : {ollama_model}")
    print(f"Test videos : {n_videos}")
    print(f"Frames/clip : {num_frames}")
    print(f"Clips/video : {num_clips}  (majority vote)")
    print(f"Frame size  : {frame_size}px short-side\n")

    windows = _clip_windows(num_clips)
    all_preds: List[int] = []
    all_labels: List[int] = []

    for vid_idx, (video_path, info) in enumerate(test_paths):
        true_label = info["label"]

        clip_preds: List[int] = []
        for start_frac, end_frac in windows:
            frames = _extract_frames(video_path, num_frames, start_frac, end_frac)
            if not frames:
                clip_preds.append(0)
                continue
            pred = clf.predict(frames, frame_size=frame_size)
            clip_preds.append(pred)

        # Majority vote across clips
        final_pred = Counter(clip_preds).most_common(1)[0][0]
        all_preds.append(final_pred)
        all_labels.append(true_label)

        correct = final_pred == true_label
        running_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
        print(
            f"[{vid_idx + 1:4d}/{n_videos}]  "
            f"{'OK' if correct else '--'}  "
            f"pred={id2label[final_pred]:<32s}  "
            f"true={id2label[true_label]:<32s}  "
            f"running_acc={running_acc:.3f}"
        )

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\nTest accuracy: {accuracy:.4f}")
    _print_per_class(all_preds, all_labels, id2label)


if __name__ == "__main__":
    main()
