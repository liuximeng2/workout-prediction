"""Run inference on a single video file.

Usage:
    python scripts/inference.py --video path/to/video.mp4
    python scripts/inference.py --video path/to/video.mp4 --checkpoint checkpoints/videomae-workout
    python scripts/inference.py --video path/to/video.mp4 --top_k 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from config.models import VideoMAEConfig
from model.video_mae.model import get_video_params
from utils.transforms import make_val_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VideoMAE workout inference")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Fine-tuned checkpoint directory (default: VideoMAEConfig.output_dir)",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to print")
    return parser.parse_args()


def load_video_clip(
    video_path: str,
    num_frames: int,
    clip_duration: float,
    val_transform,
) -> torch.Tensor:
    """Decode a single video and return a ``(1, T, C, H, W)`` tensor."""
    import pytorchvideo.data

    dataset = pytorchvideo.data.LabeledVideoDataset(
        labeled_video_paths=[(video_path, {"label": 0})],
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )
    sample = next(iter(dataset))
    # (C, T, H, W) → (T, C, H, W) → (1, T, C, H, W)
    return sample["video"].permute(1, 0, 2, 3).unsqueeze(0)


def main():
    args = parse_args()
    cfg = VideoMAEConfig()
    ckpt = args.checkpoint or cfg.output_dir
    video_path = str(Path(args.video).resolve())

    # ── Load model ────────────────────────────────────────────────────────────
    image_processor = VideoMAEImageProcessor.from_pretrained(ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(ckpt)
    model.eval()

    mean, std, resize_to = get_video_params(image_processor)
    num_frames = model.config.num_frames
    clip_duration = num_frames * cfg.sample_rate / cfg.fps

    # ── Pre-process video ─────────────────────────────────────────────────────
    val_transform = make_val_transform(num_frames, resize_to, mean, std)
    pixel_values = load_video_clip(video_path, num_frames, clip_duration, val_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pixel_values = pixel_values.to(device)
    model = model.to(device)

    # ── Forward pass ──────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits  # (1, num_classes)

    probs = F.softmax(logits, dim=-1).squeeze(0)
    top_k = min(args.top_k, len(probs))
    top_probs, top_indices = probs.topk(top_k)

    print(f"\nPredictions for: {args.video}")
    print(f"{'Rank':<6}{'Label':<35}{'Score':>8}")
    print("-" * 51)
    for rank, (prob, idx) in enumerate(
        zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()), start=1
    ):
        label = model.config.id2label[idx]
        print(f"{rank:<6}{label:<35}{prob:>8.4f}")


if __name__ == "__main__":
    main()
