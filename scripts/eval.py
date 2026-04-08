"""Evaluate a fine-tuned checkpoint on the held-out test split.

Usage:
    python scripts/run_evaluation.py --checkpoint checkpoints/videomae-workout
    python scripts/run_evaluation.py --checkpoint checkpoints/videomae-workout --batch_size 8

Note: This file must not be named ``evaluate.py`` — that name shadows Hugging Face's
``evaluate`` package on ``sys.path`` when running ``python scripts/<name>.py``.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import evaluate as hf_evaluate
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from config.models import VideoMAEConfig
from model.video_mae.model import get_video_params
from utils import build_datasets, build_label_maps, make_val_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a workout video classifier on the test split")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned checkpoint directory (default: VideoMAEConfig.output_dir)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def collate_fn(examples):
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def main():
    args = parse_args()
    cfg = VideoMAEConfig()
    ckpt = args.checkpoint or cfg.output_dir

    # ── Load fine-tuned model ─────────────────────────────────────────────────
    image_processor = VideoMAEImageProcessor.from_pretrained(ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(ckpt)
    model.eval()

    mean, std, resize_to = get_video_params(image_processor)
    num_frames = model.config.num_frames
    clip_duration = num_frames * cfg.sample_rate / cfg.fps

    # ── Build test dataset ────────────────────────────────────────────────────
    label2id, id2label = build_label_maps(cfg.data_roots)
    val_transform = make_val_transform(num_frames, resize_to, mean, std)
    _, _, test_dataset = build_datasets(
        data_roots=cfg.data_roots,
        label2id=label2id,
        clip_duration=clip_duration,
        train_transform=val_transform,
        val_transform=val_transform,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )
    print(f"Test videos: {test_dataset.num_videos}")

    # ── Inference loop ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=2,
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(pixel_values=batch["pixel_values"].to(device))
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].tolist())

    # ── Overall accuracy ──────────────────────────────────────────────────────
    results = hf_evaluate.load("accuracy").compute(
        predictions=all_preds, references=all_labels
    )
    print(f"\nTest accuracy: {results['accuracy']:.4f}")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    class_correct: dict = defaultdict(int)
    class_total: dict = defaultdict(int)
    for pred, label in zip(all_preds, all_labels):
        cls = id2label[label]
        class_total[cls] += 1
        if pred == label:
            class_correct[cls] += 1

    print("\nPer-class accuracy:")
    for cls in sorted(class_total):
        acc = class_correct[cls] / class_total[cls]
        print(f"  {cls:<32s}  {acc:.2%}  ({class_correct[cls]}/{class_total[cls]})")


if __name__ == "__main__":
    main()
