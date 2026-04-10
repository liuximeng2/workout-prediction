"""Evaluate a fine-tuned checkpoint on the held-out test split.

Supports both VideoMAE (HuggingFace) and Two-Stream CNN checkpoints.

Usage:
    # VideoMAE (HuggingFace checkpoint directory)
    python scripts/eval.py --model video_mae --checkpoint checkpoints/videomae-workout

    # Two-Stream CNN (.pt file)
    python scripts/eval.py --model two_stream --checkpoint checkpoints/two-stream-workout/best.pt

Note: This file must not be named ``evaluate.py`` — that name shadows Hugging Face's
``evaluate`` package on ``sys.path`` when running ``python scripts/<name>.py``.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset import (
    _collect_labeled_paths,
    _split_paths,
    build_label_maps,
    filter_valid_flow,
    resolve_data_roots,
    resolve_roots_for_label_maps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a workout video classifier on the test split.")
    parser.add_argument(
        "--model",
        type=str,
        default="video_mae",
        choices=["video_mae", "two_stream"],
        help="Model architecture (default: video_mae).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. HF directory for video_mae; .pt file for two_stream. "
             "Defaults to the model config's output_dir / best.pt.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def _video_mae_collate(examples):
    pixel_values = torch.stack(
        [ex["video"].permute(1, 0, 2, 3) for ex in examples]
    )
    labels = torch.tensor([ex["label"] for ex in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# ── VideoMAE evaluation ───────────────────────────────────────────────────────

def eval_video_mae(ckpt: str, batch_size: int) -> None:
    import evaluate as hf_evaluate
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    from config.models import VideoMAEConfig
    from model.video_mae.model import get_video_params
    from utils import (
        build_datasets,
        make_val_transform,
    )

    cfg = VideoMAEConfig()
    repo_root = Path(__file__).resolve().parent.parent

    image_processor = VideoMAEImageProcessor.from_pretrained(ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(ckpt)
    model.eval()

    mean, std, resize_to = get_video_params(image_processor)
    num_frames = model.config.num_frames
    clip_duration = num_frames * cfg.sample_rate / cfg.fps

    data_roots = resolve_data_roots(cfg.data_roots, repo_root=repo_root)
    test_roots = (
        resolve_data_roots(cfg.test_data_roots, repo_root=repo_root)
        if cfg.test_data_roots else None
    )
    label2id, id2label = build_label_maps(
        resolve_roots_for_label_maps(cfg.data_roots, cfg.test_data_roots, repo_root=repo_root)
    )
    val_transform = make_val_transform(num_frames, resize_to, mean, std)
    _, _, test_dataset = build_datasets(
        data_roots=data_roots,
        label2id=label2id,
        clip_duration=clip_duration,
        train_transform=val_transform,
        val_transform=val_transform,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        seed=cfg.seed,
        test_data_roots=test_roots,
    )
    print(f"Test videos: {test_dataset.num_videos}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader = DataLoader(test_dataset, batch_size=batch_size,
                        collate_fn=_video_mae_collate, num_workers=2)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(pixel_values=batch["pixel_values"].to(device))
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].tolist())

    results = hf_evaluate.load("accuracy").compute(
        predictions=all_preds, references=all_labels
    )
    print(f"\nTest accuracy: {results['accuracy']:.4f}")
    _print_per_class(all_preds, all_labels, id2label)


def _two_stream_collate(batch):
    videos = torch.stack([b["video"] for b in batch])
    flows  = torch.stack([b["flow"]  for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"video": videos, "flow": flows, "label": labels}


# ── Two-Stream evaluation ─────────────────────────────────────────────────────

def eval_two_stream(ckpt_path: str, batch_size: int) -> None:
    from config.models.two_stream_config import TwoStreamConfig
    from model.two_stream.model import build_model
    from utils.flow_dataset import TwoStreamDataset

    cfg = TwoStreamConfig()
    repo_root = Path(__file__).resolve().parent.parent

    # ── Load checkpoint ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    label2id: dict = ckpt["label2id"]
    id2label: dict = ckpt["id2label"]
    num_classes = len(label2id)

    model, _ = build_model(
        num_classes=num_classes,
        num_flow_frames=cfg.num_flow_frames,
        learnable_fusion=cfg.learnable_fusion,
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    print(f"Loaded epoch {ckpt.get('epoch', '?')} "
          f"(best val acc in ckpt: {ckpt.get('best_val_acc', float('nan')):.3f})")

    # ── Build test dataset ────────────────────────────────────────────────────
    data_roots = list(resolve_data_roots(cfg.data_roots, repo_root=repo_root))
    flow_root = repo_root / cfg.flow_root

    # Use explicit test root if present, else fall back to stratified split
    test_paths = None
    if cfg.test_data_roots:
        test_roots = list(resolve_data_roots(cfg.test_data_roots, repo_root=repo_root))
        if test_roots and test_roots[0].is_dir():
            all_test = _collect_labeled_paths(test_roots, label2id)
            all_test = filter_valid_flow(all_test, flow_root, repo_root=repo_root)
            if all_test:
                test_paths = all_test

    if test_paths is None:
        all_paths = _collect_labeled_paths(data_roots, label2id)
        all_paths = filter_valid_flow(all_paths, flow_root, repo_root=repo_root)
        _, _, test_paths = _split_paths(all_paths, cfg.train_split, cfg.val_split, cfg.seed)

    test_ds = TwoStreamDataset(
        labeled_paths=test_paths,
        flow_root=flow_root,
        clip_duration=cfg.clip_duration,
        num_flow_frames=cfg.num_flow_frames,
        mode="val",
        repo_root=repo_root,
    )
    print(f"Test samples: {len(test_ds)}")

    pin_memory = device.type == "cuda"
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=pin_memory,
                        collate_fn=_two_stream_collate)

    # ── Inference loop ────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            video  = batch["video"].to(device)
            flow   = batch["flow"].to(device)
            labels = batch["label"].to(device)

            logits = model(video, flow)
            loss   = criterion(logits, labels)

            preds = logits.argmax(dim=-1)
            bs = labels.size(0)
            total_loss += loss.item() * bs
            correct    += (preds == labels).sum().item()
            total      += bs

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print(f"\nTest loss    : {total_loss / total:.4f}")
    print(f"Test accuracy: {correct / total:.4f}")
    print(f"Flow weight  : {model.flow_weight:.3f}")
    _print_per_class(all_preds, all_labels, id2label)


# ── Shared per-class printer ──────────────────────────────────────────────────

def _print_per_class(
    preds: list,
    labels: list,
    id2label: dict,
) -> None:
    class_correct: dict = defaultdict(int)
    class_total:   dict = defaultdict(int)
    for pred, label in zip(preds, labels):
        cls = id2label[label]
        class_total[cls] += 1
        if pred == label:
            class_correct[cls] += 1

    print("\nPer-class accuracy:")
    for cls in sorted(class_total):
        acc = class_correct[cls] / class_total[cls]
        print(f"  {cls:<32s}  {acc:.2%}  ({class_correct[cls]}/{class_total[cls]})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    if args.model == "video_mae":
        from config.models import VideoMAEConfig
        ckpt = args.checkpoint or str(repo_root / VideoMAEConfig().output_dir)
        print(f"Model      : VideoMAE")
        print(f"Checkpoint : {ckpt}\n")
        eval_video_mae(ckpt, args.batch_size)

    elif args.model == "two_stream":
        from config.models.two_stream_config import TwoStreamConfig
        default_ckpt = repo_root / TwoStreamConfig().output_dir / "best.pt"
        ckpt = args.checkpoint or str(default_ckpt)
        print(f"Model      : Two-Stream CNN")
        print(f"Checkpoint : {ckpt}\n")
        eval_two_stream(ckpt, args.batch_size)


if __name__ == "__main__":
    main()
