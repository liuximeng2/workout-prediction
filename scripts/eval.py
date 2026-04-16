"""Evaluate a fine-tuned checkpoint on the held-out test split.

Supports both VideoMAE (HuggingFace) and Two-Stream CNN checkpoints.

Usage:
    # VideoMAE (HuggingFace checkpoint directory)
    python scripts/eval.py --model video_mae --checkpoint checkpoints/videomae

    python scripts/eval.py --model vivit --checkpoint checkpoints/vivit

    # Two-Stream CNN (.pt file)
    python scripts/eval.py --model two_stream --checkpoint checkpoints/two-stream-workout/best.pt

    # Multi-clip evaluation (average logits over 5 evenly-spaced clips per video)
    python scripts/eval.py --model two_stream --checkpoint best.pt --num_clips 5

    # Pose MLP (.pt file)
    python scripts/eval.py --model pose --checkpoint checkpoints/pose/best.pt

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
        choices=["video_mae", "two_stream", "vivit", "pose", "cnn_fusion"],
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
    parser.add_argument(
        "--num_clips",
        type=int,
        default=5,
        help="Number of evenly-spaced clips to sample per video. "
             "Logits are averaged across clips before prediction (default: 1 = center clip only).",
    )
    return parser.parse_args()


def _clip_positions(num_clips: int) -> list:
    """Return *num_clips* evenly-spaced positions in [0, 1]."""
    if num_clips == 1:
        return [0.5]                       # centre clip (matches old behaviour)
    return [i / (num_clips - 1) for i in range(num_clips)]


def _aggregate_clip_logits(
    per_video_logits: list,
    video_labels: list,
) -> tuple:
    """Average logits across clips per video and return predictions + labels."""
    all_preds, all_labels = [], []
    for logits_list, label in zip(per_video_logits, video_labels):
        avg = torch.stack(logits_list).mean(dim=0)
        all_preds.append(avg.argmax().item())
        all_labels.append(label)
    return all_preds, all_labels


def _video_mae_collate(examples):
    pixel_values = torch.stack(
        [ex["video"].permute(1, 0, 2, 3) for ex in examples]
    )
    labels = torch.tensor([ex["label"] for ex in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def _pose_collate(batch):
    features = torch.stack([b["features"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"features": features, "label": labels}


# ── VideoMAE evaluation ───────────────────────────────────────────────────────

def eval_video_mae(ckpt: str, batch_size: int, num_clips: int = 1) -> None:
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

    positions = _clip_positions(num_clips)
    n_videos = len(test_dataset)
    per_video_logits = [[] for _ in range(n_videos)]
    video_labels = [0] * n_videos

    print(f"Clips/video: {num_clips}  positions: "
          f"{[f'{p:.2f}' for p in positions]}")

    with torch.no_grad():
        for clip_idx, pos in enumerate(positions):
            test_dataset.clip_position = pos
            loader = DataLoader(test_dataset, batch_size=batch_size,
                                collate_fn=_video_mae_collate, num_workers=2)
            vid_offset = 0
            for batch in loader:
                outputs = model(pixel_values=batch["pixel_values"].to(device))
                logits = outputs.logits.cpu()
                labels = batch["labels"]
                for i in range(logits.size(0)):
                    per_video_logits[vid_offset + i].append(logits[i])
                    video_labels[vid_offset + i] = labels[i].item()
                vid_offset += logits.size(0)
            print(f"  clip {clip_idx + 1}/{num_clips} (pos={pos:.2f}) done")

    all_preds, all_labels = _aggregate_clip_logits(per_video_logits, video_labels)

    results = hf_evaluate.load("accuracy").compute(
        predictions=all_preds, references=all_labels
    )
    print(f"\nTest accuracy: {results['accuracy']:.4f}")
    _print_per_class(all_preds, all_labels, id2label)


def eval_vivit(ckpt: str, batch_size: int, num_clips: int = 1) -> None:
    import evaluate as hf_evaluate
    from transformers import VivitForVideoClassification, VivitImageProcessor

    from config.models import ViViTConfig
    from model.vivit.model import get_video_params
    from utils import (
        build_datasets,
        make_val_transform,
    )

    cfg = ViViTConfig()
    repo_root = Path(__file__).resolve().parent.parent

    image_processor = VivitImageProcessor.from_pretrained(ckpt)
    model = VivitForVideoClassification.from_pretrained(ckpt)
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

    positions = _clip_positions(num_clips)
    n_videos = len(test_dataset)
    per_video_logits = [[] for _ in range(n_videos)]
    video_labels = [0] * n_videos

    print(f"Clips/video: {num_clips}  positions: "
          f"{[f'{p:.2f}' for p in positions]}")

    with torch.no_grad():
        for clip_idx, pos in enumerate(positions):
            test_dataset.clip_position = pos
            loader = DataLoader(test_dataset, batch_size=batch_size,
                                collate_fn=_video_mae_collate, num_workers=2)
            vid_offset = 0
            for batch in loader:
                outputs = model(pixel_values=batch["pixel_values"].to(device))
                logits = outputs.logits.cpu()
                labels = batch["labels"]
                for i in range(logits.size(0)):
                    per_video_logits[vid_offset + i].append(logits[i])
                    video_labels[vid_offset + i] = labels[i].item()
                vid_offset += logits.size(0)
            print(f"  clip {clip_idx + 1}/{num_clips} (pos={pos:.2f}) done")

    all_preds, all_labels = _aggregate_clip_logits(per_video_logits, video_labels)

    results = hf_evaluate.load("accuracy").compute(
        predictions=all_preds, references=all_labels
    )
    print(f"\nTest accuracy: {results['accuracy']:.4f}")
    _print_per_class(all_preds, all_labels, id2label)


# ── Pose MLP evaluation ──────────────────────────────────────────────────────

def eval_pose(ckpt_path: str, batch_size: int) -> None:
    from config.models.pose_config import PoseConfig
    from model.pose.model import build_model
    from utils.pose_dataset import PoseFeatureDataset, filter_valid_pose

    cfg = PoseConfig()
    repo_root = Path(__file__).resolve().parent.parent

    # ── Load checkpoint ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    label2id: dict = ckpt["label2id"]
    id2label: dict = ckpt["id2label"]
    num_classes = len(label2id)

    model, _ = build_model(
        num_classes=num_classes,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    print(f"Loaded epoch {ckpt.get('epoch', '?')} "
          f"(best val acc in ckpt: {ckpt.get('best_val_acc', float('nan')):.3f})")

    # ── Build test dataset ────────────────────────────────────────────────────
    data_roots = list(resolve_data_roots(cfg.data_roots, repo_root=repo_root))
    pose_root = repo_root / cfg.pose_root

    test_paths = None
    if cfg.test_data_roots:
        test_roots = list(resolve_data_roots(cfg.test_data_roots, repo_root=repo_root))
        if test_roots and test_roots[0].is_dir():
            all_test = _collect_labeled_paths(test_roots, label2id)
            all_test = filter_valid_pose(all_test, pose_root, repo_root=repo_root)
            if all_test:
                test_paths = all_test

    if test_paths is None:
        all_paths = _collect_labeled_paths(data_roots, label2id)
        all_paths = filter_valid_pose(all_paths, pose_root, repo_root=repo_root)
        _, _, test_paths = _split_paths(all_paths, cfg.train_split, cfg.val_split, cfg.seed)

    test_ds = PoseFeatureDataset(
        labeled_paths=test_paths,
        pose_root=pose_root,
        repo_root=repo_root,
    )
    print(f"Test samples: {len(test_ds)}")

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=_pose_collate)

    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    print(f"\nTest loss    : {avg_loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    _print_per_class(all_preds, all_labels, id2label)


def _cnn_fusion_collate(batch):
    videos = torch.stack([b["video"] for b in batch])  # (B, C, T, H, W)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"video": videos, "label": labels}


# ── CNN Fusion evaluation ─────────────────────────────────────────────────────

def eval_cnn_fusion(ckpt_path: str, batch_size: int, num_clips: int = 1) -> None:
    from config.models.cnn_fusion_config import CNNFusionConfig
    from model.cnn_fusion.model import build_model
    from utils.dataset import VideoClipDataset
    from utils.transforms import make_val_transform

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD  = (0.229, 0.224, 0.225)
    _RESIZE_TO     = (224, 224)

    cfg = CNNFusionConfig()
    repo_root = Path(__file__).resolve().parent.parent

    # ── Load checkpoint ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    label2id: dict = ckpt["label2id"]
    id2label: dict = ckpt["id2label"]
    num_classes = len(label2id)

    model, _ = build_model(
        num_classes=num_classes,
        backbone=cfg.backbone,
        num_frames=cfg.num_frames,
        aggregation=cfg.aggregation,
        dropout=cfg.dropout,
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    print(f"Loaded epoch {ckpt.get('epoch', '?')} "
          f"(best val acc in ckpt: {ckpt.get('best_val_acc', float('nan')):.3f})")

    # ── Build test dataset ────────────────────────────────────────────────────
    data_roots = list(resolve_data_roots(cfg.data_roots, repo_root=repo_root))

    test_paths = None
    if cfg.test_data_roots:
        test_roots = list(resolve_data_roots(cfg.test_data_roots, repo_root=repo_root))
        if test_roots and test_roots[0].is_dir():
            all_test = _collect_labeled_paths(test_roots, label2id)
            if all_test:
                test_paths = all_test

    if test_paths is None:
        all_paths = _collect_labeled_paths(data_roots, label2id)
        _, _, test_paths = _split_paths(all_paths, cfg.train_split, cfg.val_split, cfg.seed)

    val_transform = make_val_transform(cfg.num_frames, _RESIZE_TO, _IMAGENET_MEAN, _IMAGENET_STD)
    test_ds = VideoClipDataset(
        labeled_paths=test_paths,
        clip_duration=cfg.clip_duration,
        transform=val_transform,
        mode="uniform",
    )
    print(f"Test samples: {len(test_ds)}")

    positions = _clip_positions(num_clips)
    n_videos = len(test_ds)
    per_video_logits = [[] for _ in range(n_videos)]
    video_labels = [0] * n_videos

    print(f"Clips/video: {num_clips}  positions: "
          f"{[f'{p:.2f}' for p in positions]}")

    pin_memory = device.type == "cuda"

    with torch.no_grad():
        for clip_idx, pos in enumerate(positions):
            test_ds.clip_position = pos
            loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=pin_memory,
                                collate_fn=_cnn_fusion_collate)
            vid_offset = 0
            for batch in loader:
                video  = batch["video"].to(device)
                labels = batch["label"]
                logits = model(video).cpu()
                for i in range(logits.size(0)):
                    per_video_logits[vid_offset + i].append(logits[i])
                    video_labels[vid_offset + i] = labels[i].item()
                vid_offset += logits.size(0)
            print(f"  clip {clip_idx + 1}/{num_clips} (pos={pos:.2f}) done")

    all_preds, all_labels = _aggregate_clip_logits(per_video_logits, video_labels)

    avg_logits = torch.stack([torch.stack(v).mean(dim=0) for v in per_video_logits])
    label_tensor = torch.tensor(all_labels, dtype=torch.long)
    avg_loss = nn.CrossEntropyLoss()(avg_logits, label_tensor).item()
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    print(f"\nTest loss    : {avg_loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    _print_per_class(all_preds, all_labels, id2label)


def _two_stream_collate(batch):
    videos = torch.stack([b["video"] for b in batch])
    flows  = torch.stack([b["flow"]  for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"video": videos, "flow": flows, "label": labels}


# ── Two-Stream evaluation ─────────────────────────────────────────────────────

def eval_two_stream(ckpt_path: str, batch_size: int, num_clips: int = 1) -> None:
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

    positions = _clip_positions(num_clips)
    n_videos = len(test_ds)
    per_video_logits = [[] for _ in range(n_videos)]
    video_labels = [0] * n_videos

    print(f"Clips/video: {num_clips}  positions: "
          f"{[f'{p:.2f}' for p in positions]}")

    pin_memory = device.type == "cuda"

    with torch.no_grad():
        for clip_idx, pos in enumerate(positions):
            test_ds.clip_position = pos
            loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=pin_memory,
                                collate_fn=_two_stream_collate)
            vid_offset = 0
            for batch in loader:
                video  = batch["video"].to(device)
                flow   = batch["flow"].to(device)
                labels = batch["label"]

                logits = model(video, flow).cpu()
                for i in range(logits.size(0)):
                    per_video_logits[vid_offset + i].append(logits[i])
                    video_labels[vid_offset + i] = labels[i].item()
                vid_offset += logits.size(0)
            print(f"  clip {clip_idx + 1}/{num_clips} (pos={pos:.2f}) done")

    all_preds, all_labels = _aggregate_clip_logits(per_video_logits, video_labels)

    # Compute loss on the averaged logits
    avg_logits = torch.stack([torch.stack(v).mean(dim=0) for v in per_video_logits])
    label_tensor = torch.tensor(all_labels, dtype=torch.long)
    avg_loss = nn.CrossEntropyLoss()(avg_logits, label_tensor).item()
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    print(f"\nTest loss    : {avg_loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
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
        eval_video_mae(ckpt, args.batch_size, args.num_clips)

    elif args.model == "vivit":
        from config.models import ViViTConfig
        ckpt = args.checkpoint or str(repo_root / ViViTConfig().output_dir)
        print(f"Model      : ViViT")
        print(f"Checkpoint : {ckpt}\n")
        eval_vivit(ckpt, args.batch_size, args.num_clips)

    elif args.model == "two_stream":
        from config.models.two_stream_config import TwoStreamConfig
        default_ckpt = repo_root / TwoStreamConfig().output_dir / "best.pt"
        ckpt = args.checkpoint or str(default_ckpt)
        print(f"Model      : Two-Stream CNN")
        print(f"Checkpoint : {ckpt}\n")
        eval_two_stream(ckpt, args.batch_size, args.num_clips)

    elif args.model == "pose":
        from config.models.pose_config import PoseConfig
        default_ckpt = repo_root / PoseConfig().output_dir / "best.pt"
        ckpt = args.checkpoint or str(default_ckpt)
        print(f"Model      : Pose MLP")
        print(f"Checkpoint : {ckpt}\n")
        eval_pose(ckpt, args.batch_size)

    elif args.model == "cnn_fusion":
        from config.models.cnn_fusion_config import CNNFusionConfig
        default_ckpt = repo_root / CNNFusionConfig().output_dir / "best.pt"
        ckpt = args.checkpoint or str(default_ckpt)
        print(f"Model      : CNN Fusion")
        print(f"Checkpoint : {ckpt}\n")
        eval_cnn_fusion(ckpt, args.batch_size, args.num_clips)


if __name__ == "__main__":
    main()
