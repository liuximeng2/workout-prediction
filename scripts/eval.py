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
    build_label_maps,
    flow_dir_for_video,
    resolve_data_roots,
    resolve_roots_for_label_maps,
)
from utils.splits import canonical_splits, filter_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a workout video classifier on the test split.")
    parser.add_argument(
        "--model",
        type=str,
        default="video_mae",
        choices=["video_mae", "two_stream", "vivit", "pose", "cnn_fusion", "llava_onevision", "video_salmonn", "videoprism"],
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
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which data split to evaluate (default: test).",
    )
    parser.add_argument(
        "--text_conditioned",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use text-conditioned embedding cache. "
             "Defaults to True for llava_onevision, False for video_salmonn. "
             "Use --no_text_conditioned to evaluate the no-text variant.",
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

def eval_video_mae(ckpt: str, batch_size: int, num_clips: int = 1, split: str = "test") -> None:
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
    datasets = build_datasets(
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
    split_dataset = {"train": datasets[0], "val": datasets[1], "test": datasets[2]}[split]
    print(f"{split.capitalize()} videos: {split_dataset.num_videos}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    positions = _clip_positions(num_clips)
    n_videos = len(split_dataset)
    per_video_logits = [[] for _ in range(n_videos)]
    video_labels = [0] * n_videos

    print(f"Clips/video: {num_clips}  positions: "
          f"{[f'{p:.2f}' for p in positions]}")

    with torch.no_grad():
        for clip_idx, pos in enumerate(positions):
            split_dataset.clip_position = pos
            loader = DataLoader(split_dataset, batch_size=batch_size,
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
    print(f"\n{split.capitalize()} accuracy: {results['accuracy']:.4f}")
    _print_per_class(all_preds, all_labels, id2label)


def eval_vivit(ckpt: str, batch_size: int, num_clips: int = 1, split: str = "test") -> None:
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
    datasets = build_datasets(
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
    split_dataset = {"train": datasets[0], "val": datasets[1], "test": datasets[2]}[split]
    print(f"{split.capitalize()} videos: {split_dataset.num_videos}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    positions = _clip_positions(num_clips)
    n_videos = len(split_dataset)
    per_video_logits = [[] for _ in range(n_videos)]
    video_labels = [0] * n_videos

    print(f"Clips/video: {num_clips}  positions: "
          f"{[f'{p:.2f}' for p in positions]}")

    with torch.no_grad():
        for clip_idx, pos in enumerate(positions):
            split_dataset.clip_position = pos
            loader = DataLoader(split_dataset, batch_size=batch_size,
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
    print(f"\n{split.capitalize()} accuracy: {results['accuracy']:.4f}")
    _print_per_class(all_preds, all_labels, id2label)


# ── Pose MLP evaluation ──────────────────────────────────────────────────────

def eval_pose(ckpt_path: str, batch_size: int, split: str = "test") -> None:
    from config.models.pose_config import PoseConfig
    from model.pose.model import build_model
    from utils.pose_dataset import PoseFeatureDataset, pose_path_for_video

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
    pose_root = repo_root / cfg.pose_root

    train_paths, val_paths, test_paths = canonical_splits(
        data_roots=cfg.data_roots,
        test_data_roots=cfg.test_data_roots,
        label2id=label2id,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        seed=cfg.seed,
        repo_root=repo_root,
    )

    def _has_pose(path_str: str) -> bool:
        return pose_path_for_video(Path(path_str), pose_root, repo_root).exists()

    split_paths = {"train": train_paths, "val": val_paths, "test": test_paths}[split]
    eval_paths = filter_split(split_paths, _has_pose, name=split)

    test_ds = PoseFeatureDataset(
        labeled_paths=eval_paths,
        pose_root=pose_root,
        repo_root=repo_root,
    )
    print(f"{split.capitalize()} samples: {len(test_ds)}")

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

def eval_cnn_fusion(ckpt_path: str, batch_size: int, num_clips: int = 1, split: str = "test") -> None:
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
    train_paths, val_paths, test_paths = canonical_splits(
        data_roots=cfg.data_roots,
        test_data_roots=cfg.test_data_roots,
        label2id=label2id,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        seed=cfg.seed,
        repo_root=repo_root,
    )
    eval_paths = {"train": train_paths, "val": val_paths, "test": test_paths}[split]

    val_transform = make_val_transform(cfg.num_frames, _RESIZE_TO, _IMAGENET_MEAN, _IMAGENET_STD)
    test_ds = VideoClipDataset(
        labeled_paths=eval_paths,
        clip_duration=cfg.clip_duration,
        transform=val_transform,
        mode="uniform",
    )
    print(f"{split.capitalize()} samples: {len(test_ds)}")

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


def _embedding_collate(examples):
    return {
        "embedding": torch.stack([e["embedding"] for e in examples]),
        "labels": torch.tensor([e["label"] for e in examples]),
    }


def _two_stream_collate(batch):
    videos = torch.stack([b["video"] for b in batch])
    flows  = torch.stack([b["flow"]  for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"video": videos, "flow": flows, "label": labels}


# ── Two-Stream evaluation ─────────────────────────────────────────────────────

def eval_two_stream(ckpt_path: str, batch_size: int, num_clips: int = 1, split: str = "test") -> None:
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
    flow_root = repo_root / cfg.flow_root

    train_paths, val_paths, test_paths = canonical_splits(
        data_roots=cfg.data_roots,
        test_data_roots=cfg.test_data_roots,
        label2id=label2id,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        seed=cfg.seed,
        repo_root=repo_root,
    )

    def _has_flow(path_str: str) -> bool:
        fdir = flow_dir_for_video(Path(path_str), flow_root, repo_root)
        return (fdir / "meta.npy").exists()

    split_paths = {"train": train_paths, "val": val_paths, "test": test_paths}[split]
    eval_paths = filter_split(split_paths, _has_flow, name=split)

    test_ds = TwoStreamDataset(
        labeled_paths=eval_paths,
        flow_root=flow_root,
        clip_duration=cfg.clip_duration,
        num_flow_frames=cfg.num_flow_frames,
        mode="val",
        repo_root=repo_root,
    )
    print(f"{split.capitalize()} samples: {len(test_ds)}")

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


# ── LLaVA-OneVision evaluation (cached embeddings) ───────────────────────────

def eval_llava_onevision(ckpt_path: str, batch_size: int, split: str = "test",
                         text_conditioned: bool = True) -> None:
    from config.models import LlavaOnevisionConfig
    from model.llava_onevision.model import LlavaOnevisionHeadOnly
    from utils.embedding_dataset import EmbeddingDataset

    cfg = LlavaOnevisionConfig()
    repo_root = Path(__file__).resolve().parent.parent
    cache_dir = repo_root / "data" / ("llava_embeddings" if text_conditioned else "llava_embeddings_notxt")

    if not cache_dir.is_dir():
        precompute_cmd = ("precompute_llava_embeddings.py" if text_conditioned
                          else "precompute_llava_embeddings.py --no_text_conditioned")
        raise RuntimeError(
            f"Embedding cache not found at {cache_dir}. "
            f"Run scripts/{precompute_cmd} first."
        )

    label2id, id2label = build_label_maps(
        resolve_roots_for_label_maps(cfg.data_roots, cfg.test_data_roots, repo_root=repo_root)
    )
    train_paths, val_paths, test_paths = canonical_splits(
        data_roots=cfg.data_roots,
        test_data_roots=cfg.test_data_roots,
        label2id=label2id,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        seed=cfg.seed,
        repo_root=repo_root,
    )
    eval_paths = {"train": train_paths, "val": val_paths, "test": test_paths}[split]

    test_ds = EmbeddingDataset(eval_paths, cache_dir)
    print(f"{split.capitalize()} videos: {test_ds.num_videos}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from safetensors.torch import load_file
    state_dict = load_file(ckpt_path, device=str(device))

    model = LlavaOnevisionHeadOnly(num_labels=len(label2id))
    model.load_state_dict(state_dict)
    model.eval().to(device)

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=_embedding_collate, num_workers=2)

    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            emb = batch["embedding"].to(device)
            labels = batch["labels"].to(device)
            logits = model(emb).float()
            total_loss += criterion(logits, labels).item() * labels.size(0)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    avg_loss = total_loss / len(all_labels)
    print(f"\nTest loss    : {avg_loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    _print_per_class(all_preds, all_labels, id2label)


# ── video-SALMONN-2 evaluation (cached embeddings) ───────────────────────────

def eval_video_salmonn(ckpt_path: str, batch_size: int, split: str = "test",
                       text_conditioned: bool = False) -> None:
    from config.models.video_salmonn_config import VideoSalmonnConfig
    from model.video_salmonn.model import VideoSalmonnHeadOnly
    from utils.embedding_dataset import EmbeddingDataset

    cfg = VideoSalmonnConfig()
    repo_root = Path(__file__).resolve().parent.parent
    cache_dir = repo_root / "data" / ("salmonn_embeddings_text" if text_conditioned else "salmonn_embeddings")

    if not cache_dir.is_dir():
        precompute_cmd = ("precompute_salmonn_embeddings.py --text_conditioned" if text_conditioned
                          else "precompute_salmonn_embeddings.py")
        raise RuntimeError(
            f"Embedding cache not found at {cache_dir}. "
            f"Run scripts/{precompute_cmd} first."
        )

    label2id, id2label = build_label_maps(
        resolve_roots_for_label_maps(cfg.data_roots, cfg.test_data_roots, repo_root=repo_root)
    )
    train_paths, val_paths, test_paths = canonical_splits(
        data_roots=cfg.data_roots,
        test_data_roots=cfg.test_data_roots,
        label2id=label2id,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        seed=cfg.seed,
        repo_root=repo_root,
    )
    eval_paths = {"train": train_paths, "val": val_paths, "test": test_paths}[split]

    test_ds = EmbeddingDataset(eval_paths, cache_dir)
    print(f"{split.capitalize()} videos: {test_ds.num_videos}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from safetensors.torch import load_file
    state_dict = load_file(ckpt_path, device=str(device))

    model = VideoSalmonnHeadOnly(num_labels=len(label2id))
    model.load_state_dict(state_dict)
    model.eval().to(device)

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=_embedding_collate, num_workers=2)

    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            emb = batch["embedding"].to(device)
            labels = batch["labels"].to(device)
            logits = model(emb).float()
            total_loss += criterion(logits, labels).item() * labels.size(0)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    avg_loss = total_loss / len(all_labels)
    print(f"\nTest loss    : {avg_loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    _print_per_class(all_preds, all_labels, id2label)


# ── VideoPrism evaluation (cached embeddings) ────────────────────────────────

def eval_videoprism(ckpt_path: str, batch_size: int, split: str = "test") -> None:
    from config.models.videoprism_config import VideoPrismConfig
    from model.videoprism.model import VideoPrismHeadOnly
    from utils.embedding_dataset import EmbeddingDataset

    cfg = VideoPrismConfig()
    repo_root = Path(__file__).resolve().parent.parent
    cache_dir = repo_root / "data" / "videoprism_embeddings"

    if not cache_dir.is_dir():
        raise RuntimeError(
            f"Embedding cache not found at {cache_dir}. "
            "Run scripts/precompute_videoprism_embeddings.py first."
        )

    label2id, id2label = build_label_maps(
        resolve_roots_for_label_maps(cfg.data_roots, cfg.test_data_roots, repo_root=repo_root)
    )
    train_paths, val_paths, test_paths = canonical_splits(
        data_roots=cfg.data_roots,
        test_data_roots=cfg.test_data_roots,
        label2id=label2id,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        seed=cfg.seed,
        repo_root=repo_root,
    )
    eval_paths = {"train": train_paths, "val": val_paths, "test": test_paths}[split]

    test_ds = EmbeddingDataset(eval_paths, cache_dir)
    print(f"{split.capitalize()} videos: {test_ds.num_videos}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from safetensors.torch import load_file
    state_dict = load_file(ckpt_path, device=str(device))

    model = VideoPrismHeadOnly(num_labels=len(label2id))
    model.load_state_dict(state_dict)
    model.eval().to(device)

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=_embedding_collate, num_workers=2)

    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            emb = batch["embedding"].to(device)
            labels = batch["labels"].to(device)
            logits = model(emb).float()
            total_loss += criterion(logits, labels).item() * labels.size(0)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    avg_loss = total_loss / len(all_labels)
    print(f"\nTest loss    : {avg_loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    _print_per_class(all_preds, all_labels, id2label)


# ── Shared per-class printer ──────────────────────────────────────────────────

def _print_per_class(
    preds: list,
    labels: list,
    id2label: dict,
) -> None:
    from sklearn.metrics import f1_score

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

    print(f"\nF1 macro:    {f1_score(labels, preds, average='macro',    zero_division=0):.4f}")
    print(f"F1 weighted: {f1_score(labels, preds, average='weighted', zero_division=0):.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    print(f"Split      : {args.split}\n")

    if args.model == "video_mae":
        from config.models import VideoMAEConfig
        ckpt = args.checkpoint or str(repo_root / VideoMAEConfig().output_dir)
        print(f"Model      : VideoMAE")
        print(f"Checkpoint : {ckpt}\n")
        eval_video_mae(ckpt, args.batch_size, args.num_clips, args.split)

    elif args.model == "vivit":
        from config.models import ViViTConfig
        ckpt = args.checkpoint or str(repo_root / ViViTConfig().output_dir)
        print(f"Model      : ViViT")
        print(f"Checkpoint : {ckpt}\n")
        eval_vivit(ckpt, args.batch_size, args.num_clips, args.split)

    elif args.model == "two_stream":
        from config.models.two_stream_config import TwoStreamConfig
        default_ckpt = repo_root / TwoStreamConfig().output_dir / "best.pt"
        ckpt = args.checkpoint or str(default_ckpt)
        print(f"Model      : Two-Stream CNN")
        print(f"Checkpoint : {ckpt}\n")
        eval_two_stream(ckpt, args.batch_size, args.num_clips, args.split)

    elif args.model == "pose":
        from config.models.pose_config import PoseConfig
        default_ckpt = repo_root / PoseConfig().output_dir / "best.pt"
        ckpt = args.checkpoint or str(default_ckpt)
        print(f"Model      : Pose MLP")
        print(f"Checkpoint : {ckpt}\n")
        eval_pose(ckpt, args.batch_size, args.split)

    elif args.model == "cnn_fusion":
        from config.models.cnn_fusion_config import CNNFusionConfig
        default_ckpt = repo_root / CNNFusionConfig().output_dir / "best.pt"
        ckpt = args.checkpoint or str(default_ckpt)
        print(f"Model      : CNN Fusion")
        print(f"Checkpoint : {ckpt}\n")
        eval_cnn_fusion(ckpt, args.batch_size, args.num_clips, args.split)

    elif args.model == "llava_onevision":
        from config.models import LlavaOnevisionConfig
        effective_text = args.text_conditioned if args.text_conditioned is not None else True
        default_ckpt_dir = "llava-onevision-workout" if effective_text else "llava-onevision-noprompt"
        default_ckpt = repo_root / "checkpoints" / default_ckpt_dir / "model.safetensors"
        ckpt = args.checkpoint or str(default_ckpt)
        print(f"Model      : LLaVA-OneVision (cached embeddings, text_conditioned={effective_text})")
        print(f"Checkpoint : {ckpt}\n")
        eval_llava_onevision(ckpt, args.batch_size, args.split, text_conditioned=effective_text)

    elif args.model == "video_salmonn":
        from config.models.video_salmonn_config import VideoSalmonnConfig
        default_ckpt = repo_root / VideoSalmonnConfig().output_dir / "model.safetensors"
        ckpt = args.checkpoint or str(default_ckpt)
        effective_text = args.text_conditioned if args.text_conditioned is not None else False
        print(f"Model      : video-SALMONN-2 (cached embeddings, text_conditioned={effective_text})")
        print(f"Checkpoint : {ckpt}\n")
        eval_video_salmonn(ckpt, args.batch_size, args.split, text_conditioned=effective_text)

    elif args.model == "videoprism":
        from config.models.videoprism_config import VideoPrismConfig
        default_ckpt = repo_root / VideoPrismConfig().output_dir / "model.safetensors"
        ckpt = args.checkpoint or str(default_ckpt)
        print(f"Model      : VideoPrism (cached embeddings)")
        print(f"Checkpoint : {ckpt}\n")
        eval_videoprism(ckpt, args.batch_size, args.split)


if __name__ == "__main__":
    main()
