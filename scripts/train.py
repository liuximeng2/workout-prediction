"""Fine-tune a video classification model on the gym workout dataset.

Usage:
    python scripts/train.py
    python scripts/train.py --model video_mae
    python scripts/train.py --batch_size 8 --num_epochs 15 --learning_rate 3e-5
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# Allow imports from the repo root regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import evaluate
from transformers import TrainingArguments, Trainer

from config.models import VideoMAEConfig, ViViTConfig, LlavaOnevisionConfig, VideoSalmonnConfig, VideoPrismConfig
from model import build_model
from utils import (
    build_datasets,
    build_label_maps,
    make_train_transform,
    make_val_transform,
    resolve_data_roots,
    resolve_roots_for_label_maps,
)
from utils.dataset import _collect_labeled_paths, _split_paths
from utils.embedding_dataset import EmbeddingDataset

# ── Registry of available model configs ──────────────────────────────────────
MODEL_CONFIGS = {
    "video_mae":       VideoMAEConfig,
    "vivit":           ViViTConfig,
    "llava_onevision": LlavaOnevisionConfig,
    "video_salmonn":   VideoSalmonnConfig,
    "videoprism":      VideoPrismConfig,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a workout video classifier")
    parser.add_argument(
        "--model",
        type=str,
        default="video_mae",
        choices=sorted(MODEL_CONFIGS),
        help="Which model architecture to use (default: video_mae)",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--model_ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument(
        "--freeze_strategy",
        type=str,
        default=None,
        choices=["full", "head_only"],
        help="Parameter freeze strategy (default: from model config)",
    )
    parser.add_argument(
        "--text_conditioned",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use text-conditioned embedding cache. "
             "Defaults to True for llava_onevision, False for video_salmonn. "
             "Use --no_text_conditioned to override.",
    )
    return parser.parse_args()


def collate_fn(examples):
    """
    Batch listed video clips and labels into model-ready tensors.
    example:
    {
        "video": torch.Tensor(1, 3, 224, 224),
        "label": int,
    }
    """
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def _embedding_collate_fn(examples):
    """Collate pre-computed embeddings — no video decode or model forward needed."""
    return {
        "embedding": torch.stack([e["embedding"] for e in examples]),
        "labels": torch.tensor([e["label"] for e in examples]),
    }


def make_llava_collate_fn(processor):
    """Return a collate_fn that also encodes a text prompt for LLaVA-OneVision."""
    import numpy as np
    from PIL import Image

    # LLaVA-OneVision chat template: wrap the prompt with the image token.
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "Classify the exercise being performed."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    def _collate(examples):
        labels = torch.tensor([example["label"] for example in examples])

        # Convert each clip's frames to a list of PIL Images (one clip = one video).
        # processor expects: text=str, videos=list[list[PIL.Image]]
        videos = []
        for example in examples:
            video = example["video"]  # (C, T, H, W)
            frames = [
                Image.fromarray(
                    (video[:, t].permute(1, 2, 0).numpy() * 255)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                for t in range(video.shape[1])
            ]
            videos.append(frames)

        # Use the full processor so it returns pixel_values AND image_sizes.
        enc = processor(
            text=[prompt] * len(examples),
            videos=videos,
            return_tensors="pt",
            padding=True,
        )

        return {
            "pixel_values_videos": enc["pixel_values_videos"],
            "image_sizes": enc.get("image_sizes"),
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    return _collate


def main():
    args = parse_args()

    # ── Build config (apply CLI overrides) ────────────────────────────────────
    cfg = MODEL_CONFIGS[args.model]()
    if args.batch_size is not None:
        cfg = replace(cfg, batch_size=args.batch_size)
    if args.num_epochs is not None:
        cfg = replace(cfg, num_epochs=args.num_epochs)
    if args.learning_rate is not None:
        cfg = replace(cfg, learning_rate=args.learning_rate)
    if args.model_ckpt is not None:
        cfg = replace(cfg, model_ckpt=args.model_ckpt)
    if args.output_dir is not None:
        cfg = replace(cfg, output_dir=args.output_dir)
    if args.push_to_hub:
        cfg = replace(cfg, push_to_hub=True)
    if args.freeze_strategy is not None:
        cfg = replace(cfg, freeze_strategy=args.freeze_strategy)
    if args.text_conditioned is not None:
        cfg = replace(cfg, text_conditioned=args.text_conditioned)

    data_roots = resolve_data_roots(cfg.data_roots)

    # ── Label maps ────────────────────────────────────────────────────────────
    label2id, id2label = build_label_maps(
        resolve_roots_for_label_maps(cfg.data_roots, cfg.test_data_roots)
    )
    print(f"Classes ({len(label2id)}): {list(label2id.keys())}")

    # ── Datasets (decide early so we can skip model load when cached) ─────────
    repo_root = Path(__file__).resolve().parent.parent

    def _cache_dir_for(model_name: str, text_conditioned: bool) -> Path:
        if model_name == "llava_onevision":
            return repo_root / "data" / ("llava_embeddings" if text_conditioned else "llava_embeddings_notxt")
        if model_name == "video_salmonn":
            return repo_root / "data" / ("salmonn_embeddings_text" if text_conditioned else "salmonn_embeddings")
        if model_name == "videoprism":
            return repo_root / "data" / "videoprism_embeddings"
        return None

    videoprism_cache_dir = repo_root / "data" / "videoprism_embeddings"

    if cfg.model_name == "llava_onevision":
        candidate = _cache_dir_for("llava_onevision", cfg.text_conditioned)
        embedding_cache_dir = candidate if candidate.is_dir() else None
    elif cfg.model_name == "video_salmonn":
        candidate = _cache_dir_for("video_salmonn", cfg.text_conditioned)
        if not candidate.is_dir():
            script = ("precompute_salmonn_embeddings.py --text_conditioned"
                      if cfg.text_conditioned else "precompute_salmonn_embeddings.py")
            raise RuntimeError(
                f"Embedding cache not found at {candidate}. "
                f"Run scripts/{script} first."
            )
        embedding_cache_dir = candidate
    elif cfg.model_name == "videoprism":
        if not videoprism_cache_dir.is_dir():
            raise RuntimeError(
                f"Embedding cache not found at {videoprism_cache_dir}. "
                "Run scripts/precompute_videoprism_embeddings.py first."
            )
        embedding_cache_dir = videoprism_cache_dir
    else:
        embedding_cache_dir = None

    use_embedding_cache = embedding_cache_dir is not None

    # ── Model & processor ─────────────────────────────────────────────────────
    if use_embedding_cache:
        if cfg.model_name == "llava_onevision":
            from model.llava_onevision.model import build_model_cached
        elif cfg.model_name == "video_salmonn":
            from model.video_salmonn.model import build_model_cached
        else:
            from model.videoprism.model import build_model_cached
        model, image_processor = build_model_cached(label2id)
    else:
        model, image_processor = build_model(
            cfg.model_name,
            model_ckpt=cfg.model_ckpt,
            label2id=label2id,
            id2label=id2label,
        )

        # Import model-specific helpers dynamically
        if cfg.model_name == "vivit":
            from model.vivit.model import apply_freeze_strategy, get_video_params
        elif cfg.model_name == "llava_onevision":
            from model.llava_onevision.model import apply_freeze_strategy, get_video_params
        else:
            from model.video_mae.model import apply_freeze_strategy, get_video_params

        apply_freeze_strategy(model, cfg.freeze_strategy)
        mean, std, resize_to = get_video_params(image_processor)

        if cfg.model_name == "llava_onevision":
            num_frames = cfg.num_frames
        else:
            num_frames = model.config.num_frames
        clip_duration = num_frames * cfg.sample_rate / cfg.fps
        print(f"num_frames={num_frames}, clip_duration={clip_duration:.2f}s")

    # ── Build datasets ────────────────────────────────────────────────────────
    if use_embedding_cache:
        print(f"Using cached embeddings from {embedding_cache_dir}")
        all_paths = _collect_labeled_paths(data_roots, label2id)
        train_paths, val_paths, _ = _split_paths(
            all_paths, cfg.train_split, cfg.val_split, cfg.seed
        )
        train_dataset = EmbeddingDataset(train_paths, embedding_cache_dir)
        val_dataset = EmbeddingDataset(val_paths, embedding_cache_dir)
        data_collator = _embedding_collate_fn
    else:
        train_transform = make_train_transform(num_frames, resize_to, mean, std)
        val_transform = make_val_transform(num_frames, resize_to, mean, std)

        train_dataset, val_dataset, _ = build_datasets(
            data_roots=data_roots,
            label2id=label2id,
            clip_duration=clip_duration,
            train_transform=train_transform,
            val_transform=val_transform,
            train_split=cfg.train_split,
            val_split=cfg.val_split,
            seed=cfg.seed,
        )
        if cfg.model_name == "llava_onevision":
            data_collator = make_llava_collate_fn(image_processor)
        else:
            data_collator = collate_fn

    print(
        f"Dataset sizes — train: {train_dataset.num_videos}, "
        f"val: {val_dataset.num_videos}"
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return accuracy_metric.compute(
            predictions=predictions, references=eval_pred.label_ids
        )

    # ── Training arguments ────────────────────────────────────────────────────
    max_steps = (train_dataset.num_videos // cfg.batch_size) * cfg.num_epochs

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best_model,
        push_to_hub=cfg.push_to_hub,
        max_steps=max_steps,
        seed=cfg.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Starting training…")
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)

    if cfg.push_to_hub:
        trainer.push_to_hub()

    print(f"Training complete. Checkpoints saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
