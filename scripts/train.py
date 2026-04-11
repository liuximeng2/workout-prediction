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

from config.models import VideoMAEConfig, ViViTConfig
from model import build_model
from utils import (
    build_datasets,
    build_label_maps,
    make_train_transform,
    make_val_transform,
    resolve_data_roots,
    resolve_roots_for_label_maps,
)

# ── Registry of available model configs ──────────────────────────────────────
MODEL_CONFIGS = {
    "video_mae": VideoMAEConfig,
    "vivit": ViViTConfig,
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

    data_roots = resolve_data_roots(cfg.data_roots)

    # ── Label maps ────────────────────────────────────────────────────────────
    label2id, id2label = build_label_maps(
        resolve_roots_for_label_maps(cfg.data_roots, cfg.test_data_roots)
    )
    print(f"Classes ({len(label2id)}): {list(label2id.keys())}")

    # ── Model & processor ─────────────────────────────────────────────────────
    model, image_processor = build_model(
        cfg.model_name,
        model_ckpt=cfg.model_ckpt,
        label2id=label2id,
        id2label=id2label,
    )

    # Import model-specific helpers dynamically
    if cfg.model_name == "vivit":
        from model.vivit.model import apply_freeze_strategy, get_video_params
    else:
        from model.video_mae.model import apply_freeze_strategy, get_video_params

    apply_freeze_strategy(model, cfg.freeze_strategy)
    mean, std, resize_to = get_video_params(image_processor)

    num_frames = model.config.num_frames
    clip_duration = num_frames * cfg.sample_rate / cfg.fps
    print(f"num_frames={num_frames}, clip_duration={clip_duration:.2f}s")

    # ── Datasets ──────────────────────────────────────────────────────────────
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
        data_collator=collate_fn,
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
