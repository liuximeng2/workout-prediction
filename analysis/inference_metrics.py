"""Test-set metrics and inference timing (aligned with ``scripts/eval.py``)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from tqdm.auto import tqdm
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from config.models import VideoMAEConfig
from model.video_mae.model import get_video_params
from utils import (
    build_datasets,
    build_label_maps,
    make_val_transform,
    resolve_data_roots,
    resolve_roots_for_label_maps,
)


@dataclass
class TestEvaluationResult:
    """Aggregated test metrics and timing."""

    accuracy: float
    f1_macro: float
    f1_micro: float
    f1_weighted: float
    n_samples: int
    n_batches_timed: int
    total_time_s: float
    wall_time_per_sample_s: float
    throughput_samples_per_s: float
    classification_report: str
    per_class_support: dict[str, int]


def _collate_fn(examples: list) -> dict[str, torch.Tensor]:
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def evaluate_checkpoint_on_test(
    checkpoint: str | Path | None = None,
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    warmup_batches: int = 2,
    cfg: VideoMAEConfig | None = None,
    repo_root: str | Path | None = None,
    show_progress: bool = True,
) -> TestEvaluationResult:
    """Run the test loader, compute accuracy / F1, and measure inference time.

    Default ``batch_size=1`` runs clips **one at a time** (good for progress visibility
    and peak memory); increase for throughput. Warm-up batches are excluded from timing.

    ``repo_root``: project root for resolving ``cfg.data_roots`` (default: parent of
    ``utils/``). Set this if your notebook cwd is not the repo root.

    ``show_progress``: tqdm bar over batches (or samples when ``batch_size==1``).
    """
    cfg = cfg or VideoMAEConfig()
    ckpt = str(Path(checkpoint or cfg.output_dir).resolve())
    rr = Path(repo_root) if repo_root else None
    data_roots = resolve_data_roots(cfg.data_roots, repo_root=rr)
    test_roots = (
        resolve_data_roots(cfg.test_data_roots, repo_root=rr) if cfg.test_data_roots else None
    )

    image_processor = VideoMAEImageProcessor.from_pretrained(ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(ckpt)
    model.eval()

    mean, std, resize_to = get_video_params(image_processor)
    num_frames = model.config.num_frames
    clip_duration = num_frames * cfg.sample_rate / cfg.fps

    label2id, id2label = build_label_maps(
        resolve_roots_for_label_maps(cfg.data_roots, cfg.test_data_roots, repo_root=rr)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=_collate_fn,
        num_workers=num_workers,
    )

    n_batches = len(loader)
    unit = "sample" if batch_size == 1 else "batch"
    pbar_kw: dict = {
        "total": n_batches,
        "desc": "Test inference",
        "unit": unit,
        "disable": not show_progress,
    }

    all_preds: list[int] = []
    all_labels: list[int] = []

    timed_batches = 0
    t_infer = 0.0

    with torch.no_grad():
        it = enumerate(loader)
        if show_progress:
            it = tqdm(it, **pbar_kw)
        for bi, batch in it:
            pv = batch["pixel_values"].to(device)
            labels = batch["labels"]

            if show_progress:
                it.set_postfix(phase="warmup" if bi < warmup_batches else "timed")

            if bi < warmup_batches:
                _ = model(pixel_values=pv)
                continue

            t0 = time.perf_counter()
            outputs = model(pixel_values=pv)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_infer += time.perf_counter() - t0
            timed_batches += 1

            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    y_true = np.array(all_labels, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)

    accuracy = float((y_true == y_pred).mean())
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        zero_division=0,
    )

    n_samples = len(all_labels)
    per_class_support: dict[str, int] = {}
    for i, name in enumerate(target_names):
        per_class_support[name] = int((y_true == i).sum())

    wall_per_sample = t_infer / n_samples if n_samples else 0.0
    thr = n_samples / t_infer if t_infer > 0 else 0.0

    return TestEvaluationResult(
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_micro=f1_micro,
        f1_weighted=f1_weighted,
        n_samples=n_samples,
        n_batches_timed=timed_batches,
        total_time_s=t_infer,
        wall_time_per_sample_s=wall_per_sample,
        throughput_samples_per_s=thr,
        classification_report=report,
        per_class_support=per_class_support,
    )
