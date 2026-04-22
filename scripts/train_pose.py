"""Train the Pose MLP classifier on engineered keypoint features.

Uses a standard PyTorch training loop (Adam + StepLR).

Usage:
    python scripts/train_pose.py
    python scripts/train_pose.py --num_epochs 200 --batch_size 32 --learning_rate 1e-3
    python scripts/train_pose.py --hidden_dim 512 --num_layers 4 --dropout 0.4
    python scripts/train_pose.py --resume checkpoints/pose-workout/best.pt
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.models.pose_config import PoseConfig
from model.pose.model import PoseMLP, build_model
from utils.dataset import (
    build_label_maps,
    resolve_data_roots,
    resolve_roots_for_label_maps,
)
from utils.pose_dataset import PoseFeatureDataset, pose_path_for_video
from utils.splits import canonical_splits, filter_split


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch):
    features = torch.stack([b["features"] for b in batch])   # (N, D)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"features": features, "label": labels}


# ── Epoch helpers ─────────────────────────────────────────────────────────────

def run_epoch(
    model: PoseMLP,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    log_every: int = 10,
    phase: str = "train",
) -> tuple[float, float]:
    """Run one epoch.  Returns (avg_loss, accuracy)."""
    is_train = phase == "train"
    model.train(is_train)

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for step, batch in enumerate(loader, 1):
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            correct += (logits.argmax(1) == labels).sum().item()
            total += bs

            if step % log_every == 0:
                print(
                    f"  [{phase}] step {step}/{len(loader)} "
                    f"loss={loss.item():.4f} "
                    f"acc={correct/total:.3f}"
                )

    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Pose MLP classifier.")
    parser.add_argument("--hidden_dim",    type=int,   default=None)
    parser.add_argument("--num_layers",    type=int,   default=None)
    parser.add_argument("--dropout",       type=float, default=None)
    parser.add_argument("--num_epochs",    type=int,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay",  type=float, default=None)
    parser.add_argument("--output_dir",    type=str,   default=None)
    parser.add_argument("--resume",        type=str,   default=None,
                        help="Path to a checkpoint .pt file to resume training from.")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = PoseConfig()
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k != "resume"}
    if overrides:
        cfg = replace(cfg, **overrides)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device : {device}")
    print(f"Config : {cfg}")

    # ── Data ──────────────────────────────────────────────────────────────────
    repo_root = Path(__file__).resolve().parent.parent
    data_roots = list(resolve_data_roots(cfg.data_roots, repo_root=repo_root))
    label_roots = resolve_roots_for_label_maps(
        cfg.data_roots, cfg.test_data_roots, repo_root=repo_root
    )
    label2id, id2label = build_label_maps(label_roots)
    num_classes = len(label2id)
    print(f"Classes: {num_classes}")

    pose_root = repo_root / cfg.pose_root

    # Canonical split first, then drop videos lacking pose *per split*.
    train_paths, val_paths, _ = canonical_splits(
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

    train_paths = filter_split(train_paths, _has_pose, name="train")
    val_paths   = filter_split(val_paths,   _has_pose, name="val")

    train_ds = PoseFeatureDataset(
        labeled_paths=train_paths,
        pose_root=pose_root,
        repo_root=repo_root,
    )
    val_ds = PoseFeatureDataset(
        labeled_paths=val_paths,
        pose_root=pose_root,
        repo_root=repo_root,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn,
    )
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, _ = build_model(
        num_classes=num_classes,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    model = model.to(device)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
    )
    criterion = nn.CrossEntropyLoss()

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        history = ckpt.get("history", history)
        print(f"Resumed from epoch {ckpt['epoch']} (best val acc: {best_val_acc:.3f})")

    # ── Output dir ────────────────────────────────────────────────────────────
    out_dir = repo_root / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({k: str(v) for k, v in asdict(cfg).items()}, indent=2)
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        t0 = time.perf_counter()
        print(f"\n── Epoch {epoch}/{cfg.num_epochs}  lr={scheduler.get_last_lr()[0]:.2e} ──")

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            log_every=cfg.log_every_n_steps, phase="train",
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device,
            log_every=cfg.log_every_n_steps, phase="val",
        )
        scheduler.step()

        elapsed = time.perf_counter() - t0
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
            f"{elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save latest checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "history": history,
            "label2id": label2id,
            "id2label": id2label,
        }
        torch.save(ckpt, out_dir / "last.pt")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt["best_val_acc"] = best_val_acc
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  ★ New best val acc: {best_val_acc:.3f}")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.3f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()
