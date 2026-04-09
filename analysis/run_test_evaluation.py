#!/usr/bin/env python3
"""CLI: test-set F1 and inference speed for a fine-tuned checkpoint (see ``inference_metrics``)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.inference_metrics import evaluate_checkpoint_on_test
from config.models import VideoMAEConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on test split with F1 and timing")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory (default: VideoMAEConfig.output_dir)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="1 = one clip per step (default; matches tqdm sample bar). Use 4–8 for speed.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers (0 is safest with tqdm/notebooks; use 2+ with batch_size>1).",
    )
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument(
        "--repo_root",
        type=str,
        default=None,
        help="Project root for config data paths (default: infer from utils/)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm (for logs / non-interactive)",
    )
    args = parser.parse_args()

    r = evaluate_checkpoint_on_test(
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup_batches=args.warmup_batches,
        repo_root=args.repo_root,
        show_progress=not args.no_progress,
    )
    print(f"Samples: {r.n_samples}")
    print(f"Accuracy:  {r.accuracy:.4f}")
    print(f"F1 macro:  {r.f1_macro:.4f}")
    print(f"F1 micro:  {r.f1_micro:.4f}")
    print(f"F1 weighted: {r.f1_weighted:.4f}")
    print(f"Inference (timed): {r.total_time_s:.2f}s for {r.n_batches_timed} batches")
    print(f"Wall time / sample: {r.wall_time_per_sample_s * 1000:.2f} ms")
    print(f"Throughput: {r.throughput_samples_per_s:.3f} samples/s")
    print("\n" + r.classification_report)


if __name__ == "__main__":
    main()
