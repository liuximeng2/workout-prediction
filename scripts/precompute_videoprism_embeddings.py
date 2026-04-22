import sys
sys.stdout.reconfigure(encoding='utf-8')

"""Pre-compute and cache VideoPrism visual embeddings.

Uses the official JAX/Flax videoprism library (pip install videoprism) to run
the ViT-B encoder.  Only the encoder is used; no fine-tuning happens here.
Saves a 768-dim mean-pooled embedding per video.

Usage:
    python scripts/precompute_videoprism_embeddings.py
    python scripts/precompute_videoprism_embeddings.py --cache_dir data/videoprism_embeddings
    python scripts/precompute_videoprism_embeddings.py --num_frames 16

Requirements (precompute only):
    pip install videoprism          # pulls JAX, Flax, TF-CPU
    # GPU acceleration requires Linux/WSL; CPU JAX works on Windows.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from config.models.videoprism_config import VideoPrismConfig
from utils.dataset import (
    VideoClipDataset,
    _collect_labeled_paths,
    build_label_maps,
    resolve_data_roots,
    resolve_roots_for_label_maps,
)

# VideoPrism expects frames in [0, 1], channels-last — no image processor needed.
_RESIZE = (288, 288)


def _cache_path(video_path: str, cache_dir: Path) -> Path:
    p = Path(video_path)
    return cache_dir / p.parent.name / f"{p.stem}.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute VideoPrism visual embeddings"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/videoprism_embeddings",
        help="Directory to write .pt cache files",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=VideoPrismConfig.num_frames,
        help="Frames per clip (default: 16)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = VideoPrismConfig()
    cache_dir = Path(args.cache_dir)
    repo_root = Path(__file__).resolve().parent.parent

    # ── Load JAX model ────────────────────────────────────────────────────────
    try:
        from videoprism import models as vp
        import jax
        import jax.numpy as jnp
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency: {e}\n"
            "Run: pip install videoprism\n"
            "(Pulls JAX, Flax, and TF-CPU. GPU JAX requires Linux/WSL.)"
        ) from e

    print("Loading VideoPrism weights…")
    model_name = "videoprism_public_v1_base"
    vp_model = vp.get_model(model_name)
    params = vp.load_pretrained_weights(model_name)["params"]
    print("  weights loaded.")

    @jax.jit
    def encode_clip(params, clip_nthwc):
        # clip_nthwc: (1, T, H, W, 3) float32 [0, 1]
        tokens, _ = vp_model.apply({"params": params}, clip_nthwc)  # (1, T*256, 768)
        return tokens.mean(axis=1)  # (1, 768)

    # ── Collect all video paths ───────────────────────────────────────────────
    data_roots = resolve_data_roots(cfg.data_roots, repo_root=repo_root)
    label_roots = resolve_roots_for_label_maps(
        cfg.data_roots, cfg.test_data_roots, repo_root=repo_root
    )
    label2id, id2label = build_label_maps(label_roots)
    all_paths = _collect_labeled_paths(data_roots, label2id)
    print(f"Videos to process: {len(all_paths)}")

    # ── Video sampling params ─────────────────────────────────────────────────
    clip_duration = args.num_frames * cfg.sample_rate / cfg.fps

    # mean=[0,0,0] / std=[1,1,1] → pipeline just does /255, leaving values in [0,1].
    from utils.transforms import make_val_transform
    val_transform = make_val_transform(args.num_frames, _RESIZE, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

    # ── Pre-compute ────────────────────────────────────────────────────────────
    skipped = 0
    for i, (video_path, info) in enumerate(all_paths):
        label = info["label"]
        out_path = _cache_path(video_path, cache_dir)

        if out_path.exists():
            skipped += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Decode center clip
        dataset = VideoClipDataset(
            labeled_paths=[(video_path, info)],
            clip_duration=clip_duration,
            transform=val_transform,
            mode="uniform",
        )
        try:
            sample = dataset[0]
        except Exception as e:
            print(f"  [WARN] skipping {video_path}: {e}")
            continue

        video_tensor = sample["video"]  # (C, T, H, W), float [0,1]

        # Convert to (1, T, H, W, C) JAX array — pass all frames as a single clip
        # so the temporal-attention layers see the full temporal context.
        frames_np = video_tensor.permute(1, 2, 3, 0).numpy().astype(np.float32)  # (T, H, W, 3)
        frames_np = np.clip(frames_np, 0.0, 1.0)
        clip_jax = jnp.array(frames_np[None, :, :, :, :])  # (1, T, H, W, 3)

        emb_jax = encode_clip(params, clip_jax)  # (1, 768)
        embedding = torch.from_numpy(np.array(emb_jax[0]).astype(np.float32))  # (768,)

        torch.save({"embedding": embedding, "label": label}, out_path)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i + 1}/{len(all_paths)}] {Path(video_path).name} → {out_path}")

    total_cached = len(all_paths) - skipped
    print(
        f"\nDone. {total_cached} new embeddings written, {skipped} already existed."
        f"\nCache directory: {cache_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
