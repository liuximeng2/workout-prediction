import sys
sys.stdout.reconfigure(encoding='utf-8')

"""Pre-compute and cache video-SALMONN-2 embeddings.

By default loads only SigLIP + mm_projector (visual-only, 3584-dim).
With --text_conditioned, also loads the full Qwen2-7B LLM and conditions
the embedding on a text prompt, similar to LLaVA OneVision.

Usage:
    python scripts/precompute_salmonn_embeddings.py
    python scripts/precompute_salmonn_embeddings.py --text_conditioned
    python scripts/precompute_salmonn_embeddings.py --cache_dir data/salmonn_embeddings
    python scripts/precompute_salmonn_embeddings.py --model_ckpt tsinghua-ee/video-SALMONN-2
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import SiglipImageProcessor

from config.models.video_salmonn_config import VideoSalmonnConfig
from model.video_salmonn.model import load_encoder, load_full_model
from utils.dataset import (
    VideoClipDataset,
    _collect_labeled_paths,
    build_label_maps,
    resolve_data_roots,
    resolve_roots_for_label_maps,
)
from utils.transforms import make_val_transform


def _cache_path(video_path: str, cache_dir: Path) -> Path:
    p = Path(video_path)
    return cache_dir / p.parent.name / f"{p.stem}.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute video-SALMONN-2 visual embeddings"
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=VideoSalmonnConfig.model_ckpt,
        help="HuggingFace checkpoint ID or local path",
    )
    parser.add_argument(
        "--text_conditioned",
        action="store_true",
        default=False,
        help="Load full Qwen2-7B LLM and condition embeddings on --text_prompt (~14 GB RAM)",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="Classify the exercise being performed.",
        help="Text prompt to condition on (only used with --text_conditioned)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to write .pt cache files "
             "(default: data/salmonn_embeddings or data/salmonn_embeddings_text)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of frames to encode at once (default: 16)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = VideoSalmonnConfig()

    if args.cache_dir is not None:
        cache_dir = Path(args.cache_dir)
    elif args.text_conditioned:
        cache_dir = Path("data/salmonn_embeddings_text")
    else:
        cache_dir = Path("data/salmonn_embeddings")

    repo_root = Path(__file__).resolve().parent.parent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {'text-conditioned' if args.text_conditioned else 'visual-only'}")
    print(f"Cache dir: {cache_dir}")

    # ── Load model(s) ─────────────────────────────────────────────────────────
    if args.text_conditioned:
        print("\nText-conditioned mode: loading full Qwen2-7B (~14 GB RAM required).")
        encoder, qwen2, tokenizer = load_full_model(args.model_ckpt)
        encoder.eval().to(device)
        qwen2.eval().to(device)

        # Pre-compute text embeddings once (reused for every video)
        text_ids = tokenizer(
            args.text_prompt, return_tensors="pt", add_special_tokens=True
        )["input_ids"].to(device)
        with torch.no_grad():
            text_embs = qwen2.model.embed_tokens(text_ids)  # (1, seq_len, 3584)
        print(f"  text prompt: '{args.text_prompt}' → {text_embs.shape[1]} tokens")
    else:
        encoder = load_encoder(args.model_ckpt)
        encoder.eval().to(device)
        qwen2 = None
        tokenizer = None
        text_embs = None

    # ── Load image processor (no sentencepiece needed) ────────────────────────
    from model.video_salmonn.model import _SIGLIP_CKPT
    image_processor = SiglipImageProcessor.from_pretrained(_SIGLIP_CKPT)

    # ── Collect all video paths ───────────────────────────────────────────────
    data_roots = resolve_data_roots(cfg.data_roots, repo_root=repo_root)
    label_roots = resolve_roots_for_label_maps(
        cfg.data_roots, cfg.test_data_roots, repo_root=repo_root
    )
    label2id, id2label = build_label_maps(label_roots)
    all_paths = _collect_labeled_paths(data_roots, label2id)
    print(f"Videos to process: {len(all_paths)}")

    # ── Video sampling params ─────────────────────────────────────────────────
    mean, std, resize_to = image_processor.image_mean, image_processor.image_std, (384, 384)
    clip_duration = cfg.num_frames * cfg.sample_rate / cfg.fps
    val_transform = make_val_transform(cfg.num_frames, resize_to, mean, std)

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

        video_tensor = sample["video"]  # (C, T, H, W)
        T = video_tensor.shape[1]

        # Convert frames to (T, C, H, W) uint8 numpy for SiglipImageProcessor
        import numpy as np
        frames_np = [
            (video_tensor[:, t].permute(1, 2, 0).numpy() * 255)
            .clip(0, 255)
            .astype(np.uint8)
            for t in range(T)
        ]

        # Process frames through SigLIP + mm_projector in batches
        all_frame_embeddings = []
        for start in range(0, T, args.batch_size):
            batch_frames = frames_np[start : start + args.batch_size]
            enc = image_processor(images=batch_frames, return_tensors="pt")
            pixel_values = enc["pixel_values"].to(device)
            with torch.no_grad():
                frame_embs = encoder(pixel_values)  # (B, 3584)
            all_frame_embeddings.append(frame_embs)

        # (T, 3584) — one embedding per frame
        frame_embs_all = torch.cat(all_frame_embeddings, dim=0)

        if args.text_conditioned:
            # Concatenate visual tokens (T, 3584) with pre-computed text embeddings,
            # then run through Qwen2-7B and extract last-token hidden state.
            visual_embs_3d = frame_embs_all.unsqueeze(0).to(qwen2.dtype)  # (1, T, 3584)
            combined = torch.cat([visual_embs_3d, text_embs], dim=1)      # (1, T+seq_len, 3584)
            with torch.no_grad():
                out = qwen2(
                    inputs_embeds=combined,
                    output_hidden_states=True,
                    return_dict=True,
                )
            embedding = out.hidden_states[-1][0, -1, :].cpu().to(torch.float32)
        else:
            # Visual-only: mean pool across frames → (3584,)
            embedding = frame_embs_all.mean(dim=0).cpu().to(torch.float32)

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
