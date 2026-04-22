import sys
sys.stdout.reconfigure(encoding='utf-8')

"""Pre-compute and cache LLaVA-OneVision video embeddings.

By default conditions the embedding on a text prompt (existing behaviour).
With --no_text_conditioned, the chat turn contains only the video token and
no task instruction; everything else (model, forward pass, extraction) is
identical.

Usage:
    python scripts/precompute_llava_embeddings.py
    python scripts/precompute_llava_embeddings.py --no_text_conditioned
    python scripts/precompute_llava_embeddings.py --cache_dir data/llava_embeddings
    python scripts/precompute_llava_embeddings.py --model_ckpt llava-hf/llava-onevision-qwen2-0.5b-ov-hf
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
import numpy as np
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from config.models import LlavaOnevisionConfig
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
    parser = argparse.ArgumentParser(description="Pre-compute LLaVA-OneVision embeddings")
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=LlavaOnevisionConfig.model_ckpt,
        help="HuggingFace checkpoint ID or local path",
    )
    parser.add_argument(
        "--text_conditioned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include task instruction in the chat prompt (default: True). "
             "Use --no_text_conditioned for video-only embedding.",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="Classify the exercise being performed.",
        help="Text instruction (only used when --text_conditioned)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to write .pt cache files "
             "(default: data/llava_embeddings or data/llava_embeddings_notxt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = LlavaOnevisionConfig()

    if args.cache_dir is not None:
        cache_dir = Path(args.cache_dir)
    elif args.text_conditioned:
        cache_dir = Path("data/llava_embeddings")
    else:
        cache_dir = Path("data/llava_embeddings_notxt")

    repo_root = Path(__file__).resolve().parent.parent
    print(f"Mode: {'text-conditioned' if args.text_conditioned else 'no-text'}")
    print(f"Cache dir: {cache_dir}")

    # ── Load model & processor ────────────────────────────────────────────────
    print(f"Loading model: {args.model_ckpt}")
    processor = LlavaOnevisionProcessor.from_pretrained(args.model_ckpt)
    base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        ignore_mismatched_sizes=True,
    )
    base_model.eval()

    # Build chat-template prompt: with or without text instruction
    if args.text_conditioned:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": args.text_prompt},
                ],
            }
        ]
    else:
        conversation = [
            {
                "role": "user",
                "content": [{"type": "video"}],
            }
        ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # ── Collect all video paths (train + val + test) ──────────────────────────
    data_roots = resolve_data_roots(cfg.data_roots, repo_root=repo_root)
    label_roots = resolve_roots_for_label_maps(cfg.data_roots, cfg.test_data_roots, repo_root=repo_root)
    label2id, id2label = build_label_maps(label_roots)
    all_paths = _collect_labeled_paths(data_roots, label2id)
    print(f"Videos to process: {len(all_paths)}")

    # ── Video sampling params (mirror training config) ────────────────────────
    clip_duration = cfg.num_frames * cfg.sample_rate / cfg.fps

    ip = processor.image_processor
    size_cfg = ip.size
    if isinstance(size_cfg, dict):
        h = size_cfg.get("height") or size_cfg.get("shortest_edge", 224)
    else:
        h = 224
    mean, std, resize_to = ip.image_mean, ip.image_std, (h, h)

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

        # Decode center clip via existing dataset machinery
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

        # Convert frames to PIL images
        frames = [
            Image.fromarray(
                (video_tensor[:, t].permute(1, 2, 0).numpy() * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            for t in range(video_tensor.shape[1])
        ]

        enc = processor(
            text=prompt,
            videos=[frames],
            return_tensors="pt",
            padding=True,
        )
        enc = {k: v.to(base_model.device) if isinstance(v, torch.Tensor) else v
               for k, v in enc.items()}

        with torch.no_grad():
            out = base_model(
                pixel_values_videos=enc.get("pixel_values_videos"),
                image_sizes=enc.get("image_sizes"),
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

        # Pool at final token of last hidden layer → (hidden_size,)
        embedding = out.hidden_states[-1][0, -1, :].cpu().to(torch.float32)

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
