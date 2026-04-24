"""video-SALMONN-2 visual encoder wrapper.

Loads only the SigLIP visual encoder + mm_projector from the
tsinghua-ee/video-SALMONN-2 safetensors.  The 7B Qwen2 LLM is skipped
entirely, keeping memory usage under ~0.5 GB.

Architecture extracted from config.json:
  mm_vision_tower  : google/siglip-so400m-patch14-384
  mm_hidden_size   : 1152   (SigLIP patch feature dim)
  hidden_size      : 3584   (Qwen2-7B / projector output dim)
  mm_projector_type: mlp2x_gelu  → Linear → GELU → Linear
  mm_vision_select_layer: -2     (second-to-last SigLIP layer)
  mm_vision_select_feature: patch (all patch tokens, no CLS)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


_SIGLIP_CKPT = "google/siglip-so400m-patch14-384"
_EMBED_DIM = 3584   # mm_projector output = Qwen2-7B hidden size


class VideoSalmonnEncoder(nn.Module):
    """SigLIP visual encoder + mm_projector from video-SALMONN-2.

    forward(pixel_values) → (B, _EMBED_DIM) mean-pooled patch embeddings.
    pixel_values: (B, C, H, W) — preprocessed by SiglipImageProcessor.
    """

    def __init__(
        self,
        vision_model: nn.Module,
        mm_projector: nn.Sequential,
    ) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.mm_projector = mm_projector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        # Second-to-last layer, all patch tokens (SigLIP has no CLS token)
        features = out.hidden_states[-2]          # (B, num_patches, 1152)
        projected = self.mm_projector(features)   # (B, num_patches, 3584)
        return projected.mean(dim=1)              # (B, 3584)


class VideoSalmonnHeadOnly(nn.Module):
    """Linear classifier trained on cached video-SALMONN-2 embeddings."""

    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(_EMBED_DIM, num_labels)

    def forward(
        self,
        embedding: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        logits = self.classifier(embedding.to(self.classifier.weight.dtype))
        loss = nn.CrossEntropyLoss()(logits.float(), labels) if labels is not None else None
        return (loss, logits) if loss is not None else logits


def build_model_cached(label2id: dict) -> tuple[VideoSalmonnHeadOnly, None]:
    """Return only the linear head — no encoder loaded."""
    return VideoSalmonnHeadOnly(num_labels=len(label2id)), None


def load_encoder(model_ckpt: str) -> VideoSalmonnEncoder:
    """Load SigLIP + mm_projector weights from video-SALMONN-2 safetensors.

    Uses the already-cached HuggingFace download if present; otherwise
    triggers a fresh download (weights only, no Python files needed).
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import SiglipVisionModel

    print(f"Locating {model_ckpt} in HF cache…")
    model_dir = Path(snapshot_download(
        model_ckpt,
        ignore_patterns=["*.bin", "*.msgpack"],  # safetensors only
    ))

    print("Loading safetensors…")
    state_dict: dict[str, torch.Tensor] = {}
    for f in sorted(model_dir.glob("model-*.safetensors")):
        state_dict.update(load_file(str(f)))

    # ── Vision tower ──────────────────────────────────────────────────────────
    VT_PREFIX = "model.vision_tower.vision_tower."
    vision_sd = {
        k[len(VT_PREFIX):]: v
        for k, v in state_dict.items()
        if k.startswith(VT_PREFIX)
    }
    print(f"  vision tower keys: {len(vision_sd)}")

    vision_model = SiglipVisionModel.from_pretrained(_SIGLIP_CKPT)
    missing, unexpected = vision_model.load_state_dict(vision_sd, strict=False)
    # video-SALMONN-2 stored 26 SigLIP layers; the base checkpoint has 27 + a
    # head.  Layer 26 and the head keep their base-SigLIP values, which is fine
    # because we always read hidden_states[-2] (output of layer 25).
    # Any missing key that is NOT layer 26 or the head is a real problem.
    real_missing = [
        k for k in missing
        if not (k.startswith("vision_model.encoder.layers.26.")
                or k.startswith("vision_model.head."))
    ]
    if real_missing:
        raise RuntimeError(f"Unexpected missing vision tower keys: {real_missing[:5]}")
    print(f"  loaded {len(vision_sd)} vision tower tensors "
          f"({len(missing)} keys kept from base SigLIP)")

    # ── mm_projector ──────────────────────────────────────────────────────────
    MP_PREFIX = "model.mm_projector."
    proj_sd = {
        k[len(MP_PREFIX):]: v
        for k, v in state_dict.items()
        if k.startswith(MP_PREFIX)
    }
    print(f"  mm_projector keys: {list(proj_sd.keys())}")

    w0 = proj_sd["0.weight"]   # (intermediate, 1152)
    w2 = proj_sd["2.weight"]   # (3584, intermediate)
    mm_projector = nn.Sequential(
        nn.Linear(w0.shape[1], w0.shape[0]),
        nn.GELU(),
        nn.Linear(w2.shape[1], w2.shape[0]),
    )
    mm_projector.load_state_dict(proj_sd, strict=True)
    print(f"  mm_projector: {w0.shape[1]} → {w0.shape[0]} → {w2.shape[0]}")

    return VideoSalmonnEncoder(vision_model, mm_projector)


def load_full_model(model_ckpt: str):
    """Load SigLIP + mm_projector + Qwen2-7B for text-conditioned embedding.

    Returns (VideoSalmonnEncoder, Qwen2ForCausalLM, tokenizer).
    Requires ~14 GB RAM in bfloat16.

    Uses two-pass selective loading via safe_open to avoid materialising the
    full state dict (~14.6 GB) and the float32 model (~28 GB) simultaneously.
    Peak RAM is ~14.5 GB instead of ~42 GB.
    """
    import json
    from huggingface_hub import snapshot_download
    from safetensors.torch import safe_open
    from transformers import SiglipVisionModel, Qwen2ForCausalLM, Qwen2Config, AutoTokenizer

    VT_PREFIX = "model.vision_tower.vision_tower."
    MP_PREFIX = "model.mm_projector."
    LM_PREFIX = "model.language_model."

    print(f"Locating {model_ckpt} in HF cache…")
    model_dir = Path(snapshot_download(
        model_ckpt,
        ignore_patterns=["*.bin", "*.msgpack"],
    ))
    shards = sorted(model_dir.glob("model-*.safetensors"))

    # ── Pass 1: extract only vision tower + mm_projector keys ────────────────
    print("Pass 1: loading vision tower + mm_projector…")
    vision_sd: dict[str, torch.Tensor] = {}
    proj_sd:   dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if key.startswith(VT_PREFIX):
                    vision_sd[key[len(VT_PREFIX):]] = sf.get_tensor(key)
                elif key.startswith(MP_PREFIX):
                    proj_sd[key[len(MP_PREFIX):]] = sf.get_tensor(key)

    vision_model = SiglipVisionModel.from_pretrained(_SIGLIP_CKPT)
    missing, _ = vision_model.load_state_dict(vision_sd, strict=False)
    real_missing = [
        k for k in missing
        if not (k.startswith("vision_model.encoder.layers.26.")
                or k.startswith("vision_model.head."))
    ]
    if real_missing:
        raise RuntimeError(f"Unexpected missing vision tower keys: {real_missing[:5]}")
    print(f"  vision tower: {len(vision_sd)} tensors loaded")
    del vision_sd

    w0 = proj_sd["0.weight"]
    w2 = proj_sd["2.weight"]
    mm_projector = nn.Sequential(
        nn.Linear(w0.shape[1], w0.shape[0]),
        nn.GELU(),
        nn.Linear(w2.shape[1], w2.shape[0]),
    )
    mm_projector.load_state_dict(proj_sd, strict=True)
    del proj_sd
    encoder = VideoSalmonnEncoder(vision_model, mm_projector)

    # ── Qwen2-7B config ───────────────────────────────────────────────────────
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        raw_cfg = json.load(f)

    llm_cfg_dict = raw_cfg.get("text_config") or raw_cfg
    qwen2_cfg = Qwen2Config(
        hidden_size=llm_cfg_dict.get("hidden_size", 3584),
        num_hidden_layers=llm_cfg_dict.get("num_hidden_layers", 28),
        num_attention_heads=llm_cfg_dict.get("num_attention_heads", 28),
        num_key_value_heads=llm_cfg_dict.get("num_key_value_heads", 4),
        intermediate_size=llm_cfg_dict.get("intermediate_size", 18944),
        max_position_embeddings=llm_cfg_dict.get("max_position_embeddings", 32768),
        vocab_size=llm_cfg_dict.get("vocab_size", 152064),
        rope_theta=llm_cfg_dict.get("rope_theta", 1000000.0),
    )

    # ── Pass 2: extract only LM keys, init model on meta device ──────────────
    # Initialising on meta allocates no RAM; assign=True transplants tensors
    # directly without a second copy, so peak is just lm_sd (~14 GB).
    print("Pass 2: loading Qwen2-7B language model (~14 GB)…")
    lm_sd: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if key.startswith(LM_PREFIX):
                    lm_sd[key[len(LM_PREFIX):]] = sf.get_tensor(key).to(torch.bfloat16)

    with torch.device("meta"):
        qwen2 = Qwen2ForCausalLM(qwen2_cfg)
    missing_lm, unexpected_lm = qwen2.load_state_dict(lm_sd, strict=False, assign=True)
    del lm_sd
    print(f"  Qwen2 LM: loaded ({len(missing_lm)} missing, {len(unexpected_lm)} unexpected)")

    # Materialize any remaining meta tensors (tied weights / buffers absent from
    # the checkpoint, e.g. lm_head.weight tied to embed_tokens, rotary inv_freq).
    head_dim = qwen2_cfg.hidden_size // qwen2_cfg.num_attention_heads
    for module in qwen2.modules():
        for param_name, param in list(module.named_parameters(recurse=False)):
            if param.is_meta:
                setattr(module, param_name,
                        nn.Parameter(torch.empty(param.shape, dtype=torch.bfloat16)))
        for buf_name, buf in list(module.named_buffers(recurse=False)):
            if buf.is_meta:
                if buf_name == "inv_freq":
                    # torch.empty → garbage → cos(inf)=NaN in rotary attention.
                    # Recompute from rope_theta, matching Qwen2RotaryEmbedding.__init__.
                    dim = buf.shape[0] * 2
                    rope_theta = float(llm_cfg_dict.get("rope_theta", 1_000_000.0))
                    inv_freq = 1.0 / (rope_theta ** (
                        torch.arange(0, dim, 2, dtype=torch.float32) / dim
                    ))
                    module.register_buffer(buf_name, inv_freq, persistent=False)
                else:
                    module.register_buffer(
                        buf_name,
                        torch.zeros(buf.shape,
                                    dtype=torch.bfloat16 if buf.is_floating_point()
                                    else buf.dtype))

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  tokenizer loaded")

    return encoder, qwen2, tokenizer


# ── Stubs required by train.py dispatcher ────────────────────────────────────

def build_model(
    model_ckpt: str,
    label2id: dict,
    id2label: dict,
) -> tuple[VideoSalmonnHeadOnly, None]:
    """train.py calls build_model() via MODEL_REGISTRY; always return head-only.

    The encoder is never trained — call load_encoder() in the precompute
    script instead.
    """
    return build_model_cached(label2id)


def apply_freeze_strategy(model: VideoSalmonnHeadOnly, strategy: str) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Freeze strategy: '{strategy}' — "
        f"trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.1f}%)"
    )


def get_video_params(image_processor) -> tuple:
    """Extract mean/std/size from SiglipImageProcessor."""
    mean = image_processor.image_mean
    std  = image_processor.image_std
    size_cfg = image_processor.size
    if isinstance(size_cfg, dict):
        h = size_cfg.get("height") or size_cfg.get("shortest_edge", 384)
    else:
        h = 384
    return mean, std, (h, h)
