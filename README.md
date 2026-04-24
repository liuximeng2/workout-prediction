# workout-prediction

Video classification of gym workout exercises using multiple model architectures.
Fine-tuned on the [Gym Workout Exercises Video](https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video) dataset (22 exercise classes, ~1 571 clips).

## Supported models

| Model | Type | Training script | Key idea |
|---|---|---|---|
| **VideoMAE** | Vision Transformer | `scripts/train.py --model video_mae` | Masked autoencoder pre-trained on Kinetics-400; fine-tuned end-to-end |
| **ViViT** | Vision Transformer | `scripts/train.py --model vivit` | Google's Video Vision Transformer (ViViT-B/16x2) |
| **LLaVA-OneVision** | Multimodal VLM + linear head | `scripts/train.py --model llava_onevision` | LLaVA-OneVision (Qwen2-0.5B); optional task prompt; full fine-tune or cached-embedding head training |
| **video-SALMONN-2** | SigLIP visual encoder + head | `scripts/train.py --model video_salmonn` | Visual encoder + projector from video-SALMONN-2; linear head on frozen embeddings (7B LLM not loaded) |
| **VideoPrism** | ViT-B video encoder + head | `scripts/train.py --model videoprism` | Google VideoPrism base encoder; linear head on frozen JAX-precomputed embeddings |
| **CNN Fusion** | ResNet-50 per-frame + sum pool | `scripts/train_cnn_fusion.py` | Shared CNN backbone over sampled frames, temporally summed, linear head |
| **Two-Stream CNN** | ResNet-50 dual stream | `scripts/train_two_stream.py` | Spatial (RGB) + temporal (optical flow) streams with learnable fusion |
| **Pose MLP** | MLP classifier | `scripts/train_pose.py` | Classifies engineered features from YOLO pose keypoints |
| **Qwen3-VL** | Vision-language (Ollama) | *(no training script)* | Zero-shot classification via `scripts/eval_qwen3_vl.py` and a local Ollama model |

Adding a new model requires a new entry in `model/__init__.py`, a model factory in `model/<name>/model.py`, and a config class under `config/models/` (plus wiring in `scripts/train.py` / `scripts/eval.py` when you add training or standard eval).

---

## Repository layout

```
workout-prediction/
├── config/
│   ├── base_config.py              # Shared settings: data roots, splits, seed
│   └── models/
│       ├── video_mae_config.py     # VideoMAE hyperparameters
│       ├── vivit_config.py         # ViViT hyperparameters
│       ├── llava_onevision_config.py
│       ├── video_salmonn_config.py
│       ├── videoprism_config.py
│       ├── cnn_fusion_config.py    # CNN Fusion hyperparameters
│       ├── two_stream_config.py    # Two-Stream CNN hyperparameters
│       ├── pose_config.py          # Pose MLP hyperparameters
│       └── qwen3_vl_config.py      # Qwen3-VL / Ollama eval settings
│
├── model/
│   ├── __init__.py                 # Model registry (build_model factory)
│   ├── video_mae/                  # VideoMAE model factory
│   ├── vivit/                      # ViViT model factory
│   ├── llava_onevision/            # LLaVA-OneVision + classification head
│   ├── video_salmonn/              # video-SALMONN-2 visual stack + head
│   ├── videoprism/                 # VideoPrism encoder + head
│   ├── cnn_fusion/                 # CNN Fusion
│   ├── two_stream/                 # Two-Stream CNN (spatial + temporal)
│   ├── pose/                       # Pose MLP classifier
│   └── qwen3_vl/                   # Ollama client (eval-only helper)
│
├── scripts/
│   ├── train.py                    # Train VideoMAE, ViViT, LLaVA, SALMONN, VideoPrism (HF Trainer)
│   ├── train_cnn_fusion.py         # Train CNN Fusion (PyTorch loop)
│   ├── train_two_stream.py         # Train Two-Stream CNN (PyTorch loop)
│   ├── train_pose.py               # Train Pose MLP (PyTorch loop)
│   ├── eval.py                     # Evaluate checkpoints on train/val/test
│   ├── eval_qwen3_vl.py            # Zero-shot eval via Ollama (Qwen3-VL)
│   ├── inference.py                # Single-video top-k prediction (VideoMAE)
│   ├── reencode_videos.py          # Re-encode raw videos to lower res/FPS
│   ├── precompute_flow.py          # Pre-compute Farneback optical flow
│   ├── precompute_llava_embeddings.py
│   ├── precompute_salmonn_embeddings.py
│   ├── precompute_videoprism_embeddings.py
│   ├── pose_est.py                 # Pre-compute YOLO pose keypoints
│   └── visualize_flow.py           # Visualize optical flow for a video
│
├── utils/
│   ├── dataset.py                  # Label maps, stratified splits, video dataset
│   ├── flow_dataset.py             # Two-stream dataset (RGB + flow)
│   ├── pose_dataset.py             # Pose feature dataset
│   ├── transforms.py               # Train / val video pre-processing pipelines
│   └── visualization.py            # GIF creation and Jupyter display helpers
│
├── analysis/                       # Notebooks and scripts for EDA & evaluation
│   ├── eda_dataset.ipynb
│   ├── analyze_videomae.ipynb
│   ├── eval_models.ipynb
│   └── visualize_pose.ipynb
│
├── download_dataset.py             # Kaggle dataset downloader
├── data/                           # Downloaded & processed data (git-ignored)
├── checkpoints/                    # Saved model checkpoints
├── requirements.txt
└── uv.toml
```

---

## Exercise classes

| | | |
|---|---|---|
| barbell biceps curl | bench press | chest fly machine |
| deadlift | decline bench press | hammer curl |
| hip thrust | incline bench press | lat pulldown |
| lateral raise | leg extension | leg raises |
| plank | pull Up | push-up |
| romanian deadlift | russian twist | shoulder press |
| squat | t bar row | tricep Pushdown |
| tricep dips | | |

---

## Getting started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) on your `PATH`
- [ffmpeg](https://ffmpeg.org/) on your `PATH` (for video re-encoding)
- A Kaggle account and [API credentials](https://www.kaggle.com/docs/api) (`~/.kaggle/kaggle.json`)
- (Optional) NVIDIA GPU with CUDA 12.8 for accelerated training
- (Optional) [Ollama](https://ollama.com/) for Qwen3-VL zero-shot evaluation (`scripts/eval_qwen3_vl.py`)

### 1 -- Create the virtual environment and install dependencies

```bash
uv venv
source .venv/bin/activate   # macOS / Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1

uv pip install -r requirements.txt
```

> **Apple Silicon (M-series):** PyTorchVideo is not on PyPI for arm64; install it from source:
> ```bash
> pip install "git+https://github.com/facebookresearch/pytorchvideo.git"
> ```

### 2 -- Download the dataset

```bash
python download_dataset.py
```

This downloads the Kaggle dataset into `data/`. Raw videos live under:
- `data/verified_data/verified_data/data_btc_10s/<class>/`
- `data/verified_data/verified_data/data_crawl_10s/<class>/`

### 3 -- Re-encode videos (recommended)

The raw dataset contains high-resolution videos with varying codecs. Re-encoding to a uniform resolution and frame rate speeds up all downstream processing (flow computation, training, etc.) and avoids codec-related errors.

```bash
python scripts/reencode_videos.py
```

This creates a `data/reencoded/` tree mirroring the original structure at 256px short-side and 15 fps. The default config (`config/base_config.py`) already points `data_roots` to the re-encoded paths.

Options:

```bash
# Custom resolution / FPS / quality
python scripts/reencode_videos.py --size 224 --fps 24 --crf 20

# Use 8 parallel workers (default)
python scripts/reencode_videos.py --workers 8

# Re-encode specific directories
python scripts/reencode_videos.py \
    --input_dirs data/verified_data/verified_data/data_btc_10s \
                 data/verified_data/verified_data/data_crawl_10s \
                 data/test/test \
    --output_root data/reencoded

# Retry videos that failed in a previous run
python scripts/reencode_videos.py --retry-failed
```

### 4 -- Pre-compute optical flow (required for Two-Stream CNN)

The Two-Stream CNN needs pre-computed Farneback optical flow stored on disk. Run this before training the two-stream model:

```bash
python scripts/precompute_flow.py
```

This processes every video from `data_roots`, computes dense optical flow between consecutive frames, and saves uint8-encoded `.npy` files under `data/flow/`.

Storage layout:
```
data/flow/<data_root_stem>/<class>/<video_stem>/
    frame_0000.npy   # flow from frame 0 -> frame 1, shape (2, H, W), uint8
    frame_0001.npy   # flow from frame 1 -> frame 2
    ...
    meta.npy         # [n_flow_frames, H, W, flow_min, flow_max]
```

Options:

```bash
# Process specific directories
python scripts/precompute_flow.py \
    --input_dirs data/reencoded/data_btc_10s data/reencoded/data_crawl_10s data/reencoded/test

# Resize frames before computing flow (default: 320px short-side)
python scripts/precompute_flow.py --max_short_side 320

# Use parallel workers
python scripts/precompute_flow.py --workers 4

# Dry run: print plan without computing
python scripts/precompute_flow.py --dry_run
```

### 5 -- Pre-compute pose keypoints (required for Pose MLP)

The Pose MLP classifier uses engineered features derived from YOLO pose estimation. Run this before training the pose model:

```bash
python scripts/pose_est.py
```

This runs YOLO26n-pose on every Nth frame of each video, extracts the dominant person's 17 COCO keypoints, and saves per-video `.npy` files under `data/pose/`.

Storage layout:
```
data/pose/<data_root_stem>/<class>/<video_stem>.npy
    # shape (T, 17, 3) -- float32
    # channels: (x_normalized, y_normalized, confidence)
```

Options:

```bash
# Use a different YOLO model
python scripts/pose_est.py --yolo_model yolo26l-pose.pt

# Process every frame (default: every 2nd)
python scripts/pose_est.py --sample_every 1

# Process specific directories
python scripts/pose_est.py \
    --input_dirs data/reencoded/data_btc_10s data/reencoded/data_crawl_10s
```

### 6 -- Pre-compute vision embeddings (required for SALMONN / VideoPrism; optional for LLaVA)

**video-SALMONN-2** and **VideoPrism** training in `scripts/train.py` expects one vector per training video under fixed directory names (see table above). Run the matching script after re-encoding:

```bash
python scripts/precompute_salmonn_embeddings.py
python scripts/precompute_videoprism_embeddings.py
```

For **text-conditioned** SALMONN embeddings (full Qwen2 forward during pre-compute), use `python scripts/precompute_salmonn_embeddings.py --text_conditioned` and train with matching settings.

**LLaVA-OneVision** can be fine-tuned from decoded video alone; if you first populate an embedding cache, training automatically switches to a lightweight head on disk:

```bash
python scripts/precompute_llava_embeddings.py
# text-free cache for ablations:
python scripts/precompute_llava_embeddings.py --no_text_conditioned
```

Use each script's `--help` for directory overrides, workers, and GPU options.

---

## Model comparison methodology

All models are compared under a **fixed-data, model-native-input** protocol:
the source videos and train/val/test split are identical for every model, but
each model consumes those videos the way its architecture prescribes. Short-
clip transformers don't pretend to see the whole video; the pose MLP doesn't
pretend to throw away 60 % of the signal it can cheaply process.

### What is held constant across all models

- **Source data** -- the same re-encoded videos (256 px short-side, 15 fps, H.264 CRF 23).
- **Train / val / test split** -- identical partition **by video path**, produced once by `utils/splits.canonical_splits` with `seed=42`. Models that need auxiliary data (flow, pose) drop unusable videos *within* their split, but never shift videos between splits.
- **Labels** -- the same `label2id` across every model.
- **Evaluation protocol** -- same per-video aggregation where the code path supports it (multi-clip mean-of-logits for pixel / flow clip models; one forward for pose; one embedding forward for cached LLaVA / SALMONN / VideoPrism heads). Qwen3-VL uses a separate zero-shot script and optional multi-clip voting there.

### What is expected to differ (and is reported, not hidden)

Each model's "input budget" -- how much of each 10 s video it actually sees -- is a property of the architecture, not a bug. The table below shows, at a glance, what slice of the source video each model consumes:

| Model | Input format | Clip length | Frames / features used | % of 10 s video per clip |
|---|---|---|---|---|
| **VideoMAE** | RGB clip | ~2.13 s | 16 frames @ stride 4 | ~21 % |
| **ViViT** | RGB clip | ~4.27 s | 32 frames @ stride 2 | ~43 % |
| **LLaVA-OneVision** | RGB frame stack (+ optional text) | ~1.07 s | 8 frames @ stride 4 @ 30 fps nominal | ~11 % |
| **video-SALMONN-2** | RGB frame stack → SigLIP tokens | ~1.07 s | 8 frames @ stride 4 @ 30 fps nominal | ~11 % |
| **VideoPrism** | RGB clip → ViT patch tokens | ~2.13 s | 16 frames @ stride 4 @ 30 fps nominal | ~21 % |
| **CNN Fusion** | RGB clip | 4.0 s | 8 frames (uniformly sampled) | 40 % |
| **Two-Stream CNN** | 1 RGB frame + 20 stacked flow frames | 4.0 s | 1 + 20 | 40 % |
| **Pose MLP** | Engineered features over full keypoint sequence | **10 s (full video)** | 144-D feature vector | **100 %** |
| **Qwen3-VL** | Uniform JPEG frames over full video | **10 s (full video)** | 16 frames default (configurable) | **100 %** span (sparse) |

Reading the table:

- **VideoMAE** sees a ~2 s window at a time for each forward pass (see table).
- **ViViT** sees ~4 s per forward pass -- the longest RGB window among VideoMAE and ViViT in this repo.
- **LLaVA-OneVision**, **video-SALMONN-2**, and **VideoPrism** use the same style of temporal window as in `train.py` (`clip_duration = num_frames × sample_rate / fps` with the defaults in each config). SALMONN and VideoPrism are typically trained on **one frozen embedding vector per video** produced with that window; LLaVA can run the full VLM on clips or switch to cached embeddings when `data/llava_embeddings*` is present.
- **CNN Fusion** and **Two-Stream** both span 4 s but consume that window very differently (8 RGB frames vs. 1 RGB + a stack of flow fields).
- **Pose MLP** consumes the entire 10 s video because 17-keypoint-per-frame sequences are cheap to aggregate -- throttling it to 2 s would be an artificial handicap that doesn't match how you would actually deploy it.
- **Qwen3-VL** is a **zero-shot** baseline: frames are spread across the whole clip (default 16), not a sliding short window trained end-to-end on this dataset.

### Closing the "how much does it see" gap at eval time

Clip-based RGB / flow models can be evaluated with `--num_clips N`: each video is sliced into *N* evenly-spaced clips, logits are averaged, then argmax. This lets a short-window model like VideoMAE cover most of a 10 s video at inference. Use the **same `num_clips`** across those models for a fair report (e.g., `--num_clips 5`). The pose MLP ignores this flag -- it already ingests the whole video in one pass. **Cached-embedding heads** (LLaVA / SALMONN / VideoPrism in `eval.py`) score **one embedding per video** unless you change the pre-computation recipe; they do not share the same multi-clip flag path as pixel models.

### What to report alongside accuracy

Accuracy alone hides the input-budget asymmetry. When comparing models, report **three** numbers side-by-side:

1. **Test accuracy** (top-1 on the canonical test split).
2. **Input budget** (clip length, frames / features per forward pass, % of video seen per clip, `num_clips` at eval time).
3. **Compute** (parameter count, GFLOPs or wall-clock inference time per video).

This makes claims like *"the pose MLP matches VideoMAE at 0.1 % of the compute"* or *"VideoMAE wins in accuracy but with 5x the per-video inference cost"* legible from a single table.

### (Optional) Ablation to isolate architecture from input budget

If you want to separate *architecture quality* from *how much of the video the model is allowed to see*, run one extra configuration where every model is constrained to the same 4 s window (e.g., force VideoMAE to a single 4 s clip, force the pose MLP to a 4 s pose window). Accuracy drops in that ablation can be attributed to input-budget reductions rather than a worse architecture. Treat this as a supplementary table, not the headline number.

---

## Data pipeline summary

The full pipeline from raw download to training-ready data:

```
download_dataset.py          Kaggle -> data/verified_data/
        |
reencode_videos.py           data/verified_data/ -> data/reencoded/   (256px, 15fps)
        |
        ├── precompute_flow.py           data/reencoded/ -> data/flow/
        ├── pose_est.py                  data/reencoded/ -> data/pose/
        ├── precompute_llava_embeddings.py   -> data/llava_embeddings/ (or ..._notxt)
        ├── precompute_salmonn_embeddings.py -> data/salmonn_embeddings/ (or ..._text)
        └── precompute_videoprism_embeddings.py -> data/videoprism_embeddings/
```

Which preprocessing steps you need depends on the model:

| Model | Needs re-encoded videos | Needs `data/flow/` | Needs `data/pose/` | Embedding cache |
|---|---|---|---|---|
| VideoMAE | Yes | No | No | No |
| ViViT | Yes | No | No | No |
| LLaVA-OneVision | Yes | No | No | Optional — if `data/llava_embeddings/` or `data/llava_embeddings_notxt/` exists, training uses cached embeddings + small head; otherwise the full model is fine-tuned from video |
| video-SALMONN-2 | Yes | No | No | **Required** — `data/salmonn_embeddings/` (or `data/salmonn_embeddings_text/` with `--text_conditioned`) |
| VideoPrism | Yes | No | No | **Required** — `data/videoprism_embeddings/` |
| CNN Fusion | Yes | No | No | No |
| Two-Stream CNN | Yes | **Yes** | No | No |
| Pose MLP | Yes | No | **Yes** | No |
| Qwen3-VL | Yes | No | No | No (Ollama at inference) |

---

## Training

### VideoMAE

```bash
python scripts/train.py --model video_mae
```

### ViViT

```bash
python scripts/train.py --model vivit
```

### LLaVA-OneVision

```bash
python scripts/train.py --model llava_onevision
# optional: --no_text_conditioned / --text_conditioned to match your embedding cache
```

### video-SALMONN-2

```bash
python scripts/precompute_salmonn_embeddings.py
python scripts/train.py --model video_salmonn
```

### VideoPrism

```bash
python scripts/precompute_videoprism_embeddings.py
python scripts/train.py --model videoprism
```

### CNN Fusion

```bash
python scripts/train_cnn_fusion.py
```

### Two-Stream CNN

```bash
python scripts/train_two_stream.py
```

### Pose MLP

```bash
python scripts/train_pose.py
```

---

## Evaluation

Evaluate any model on the held-out test split:

```bash
# VideoMAE
python scripts/eval.py --model video_mae --checkpoint checkpoints/videomae

# ViViT
python scripts/eval.py --model vivit --checkpoint checkpoints/vivit

# Two-Stream CNN
python scripts/eval.py --model two_stream --checkpoint checkpoints/two-stream/best.pt

# Pose MLP
python scripts/eval.py --model pose --checkpoint checkpoints/pose-workout/best.pt

# CNN Fusion
python scripts/eval.py --model cnn_fusion --checkpoint checkpoints/cnn-fusion/best.pt

# Cached-embedding heads (paths follow each config's output_dir; default filenames shown)
python scripts/eval.py --model llava_onevision --checkpoint checkpoints/llava-onevision-workout/model.safetensors
python scripts/eval.py --model video_salmonn --checkpoint checkpoints/video-salmonn/model.safetensors
python scripts/eval.py --model videoprism --checkpoint checkpoints/videoprism/model.safetensors
```

Multi-clip evaluation (average predictions over multiple temporal clips per video) applies to **pixel / flow** models, for example:

```bash
python scripts/eval.py --model two_stream --checkpoint best.pt --num_clips 5
```

### Qwen3-VL (zero-shot, Ollama)

Requires [Ollama](https://ollama.com/) running locally with a Qwen3-VL image pulled (see `config/models/qwen3_vl_config.py` for defaults):

```bash
python scripts/eval_qwen3_vl.py
python scripts/eval_qwen3_vl.py --num_frames 16 --num_clips 3
```

## Inference

Run a single-video prediction with VideoMAE:

```bash
python scripts/inference.py --video path/to/squat.mp4 --top_k 5
```

Example output:

```
Predictions for: path/to/squat.mp4
Rank  Label                              Score
---------------------------------------------------
1     squat                             0.9421
2     romanian deadlift                 0.0213
3     deadlift                          0.0148
4     leg extension                     0.0089
5     hip thrust                        0.0062
```

---

## Configuration reference

### `config/base_config.py` -- shared across all models

| Field | Default | Description |
|---|---|---|
| `data_roots` | `data/reencoded/data_btc_10s`, `data/reencoded/data_crawl_10s` | Class-folder roots (videos merged for training) |
| `test_data_roots` | `data/reencoded/test` | Hold-out test videos (same layout as data_roots) |
| `train_split` | `0.70` | Fraction of videos for training |
| `val_split` | `0.15` | Fraction of videos for validation (test = 1 - train - val) |
| `output_base` | `checkpoints` | Base directory for checkpoints |
| `seed` | `42` | Random seed for splits and training |

### Model-specific configs

Each model config inherits from `BaseConfig` and adds its own hyperparameters:

- `config/models/video_mae_config.py` -- VideoMAE: pre-trained checkpoint, sample rate, freeze strategy
- `config/models/vivit_config.py` -- ViViT: pre-trained checkpoint, sample rate, freeze strategy
- `config/models/llava_onevision_config.py` -- LLaVA: frames, sample rate, text prompt / conditioning
- `config/models/video_salmonn_config.py` -- video-SALMONN-2: embedding layout, head training hyperparameters
- `config/models/videoprism_config.py` -- VideoPrism: embedding layout, head training hyperparameters
- `config/models/cnn_fusion_config.py` -- CNN Fusion: backbone, clip length, optimizer settings
- `config/models/two_stream_config.py` -- Two-Stream: flow root, number of flow frames, fusion mode, SGD parameters
- `config/models/pose_config.py` -- Pose MLP: pose root, hidden dim, number of layers, dropout
- `config/models/qwen3_vl_config.py` -- Qwen3-VL: Ollama URL, frame count, multi-clip voting

### Adding a new model

1. Create `model/<name>/model.py` with a `build_model(...)` function that returns `(model, processor)`.
2. Register it in `model/__init__.py` under `MODEL_REGISTRY`.
3. Add `config/models/<name>_config.py` subclassing `BaseConfig`.
4. Add the config to `MODEL_CONFIGS` in the appropriate training script.

---

## References

- [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://huggingface.co/papers/2203.12602)
- [MCG-NJU/videomae-base on HuggingFace](https://huggingface.co/MCG-NJU/videomae-base)
- [ViViT: A Video Vision Transformer](https://huggingface.co/papers/2103.15691)
- [Two-Stream Convolutional Networks for Action Recognition in Videos](https://arxiv.org/abs/1406.2199)
- [HuggingFace Video Classification guide](https://huggingface.co/docs/transformers/tasks/video_classification)
- [PyTorchVideo](https://pytorchvideo.org/)
- [Ultralytics YOLO Pose](https://docs.ultralytics.com/tasks/pose/)
- [LLaVA-OneVision (Hugging Face)](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
- [video-SALMONN-2](https://huggingface.co/tsinghua-ee/video-SALMONN-2)
- [VideoPrism](https://huggingface.co/google/videoprism-base-f16r288)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) · [Ollama](https://ollama.com/)
