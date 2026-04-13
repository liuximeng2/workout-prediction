# workout-prediction

Video classification of gym workout exercises using multiple model architectures.
Fine-tuned on the [Gym Workout Exercises Video](https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video) dataset (22 exercise classes, ~1 571 clips).

## Supported models

| Model | Type | Training script | Key idea |
|---|---|---|---|
| **VideoMAE** | Vision Transformer | `scripts/train.py --model video_mae` | Masked autoencoder pre-trained on Kinetics-400; fine-tuned end-to-end |
| **ViViT** | Vision Transformer | `scripts/train.py --model vivit` | Google's Video Vision Transformer (ViViT-B/16x2) |
| **Two-Stream CNN** | ResNet-50 dual stream | `scripts/train_two_stream.py` | Spatial (RGB) + temporal (optical flow) streams with learnable fusion |
| **Pose MLP** | MLP classifier | `scripts/train_pose.py` | Classifies engineered features from YOLO pose keypoints |

Adding a new model requires only a new entry in `model/__init__.py`, a model factory in `model/<name>/model.py`, and a config class under `config/models/`.

---

## Repository layout

```
workout-prediction/
├── config/
│   ├── base_config.py              # Shared settings: data roots, splits, seed
│   └── models/
│       ├── video_mae_config.py     # VideoMAE hyperparameters
│       ├── vivit_config.py         # ViViT hyperparameters
│       ├── two_stream_config.py    # Two-Stream CNN hyperparameters
│       └── pose_config.py          # Pose MLP hyperparameters
│
├── model/
│   ├── __init__.py                 # Model registry (build_model factory)
│   ├── video_mae/                  # VideoMAE model factory
│   ├── vivit/                      # ViViT model factory
│   ├── two_stream/                 # Two-Stream CNN (spatial + temporal)
│   └── pose/                       # Pose MLP classifier
│
├── scripts/
│   ├── train.py                    # Train VideoMAE / ViViT (HuggingFace Trainer)
│   ├── train_two_stream.py         # Train Two-Stream CNN (PyTorch loop)
│   ├── train_pose.py               # Train Pose MLP (PyTorch loop)
│   ├── eval.py                     # Evaluate any model on the test split
│   ├── inference.py                # Single-video top-k prediction (VideoMAE)
│   ├── reencode_videos.py          # Re-encode raw videos to lower res/FPS
│   ├── precompute_flow.py          # Pre-compute Farneback optical flow
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

---

## Data pipeline summary

The full pipeline from raw download to training-ready data:

```
download_dataset.py          Kaggle -> data/verified_data/
        |
reencode_videos.py           data/verified_data/ -> data/reencoded/   (256px, 15fps)
        |
        ├── precompute_flow.py   data/reencoded/ -> data/flow/        (optical flow .npy)
        └── pose_est.py          data/reencoded/ -> data/pose/        (keypoints .npy)
```

Which preprocessing steps you need depends on the model:

| Model | Needs re-encoded videos | Needs `data/flow/` | Needs `data/pose/` |
|---|---|---|---|
| VideoMAE | Yes | No | No |
| ViViT | Yes | No | No |
| Two-Stream CNN | Yes | **Yes** | No |
| Pose MLP | Yes | No | **Yes** |

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
```

Multi-clip evaluation (average predictions over multiple temporal clips per video):

```bash
python scripts/eval.py --model two_stream --checkpoint best.pt --num_clips 5
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
- `config/models/two_stream_config.py` -- Two-Stream: flow root, number of flow frames, fusion mode, SGD parameters
- `config/models/pose_config.py` -- Pose MLP: pose root, hidden dim, number of layers, dropout

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
