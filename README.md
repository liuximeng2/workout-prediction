# workout-prediction

Video classification of gym workout exercises using **VideoMAE** (Masked Autoencoder for Video).  
The model is fine-tuned on the [Gym Workout Exercises Video](https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video) dataset (22 exercise classes, ~1 571 clips).

The repository is structured to support **multiple model architectures** — adding a new model requires only a new entry in `model/__init__.py` and a config class under `config/models/`.

---

## Repository layout

```
workout-prediction/
├── config/
│   ├── base_config.py          # Shared settings: data root, splits, seed
│   └── models/
│       └── video_mae_config.py # VideoMAE-specific hyperparameters
│
├── scripts/
│   ├── train.py                # Fine-tuning (--model selects architecture)
│   ├── evaluate.py             # Test-split accuracy + per-class breakdown
│   └── inference.py            # Single-video top-k prediction
│
├── model/
│   ├── __init__.py             # Model registry (build_model factory)
│   └── video_mae/
│       ├── __init__.py
│       └── model.py            # VideoMAE factory + video-param helpers
│
├── utils/
│   ├── __init__.py
│   ├── transforms.py           # Train / val video pre-processing pipelines
│   ├── dataset.py              # Label maps, stratified splits, LabeledVideoDataset
│   └── visualization.py        # GIF creation and inline Jupyter display
│
├── download_dataset.py         # Kaggle dataset downloader
├── data/                       # Downloaded dataset (git-ignored)
├── checkpoints/                # Saved checkpoints (git-ignored)
└── requirements.txt
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

## Environment setup

This project uses [uv](https://docs.astral.sh/uv/) to manage the virtual environment.

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) on your `PATH`
- A Kaggle account and [API credentials](https://www.kaggle.com/docs/api) (`~/.kaggle/kaggle.json`)

### Create the virtual environment

```bash
uv venv
source .venv/bin/activate   # macOS / Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
uv pip install -r requirements.txt
```

> **Apple Silicon (M-series):** PyTorchVideo is not on PyPI for arm64; install it from source:
> ```bash
> pip install "git+https://github.com/facebookresearch/pytorchvideo.git"
> ```

---

## Quickstart

### 1 — Download the dataset

```bash
python download_dataset.py
```

Videos are saved under `data/verified_data/verified_data/data_btc_10s/<class>/`.

### 2 — Fine-tune VideoMAE

```bash
python scripts/train.py
```

Override defaults via CLI flags:

```bash
python scripts/train.py \
    --model video_mae \
    --batch_size 8 \
    --num_epochs 15 \
    --learning_rate 3e-5 \
    --output_dir checkpoints/my-run
```

Model-specific defaults live in `config/models/video_mae_config.py`.  
Global settings (data path, splits, seed) live in `config/base_config.py`.

### 3 — Evaluate on the test split

```bash
python scripts/evaluate.py --checkpoint checkpoints/videomae-workout
```

Prints overall accuracy and a per-class breakdown.

### 4 — Inference on a single video

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

### `config/base_config.py` — shared across all models

| Field | Default | Description |
|---|---|---|
| `data_root` | `data/verified_data/…/data_btc_10s` | Root of the class folders |
| `train_split` | `0.70` | Fraction of videos for training |
| `val_split` | `0.15` | Fraction of videos for validation |
| `output_base` | `checkpoints` | Base directory for checkpoints |
| `seed` | `42` | Random seed for split & training |

### `config/models/video_mae_config.py` — VideoMAE-specific

| Field | Default | Description |
|---|---|---|
| `model_name` | `video_mae` | Registry key used by `model/__init__.py` |
| `model_ckpt` | `MCG-NJU/videomae-base` | Pre-trained HuggingFace checkpoint |
| `output_dir` | `checkpoints/videomae-workout` | Where to save checkpoints |
| `sample_rate` | `4` | Temporal stride when sampling frames |
| `fps` | `30` | Source video frame rate |
| `num_epochs` | `10` | Training epochs |
| `batch_size` | `4` | Per-device batch size |
| `learning_rate` | `5e-5` | AdamW learning rate |
| `warmup_ratio` | `0.1` | Fraction of steps for LR warmup |
| `push_to_hub` | `False` | Upload model to HuggingFace Hub |

### Adding a new model

1. Create `model/<name>/model.py` with a `build_model(model_ckpt, label2id, id2label)` function.
2. Register it in `model/__init__.py` under `MODEL_REGISTRY`.
3. Add `config/models/<name>_config.py` subclassing `BaseConfig`.
4. Add the config to `MODEL_CONFIGS` in `scripts/train.py`.

---

## How it works

1. **VideoMAE** pre-trains a Vision Transformer (ViT) encoder using masked video autoencoding on large video corpora (Kinetics-400).
2. We replace the decoder with a linear classification head (22 classes).
3. The pre-trained encoder weights are fine-tuned end-to-end with the HuggingFace `Trainer`.

### Video preprocessing

- **Training:** uniform temporal subsampling → pixel normalisation → random short-side scale [256, 320] → random crop → random horizontal flip.
- **Val / Test:** uniform temporal subsampling → pixel normalisation → short-side scale → centre crop.

Clip duration is computed as `num_frames × sample_rate / fps` (default ≈ 2.13 s for the base model with 16 frames).

---

## References

- [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://huggingface.co/papers/2203.12602)
- [MCG-NJU/videomae-base on HuggingFace](https://huggingface.co/MCG-NJU/videomae-base)
- [HuggingFace Video Classification guide](https://huggingface.co/docs/transformers/tasks/video_classification)
- [PyTorchVideo](https://pytorchvideo.org/)
