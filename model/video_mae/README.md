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