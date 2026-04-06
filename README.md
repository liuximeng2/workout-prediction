# workout-prediction

## Environment setup

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python virtual environment and dependencies.

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed and on your `PATH`
- A Kaggle account and [API credentials](https://www.kaggle.com/docs/api) if you download datasets with `kagglehub` (place `kaggle.json` in `~/.kaggle/` or set `KAGGLE_USERNAME` / `KAGGLE_KEY`)

### Create the virtual environment

From the repository root:

```bash
uv venv
```

That creates `.venv/` in this directory. Activate it:

```bash
source .venv/bin/activate   # macOS / Linux
```

On Windows (PowerShell): `.venv\Scripts\Activate.ps1`

### Install dependencies

```bash
uv pip install -r requirements.txt
```

Or install packages directly into the venv without activating:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

### Download the dataset

The gym workout exercises video dataset is fetched with `kagglehub` into `./data/` (that folder is listed in `.gitignore`).

```bash
source .venv/bin/activate
python download_dataset.py
```

The script prints the local path to the downloaded files when finished.
