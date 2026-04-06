"""Download the gym workout exercises video dataset into ./data."""

from pathlib import Path

import kagglehub

DATASET = "philosopher0808/gym-workoutexercises-video"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download(DATASET, output_dir=str(OUTPUT_DIR))
    print("Path to dataset files:", path)


if __name__ == "__main__":
    main()
