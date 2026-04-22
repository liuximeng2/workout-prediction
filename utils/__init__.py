"""Utility modules for the workout VideoMAE classifier."""

from utils.dataset import (
    build_datasets,
    build_label_maps,
    filter_valid_flow,
    flow_dir_for_video,
    resolve_data_roots,
    resolve_roots_for_label_maps,
)
from utils.flow_dataset import TwoStreamDataset
from utils.splits import canonical_splits, filter_split
from utils.transforms import make_train_transform, make_val_transform

__all__ = [
    "build_datasets",
    "build_label_maps",
    "canonical_splits",
    "filter_split",
    "filter_valid_flow",
    "flow_dir_for_video",
    "resolve_data_roots",
    "resolve_roots_for_label_maps",
    "TwoStreamDataset",
    "make_train_transform",
    "make_val_transform",
    "display_gif",
    "save_gif",
]


def __getattr__(name: str):
    # Import visualization lazily: it pulls in OpenCV, whose wheel bundles libavdevice
    # and triggers duplicate ObjC class warnings (and possible crashes) alongside PyAV.
    if name == "display_gif":
        from utils.visualization import display_gif as _display_gif

        return _display_gif
    if name == "save_gif":
        from utils.visualization import save_gif as _save_gif

        return _save_gif
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
