"""Pose feature dataset: loads precomputed keypoints and engineers temporal features.

Each video's keypoints are stored as a (T, 17, 3) .npy file where the 3 channels
are (x_norm, y_norm, confidence).  This module computes a fixed-length feature
vector per video from joint angles, velocities, and periodicity.

COCO 17 keypoint indices:
    0: nose          1: left_eye       2: right_eye      3: left_ear
    4: right_ear     5: left_shoulder  6: right_shoulder  7: left_elbow
    8: right_elbow   9: left_wrist    10: right_wrist    11: left_hip
   12: right_hip    13: left_knee     14: right_knee     15: left_ankle
   16: right_ankle
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Joint angle definitions: (point_a, vertex, point_c) — angle at the vertex
ANGLE_DEFS = {
    "left_elbow":    (5, 7, 9),     # shoulder → elbow → wrist
    "right_elbow":   (6, 8, 10),
    "left_shoulder": (11, 5, 7),    # hip → shoulder → elbow
    "right_shoulder": (12, 6, 8),
    "left_knee":     (11, 13, 15),  # hip → knee → ankle
    "right_knee":    (12, 14, 16),
    "left_hip":      (5, 11, 13),   # shoulder → hip → knee
    "right_hip":     (6, 12, 14),
}

# Pairs for inter-joint distances (meaningful for exercise discrimination)
DISTANCE_PAIRS = [
    (9, 10),   # wrist to wrist
    (15, 16),  # ankle to ankle
    (9, 15),   # left wrist to left ankle
    (10, 16),  # right wrist to right ankle
    (0, 11),   # nose to left hip  (torso bend indicator)
    (0, 12),   # nose to right hip
]


def _angle_between(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Compute angle (in degrees) at the vertex for each frame.

    Args:
        a, vertex, c: Arrays of shape (T, 2).

    Returns:
        Angle array of shape (T,).
    """
    v1 = a - vertex
    v2 = c - vertex
    cos = np.sum(v1 * v2, axis=1) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
    )
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _temporal_stats(series: np.ndarray) -> np.ndarray:
    """Compute summary statistics for a 1D temporal signal.

    Returns a fixed-length vector:
        [mean, std, min, max, range, velocity_mean, velocity_std, velocity_max, fft_peak_freq]
    """
    if len(series) < 2:
        return np.zeros(9, dtype=np.float32)

    mean = np.mean(series)
    std = np.std(series)
    smin = np.min(series)
    smax = np.max(series)
    srange = smax - smin

    # Velocity (first difference)
    vel = np.diff(series)
    vel_mean = np.mean(np.abs(vel))
    vel_std = np.std(vel)
    vel_max = np.max(np.abs(vel))

    # Dominant frequency via FFT (captures repetition cadence)
    if len(series) >= 4:
        fft_mag = np.abs(np.fft.rfft(series - mean))
        # Skip DC component (index 0)
        fft_mag[0] = 0
        peak_freq = np.argmax(fft_mag) / len(series)
    else:
        peak_freq = 0.0

    return np.array(
        [mean, std, smin, smax, srange, vel_mean, vel_std, vel_max, peak_freq],
        dtype=np.float32,
    )


def extract_features(keypoints: np.ndarray) -> np.ndarray:
    """Extract a fixed-length feature vector from (T, 17, 3) keypoints.

    Feature groups:
        1. Joint angle temporal stats (8 angles × 9 stats = 72)
        2. Inter-joint distance temporal stats (6 pairs × 9 stats = 54)
        3. Torso orientation stats (1 signal × 9 stats = 9)
        4. Overall body velocity stats (1 signal × 9 stats = 9)

    Total: 144 features.
    """
    T = keypoints.shape[0]
    xy = keypoints[:, :, :2]   # (T, 17, 2)
    conf = keypoints[:, :, 2]  # (T, 17)

    features = []

    # 1. Joint angles over time
    for _name, (a_idx, v_idx, c_idx) in ANGLE_DEFS.items():
        angles = _angle_between(xy[:, a_idx], xy[:, v_idx], xy[:, c_idx])
        # Weight by minimum confidence of the three joints
        min_conf = np.minimum(np.minimum(conf[:, a_idx], conf[:, v_idx]), conf[:, c_idx])
        # Zero out low-confidence frames
        angles = np.where(min_conf > 0.3, angles, np.nan)
        # Replace NaN with median (robust to missing detections)
        if np.any(~np.isnan(angles)):
            angles = np.where(np.isnan(angles), np.nanmedian(angles), angles)
        else:
            angles = np.zeros(T, dtype=np.float32)
        features.append(_temporal_stats(angles))

    # 2. Inter-joint distances over time
    for j1, j2 in DISTANCE_PAIRS:
        dist = np.linalg.norm(xy[:, j1] - xy[:, j2], axis=1)
        features.append(_temporal_stats(dist))

    # 3. Torso orientation (angle of shoulder_mid → hip_mid from vertical)
    shoulder_mid = (xy[:, 5] + xy[:, 6]) / 2
    hip_mid = (xy[:, 11] + xy[:, 12]) / 2
    torso_vec = shoulder_mid - hip_mid  # points upward
    # Angle from vertical (0, -1)
    torso_angle = np.degrees(np.arctan2(torso_vec[:, 0], -torso_vec[:, 1] + 1e-8))
    features.append(_temporal_stats(torso_angle))

    # 4. Overall body center velocity
    body_center = (shoulder_mid + hip_mid) / 2
    if T >= 2:
        center_vel = np.linalg.norm(np.diff(body_center, axis=0), axis=1)
    else:
        center_vel = np.zeros(1, dtype=np.float32)
    features.append(_temporal_stats(center_vel))

    return np.concatenate(features)


# Number of features returned by extract_features
FEATURE_DIM = (
    len(ANGLE_DEFS) * 9       # 8 × 9 = 72
    + len(DISTANCE_PAIRS) * 9  # 6 × 9 = 54
    + 9                        # torso orientation
    + 9                        # body center velocity
)  # = 144


def pose_path_for_video(
    video_path: Path,
    pose_root: Path,
    repo_root: Optional[Path] = None,
) -> Path:
    """Return the expected pose .npy path for a given video path.

    Mirrors the layout written by ``scripts/pose_est.py``.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    video_path = Path(video_path).resolve()
    try:
        rel = video_path.relative_to(repo_root)
        stem_parts = rel.parts[1:-1]
    except ValueError:
        stem_parts = (video_path.parent.parent.name, video_path.parent.name)
    return pose_root / Path(*stem_parts) / f"{video_path.stem}.npy"


def filter_valid_pose(
    labeled_paths: List[Tuple[str, Dict]],
    pose_root: Path,
    repo_root: Optional[Path] = None,
    verbose: bool = True,
) -> List[Tuple[str, Dict]]:
    """Remove entries whose pre-computed pose .npy is missing."""
    valid, dropped = [], []
    for path_str, info in labeled_paths:
        ppath = pose_path_for_video(Path(path_str), pose_root, repo_root)
        if ppath.exists():
            valid.append((path_str, info))
        else:
            dropped.append(path_str)

    if verbose:
        print(
            f"[pose filter] {len(valid)} valid, {len(dropped)} dropped "
            f"(no pose data) out of {len(labeled_paths)} total."
        )
    return valid


class PoseFeatureDataset(Dataset):
    """Dataset that loads precomputed keypoints and returns engineered feature vectors.

    The full per-video keypoint sequence is used — FFT-based cadence and
    temporal statistics benefit from the longest available horizon, and the
    pose MLP's input is fixed-dim regardless of T.

    Args:
        labeled_paths: List of ``(video_path_str, {"label": int})`` tuples.
        pose_root:     Root directory containing ``<class>/<video>.npy`` pose files.
        repo_root:     Project root for resolving relative paths.
    """

    def __init__(
        self,
        labeled_paths: List[Tuple[str, Dict]],
        pose_root: Path,
        repo_root: Optional[Path] = None,
    ) -> None:
        self.labeled_paths = labeled_paths
        self.pose_root = pose_root
        self.repo_root = repo_root or Path(__file__).resolve().parent.parent

    def __len__(self) -> int:
        return len(self.labeled_paths)

    @property
    def num_videos(self) -> int:
        return len(self.labeled_paths)

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path_str, info = self.labeled_paths[idx]
        label = info["label"]

        pose_path = pose_path_for_video(
            Path(path_str), self.pose_root, self.repo_root
        )
        keypoints = np.load(pose_path)  # (T, 17, 3)
        features = extract_features(keypoints)

        return {
            "features": torch.from_numpy(features),
            "label": label,
        }
