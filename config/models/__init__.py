"""Per-model configuration classes."""

from config.models.video_mae_config import VideoMAEConfig
from config.models.two_stream_config import TwoStreamConfig
from config.models.vivit_config import ViViTConfig
from config.models.pose_config import PoseConfig
from config.models.cnn_fusion_config import CNNFusionConfig
from config.models.qwen3_vl_config import Qwen3VLConfig

__all__ = ["VideoMAEConfig", "TwoStreamConfig", "ViViTConfig", "PoseConfig", "CNNFusionConfig", "Qwen3VLConfig"]
