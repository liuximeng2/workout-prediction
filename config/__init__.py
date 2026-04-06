"""Configuration package.

Shared/global settings live in ``base_config.py``.
Model-specific settings live in ``models/<model_name>_config.py``.
"""

from config.base_config import BaseConfig

__all__ = ["BaseConfig"]
