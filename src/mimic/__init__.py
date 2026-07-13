"""MIMIC — a generative multimodal foundation model of the central dogma.

Public, lightweight package for inference, embedding, and generation over DNA,
RNA, and protein modalities.
"""
from loguru import logger

from .mimic import MIMIC
from .pretrained import build_model, load_pretrained
from . import strategies

__all__ = ["MIMIC", "build_model", "load_pretrained", "strategies", "logger"]
