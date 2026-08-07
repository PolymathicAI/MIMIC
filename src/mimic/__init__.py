"""MIMIC — a generative multimodal foundation model of the central dogma.

Public, lightweight package for inference, embedding, and generation over DNA,
RNA, and protein modalities.
"""
from loguru import logger

from .mimic import MIMIC
from .pretrained import build_model, load_pretrained
from . import strategies

# Structure I/O is re-exported lazily: it pulls in biotite, and importing `mimic`
# should stay cheap for the (many) users who never touch protein structures.
_STRUCTURE_EXPORTS = {
    "load_structure",
    "parse_structure_string",
    "pdb_to_numpy",
    "clean_structure",
    "struct_to_numpy",
    "numpy_to_struct",
}

__all__ = [
    "MIMIC", "build_model", "load_pretrained", "strategies", "logger",
    *sorted(_STRUCTURE_EXPORTS),
]


def __getattr__(name):
    if name in _STRUCTURE_EXPORTS:
        from .tokenizers.protein.structure import struct_utils

        return getattr(struct_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
