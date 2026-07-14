"""MIMIC model backbone: encoder/decoder transformer + per-modality embeddings."""

# Importing x_transformer registers the encoder/decoder presets
# (e.g. xt_encoder_20L_1536D) in the ENCODER/DECODER registries.
from . import x_transformer  # noqa: F401
from .base import EvaEncoder, EvaDecoder  # noqa: F401
from .registries import build_encoder, build_decoder  # noqa: F401
