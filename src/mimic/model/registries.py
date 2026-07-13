from typing import Callable, TypeVar
from .base import EvaEncoder, EvaDecoder

# Type-safe global registries
ENCODER_REGISTRY: dict[str, Callable[..., EvaEncoder]] = {}
DECODER_REGISTRY: dict[str, Callable[..., EvaDecoder]] = {}

# Decorator for encoder registration
def register_encoder(name: str):
    def decorator(creator_fn: Callable[..., EvaEncoder]):
        if name in ENCODER_REGISTRY:
            raise ValueError(f"Encoder '{name}' already registered.")
        ENCODER_REGISTRY[name] = creator_fn
        return creator_fn
    return decorator

# Decorator for decoder registration
def register_decoder(name: str):
    def decorator(creator_fn: Callable[..., EvaDecoder]):
        if name in DECODER_REGISTRY:
            raise ValueError(f"Decoder '{name}' already registered.")
        DECODER_REGISTRY[name] = creator_fn
        return creator_fn
    return decorator

# Factory to instantiate encoder by name
def build_encoder(name: str, **kwargs) -> EvaEncoder:
    if name not in ENCODER_REGISTRY:
        raise KeyError(f"Unknown encoder '{name}'. Available: {list(ENCODER_REGISTRY.keys())}")
    encoder = ENCODER_REGISTRY[name](**kwargs)
    if not isinstance(encoder, EvaEncoder):
        raise TypeError(f"Object returned by encoder creator '{name}' is not an EvaEncoder.")
    return encoder

# Factory to instantiate decoder by name
def build_decoder(name: str, **kwargs) -> EvaDecoder:
    if name not in DECODER_REGISTRY:
        raise KeyError(f"Unknown decoder '{name}'. Available: {list(DECODER_REGISTRY.keys())}")
    decoder = DECODER_REGISTRY[name](**kwargs)
    if not isinstance(decoder, EvaDecoder):
        raise TypeError(f"Object returned by decoder creator '{name}' is not an EvaDecoder.")
    return decoder
