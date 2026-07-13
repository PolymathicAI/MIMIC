"""Loading and construction for the public MIMIC model.

`build_model` turns an inference config (the small set of architecture fields that
were previously buried in the training argparse Namespace) plus the modality
registry into a ready `MIMIC` module — with no data configs, dataloaders, or
training code. `load_pretrained` (P2+) will download a cleaned checkpoint from the
Hugging Face Hub, read its `config.json`, build the model, and load the weights.
"""
import json
import os

from loguru import logger

import torch
from safetensors.torch import load_file

from .mimic import MIMIC
from .model import build_encoder, build_decoder
from .model import encoder_embeddings as _text_emb_module
from .modality_info import MODALITY_INFO, build_modality_info, generate_uint15_hash


# Inference config for MIMIC 1.0.
# These are the only architecture fields the model needs to be reconstructed;
# `in_domains`/`out_domains` are filled from the checkpoint's config.json at load
# time (None here => build over the full modality registry, used for smoke tests).
_CONFIG_1_0 = {
    "encoder": "xt_encoder_20L_1536D",
    "decoder": "xt_decoder_12L_1536D",
    "encoder_dim": None,
    "decoder_dim": None,
    "encoder_depth": None,
    "decoder_depth": None,
    "encoder_position_code": "rotary",
    "decoder_position_code": "rotary",
    "rotary_emb_ratio": 0.75,
    "encoder_use_alibi": False,
    "decoder_use_alibi": False,
    "attn_flash": True,
    "mixed_attention": True,
    "unidir_attention_ratio": 0.5,
    "num_register_tokens": 5,
    "sum_modality_groups": True,
    "exclude_absent_tokens": True,
    "decoder_causal_mask": False,
    "decoder_sep_mask": True,
    "share_model_embeddings": True,
    "drop_enc_rate_min": 0.0,
    "drop_enc_rate_max": 0.1,
    "class_balance_max": None,
    "freeze_llm_emb": True,
    "num_input_tokens": 10000,
    "num_target_tokens": 1000,
    "is_target_autoregr": False,
    "dtype": "bfloat16",
    "in_domains": None,
    "out_domains": None,
    # inference: allocate text embeddings empty and fill from the checkpoint
    "init_text_from_biobert": False,
}

_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32,
}


def build_model(config: dict = None, modality_info: dict = None, group_info: dict = None) -> MIMIC:
    """Construct a `MIMIC` module from an inference config (defaults to MIMIC 1.0).

    Args:
        config: overrides merged onto `_CONFIG_1_0`.
        modality_info / group_info: pre-built registries; if omitted they are
            reconstructed from the config's in/out domains + `num_input_tokens`.
    Returns a model in eval mode with (random) initialized weights; callers load a
    checkpoint on top.
    """
    cfg = {**_CONFIG_1_0, **(config or {})}

    # The text token embedding is shared across text modalities and encoder/decoder
    # via a module-level global. Reset it so each build gets its own (otherwise a
    # second build in the same process would share tensors with the first).
    _text_emb_module._SHARED_TEXT_TOKEN_EMB = None

    in_domains = cfg["in_domains"] or sorted(MODALITY_INFO.keys())
    out_domains = cfg["out_domains"] or list(in_domains)

    if modality_info is None or group_info is None:
        modality_info, group_info = build_modality_info(
            in_domains, out_domains, cfg["num_input_tokens"]
        )

    logger.info(
        f"Building MIMIC: encoder={cfg['encoder']} decoder={cfg['decoder']} "
        f"over {len(in_domains)} in / {len(out_domains)} out modalities"
    )

    # --- per-modality embeddings ---
    encoder_embeddings = {}
    for mod in in_domains:
        info = modality_info[mod]
        emb_kwargs = {"position_code": cfg["encoder_position_code"]}
        if info["type"] == "text_token_all_targets":
            emb_kwargs["freeze_llm_emb"] = cfg["freeze_llm_emb"]
            emb_kwargs["init_text_from_biobert"] = cfg["init_text_from_biobert"]
        encoder_embeddings[mod] = info["encoder_embedding"](**emb_kwargs)

    decoder_embeddings = {}
    for mod in out_domains:
        info = modality_info[mod]
        emb_kwargs = {"position_code": cfg["decoder_position_code"]}
        if info["type"] == "text_token_all_targets":
            emb_kwargs["init_text_from_biobert"] = cfg["init_text_from_biobert"]
        decoder_embeddings[mod] = info["decoder_embedding"](**emb_kwargs)

    # --- encoder / decoder backbones ---
    encoder_kwargs = {}
    if cfg["encoder_dim"] is not None:
        encoder_kwargs["dim"] = cfg["encoder_dim"]
    if cfg["encoder_depth"] is not None:
        encoder_kwargs["depth"] = cfg["encoder_depth"]
    if cfg["encoder"].startswith("xt_"):
        encoder_kwargs["attn_flash"] = cfg["attn_flash"]
        encoder_kwargs["rotary_pos_emb"] = cfg["encoder_position_code"] == "rotary"
        encoder_kwargs["rotary_emb_ratio"] = cfg["rotary_emb_ratio"]
        encoder_kwargs["use_alibi"] = cfg["encoder_use_alibi"]
        if cfg["mixed_attention"]:
            encoder_kwargs["mixed_attention"] = True
            encoder_kwargs["unidir_attention_ratio"] = cfg["unidir_attention_ratio"]
            encoder_kwargs["num_register_tokens"] = cfg["num_register_tokens"]
    encoder = build_encoder(cfg["encoder"], **encoder_kwargs)

    decoder_kwargs = {}
    if cfg["decoder_dim"] is not None:
        decoder_kwargs["dim"] = cfg["decoder_dim"]
    if cfg["decoder_depth"] is not None:
        decoder_kwargs["depth"] = cfg["decoder_depth"]
    if cfg["decoder"].startswith("xt_"):
        decoder_kwargs["attn_flash"] = cfg["attn_flash"]
        decoder_kwargs["rotary_pos_emb"] = cfg["decoder_position_code"] == "rotary"
        decoder_kwargs["rotary_emb_ratio"] = cfg["rotary_emb_ratio"]
        # mimic ships x-transformer (non-mamba) encoders, so cross-attn is rotary
        # whenever the decoder uses rotary.
        decoder_kwargs["rotary_cross_attn"] = cfg["decoder_position_code"] == "rotary"
        decoder_kwargs["use_alibi"] = cfg["decoder_use_alibi"]
        if cfg["mixed_attention"]:
            decoder_kwargs["mixed_attention"] = True
            decoder_kwargs["unidir_attention_ratio"] = cfg["unidir_attention_ratio"]
    decoder = build_decoder(cfg["decoder"], **decoder_kwargs)

    assert encoder.dim == decoder.dim, (
        f"encoder dim {encoder.dim} != decoder dim {decoder.dim}"
    )

    model = MIMIC(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder=encoder,
        decoder=decoder,
        modality_info=modality_info,
        group_info=group_info,
        sum_modality_groups=cfg["sum_modality_groups"],
        exclude_absent_tokens=cfg["exclude_absent_tokens"],
        decoder_causal_mask=cfg["decoder_causal_mask"],
        decoder_sep_mask=cfg["decoder_sep_mask"],
        num_register_tokens=cfg["num_register_tokens"],
        share_modality_embeddings=cfg["share_model_embeddings"],
        drop_enc_rate_min=cfg["drop_enc_rate_min"],
        drop_enc_rate_max=cfg["drop_enc_rate_max"],
        class_balance_max=cfg["class_balance_max"],
        register_token_id=generate_uint15_hash("register"),
        num_input_tokens=cfg["num_input_tokens"],
        num_target_tokens=cfg["num_target_tokens"],
        is_target_autoregr=cfg["is_target_autoregr"],
        dtype=cfg["dtype"],
    )
    model.eval()
    return model


def load_pretrained(
    version: str = "1.0",
    local_path: str = None,
    device: str = "auto",
    hf_repo: str = "polymathic-ai/MIMIC",
    revision: str = None,
) -> MIMIC:
    """Load a released MIMIC model (``config.json`` + ``model.safetensors``).

    Args:
        version: release version; resolves to the git tag ``v<version>`` at the repo
            root (e.g. ``"1.0"`` -> tag ``v1.0``).
        local_path: load from this local directory instead of the Hugging Face Hub.
        device: ``"auto"`` (cuda if available, else cpu), ``"cuda"``, or ``"cpu"``.
        hf_repo: Hugging Face repo id.
        revision: explicit branch/tag/commit; overrides the ``v<version>`` default.

    Returns a ready `MIMIC` in eval mode on ``device``. Weights are held in fp32 and
    autocast to the config dtype at compute time (matching training-time inference).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if local_path is not None:
        cfg_path = os.path.join(local_path, "config.json")
        weights_path = os.path.join(local_path, "model.safetensors")
    else:
        from huggingface_hub import hf_hub_download
        # Versions live at the repo root, pinned by an immutable git tag v<version>.
        rev = revision or f"v{version}"
        cfg_path = hf_hub_download(hf_repo, "config.json", revision=rev)
        weights_path = hf_hub_download(hf_repo, "model.safetensors", revision=rev)

    with open(cfg_path) as f:
        config = json.load(f)
    # Flash attention is CUDA-only; fall back to standard attention off-GPU.
    config["attn_flash"] = bool(config.get("attn_flash", True)) and device == "cuda"

    model = build_model(config)
    state = {k: v.float() for k, v in load_file(weights_path).items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    logger.info(
        f"Loaded MIMIC {config.get('mimic_version', version)} on {device} "
        f"({len(state)} tensors)"
    )
    return model
