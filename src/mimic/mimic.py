# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import random
import copy
from collections.abc import Sequence, Set
from functools import partial
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
import torch.nn.functional as F

from .data import fill_incomplete_mod_dict, pad_and_collate
from .model.base import EvaEncoder, EvaDecoder
from loguru import logger


# Autocast dtype used for inference; mirrors the training `--dtype` flag.
_DTYPE_MAP = {
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32,
}


class MIMIC(nn.Module):
    """MIMIC model (central-dogma multimodal encoder-decoder).

    Args:
        encoder_embeddings: Dict of encoder embedding modules.
        decoder_embeddings: Dict of decoder embedding modules.
        modality_info: Dict containing modality information.
        encoder: Encoder module.
        decoder: Decoder module.
        modality_info: Dict containing modality information.
        decoder_causal_mask: Whether to use causal mask in the decoder.
        NOTE: finish this.
    """
    def __init__(self,
                 encoder_embeddings: Dict[str, nn.Module],
                 decoder_embeddings: Dict[str, nn.Module],
                 encoder: EvaEncoder,
                 decoder: EvaDecoder,
                 modality_info: Dict[str, Any],
                 group_info: Dict[str, Any],
                 sum_modality_groups: bool,
                 exclude_absent_tokens: bool,
                 decoder_causal_mask: bool = False,
                 decoder_sep_mask: bool = True,
                 num_register_tokens: int = 0,
                 share_modality_embeddings: bool = True,
                 drop_enc_rate_min: float = 1.0,
                 drop_enc_rate_max: float = 1.0,
                 class_balance_max: float = None,
                 register_token_id: int = None,
                 num_input_tokens: int = 10000,
                 num_target_tokens: int = 1000,
                 is_target_autoregr: bool = False,
                 dtype: str = "bfloat16",
                 ):
        super().__init__()

        self.modality_info = modality_info
        self.group_info = group_info
        # Inference-time budgets (previously read off the training argparse Namespace).
        self.num_input_tokens = num_input_tokens
        self.num_target_tokens = num_target_tokens
        self.is_target_autoregr = is_target_autoregr
        self.dtype_str = dtype
        self.autocast_dtype = _DTYPE_MAP[dtype]
        self.decoder_causal_mask = decoder_causal_mask
        self.decoder_sep_mask = decoder_sep_mask
        self.init_std = 0.02
        self.num_register_tokens = num_register_tokens
        # Integer id used to mark register tokens in the encoder modality mask.
        # Must equal generate_uint15_hash("register"); the builder passes it, but
        # fall back to computing it for standalone construction.
        if register_token_id is None:
            from .modality_info import generate_uint15_hash
            register_token_id = generate_uint15_hash("register")
        self.register_token_id = register_token_id
        self.sum_modality_groups = sum_modality_groups
        self.exclude_absent_tokens = exclude_absent_tokens
        self.class_balance_max = class_balance_max

        assert drop_enc_rate_min <= drop_enc_rate_max and drop_enc_rate_max <= 1, "Minimum drop rate must be less than or equal to maximum drop rate and maximum drop rate must be less than or equal to 1."
        self.drop_enc_tokens = drop_enc_rate_min < 1
        self.drop_enc_rate = (drop_enc_rate_min, drop_enc_rate_max)

        self.encoder = encoder
        self.decoder = decoder

        # Encoder embeddings & init
        self.encoder_modalities = set(encoder_embeddings.keys())
        for emb in encoder_embeddings.values():
            emb.init(dim_tokens=encoder.dim, init_std=self.init_std)
        self.encoder_embeddings = nn.ModuleDict(encoder_embeddings)

        # Decoder embeddings & init
        self.decoder_modalities = set(decoder_embeddings.keys())
        for emb in decoder_embeddings.values():
            emb.init(dim_tokens=decoder.dim, init_std=self.init_std)
        self.decoder_embeddings = nn.ModuleDict(decoder_embeddings) 

        # Share modality embeddings across the encoder and decoder embedding modules
        if share_modality_embeddings:
            self.share_modality_embeddings()

        self.decoder_proj_context = nn.Linear(encoder.dim, decoder.dim)

        # Use decoder dimension for mask token since it's only used in decoder context
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder.dim))
        nn.init.normal_(self.mask_token, std=self.init_std)

        # Additional register tokens that can be used by the encoder during fine-tuning
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, encoder.dim))
            nn.init.normal_(self.register_tokens, std=self.init_std)
        else:
            self.register_tokens = None

        if self.drop_enc_tokens:
            # make a learnable mask vector for the input of the decoder when dropping encoder output tokens
            self.drop_enc_mask = nn.Parameter(torch.zeros(1, 1, encoder.dim))
            nn.init.normal_(self.drop_enc_mask, std=self.init_std)

        # Weight init
        self.init_weights()

        # Tokenizer / group lookups + stateful input buffer for the user-facing
        # input()/embed()/generate() API.
        self._setup_mods_groups()
        self.tok_input = None
        self.raw_input = None

    def share_modality_embeddings(self):
        """Share modality embeddings across the encoder and decoder embedding modules."""
        shared_modalities = self.encoder_modalities & self.decoder_modalities
        for mod in shared_modalities:
            # Only share embeddings if encoder and decoder have the same dimension
            if self.encoder.dim == self.decoder.dim:
                self.decoder_embeddings[mod].mod_emb = self.encoder_embeddings[mod].mod_emb
            else:
                # Skip sharing when dimensions differ to avoid dimension mismatch errors
                logger.warning(f"Skipping embedding sharing for modality '{mod}' due to different encoder/decoder dimensions (encoder: {self.encoder.dim}, decoder: {self.decoder.dim})")

    def init_weights(self):
        """Weight initialization following MAE's initialization scheme"""

        for name, m in self.named_modules():
            # Skipping tokenizers to avoid reinitializing them
            if "tokenizer" in name:
                continue
            # Skip text_token_emb components (BioBERT embeddings + projection + layer norm)
            # These are already properly and carefully initialized in the embedding modules with specific settings
            # This code prevents overwrite those initializations
            elif "text_token_emb" in name:
                continue
            # Linear
            elif isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # Embedding
            elif isinstance(m, nn.Embedding):
                # Only reinitialize embeddings that are trainable, skip frozen pre-trained embeddings
                if m.weight.requires_grad:
                    nn.init.normal_(m.weight, std=self.init_std)
            # Conv2d
            elif isinstance(m, nn.Conv2d):
                if '.proj' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def get_num_layers_encoder(self):
        return self.encoder.get_num_layers()

    def get_num_layers_decoder(self):
        return self.decoder.get_num_layers()

    def get_num_layers(self):
        return self.get_num_layers_encoder() + self.get_num_layers_decoder()

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = set()

        for mod, emb_module in self.encoder_embeddings.items():
            if hasattr(emb_module, 'no_weight_decay'):
                to_skip = emb_module.no_weight_decay()
                to_skip = set([f'encoder_embeddings.{mod}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        for mod, emb_module in self.decoder_embeddings.items():
            if hasattr(emb_module, 'no_weight_decay'):
                to_skip = emb_module.no_weight_decay()
                to_skip = set([f'decoder_embeddings.{mod}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def cat_encoder_tensors(self, mod_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        """Concatenate encoder tensors from different modalities.

        Args:
            mod_dict (dict): A dictionary containing information for each modality. 
                             Expected keys for each modality are 'x' (input tokens), 
                             'emb' (embeddings), 'input_mask', etc.

        Returns:
            tuple:
                - encoder_tokens_all (torch.Tensor): Concatenated encoder tokens from all modalities. Shape (B, O, D) where O is the total number of all encoder tokens.
                - emb_all (torch.Tensor): Concatenated encoder embeddings from all modalities. Shape (B, O, D)
                - encoder_mask_all (torch.Tensor): Concatenated boolean masks indicating which tokens are part of the encoder input (set to 0 for valid tokens, 1 otherwise). Shape (B, O)
                - mod_mask_all (torch.Tensor): Concatenated integer mask marking the modality type for each encoder token. Shape (B, O)
        """

        encoder_tokens_all = []
        emb_all = []
        encoder_mask_all = []
        mod_mask_all = []

        if self.sum_modality_groups:
            # iterate over the summation groups and sum the x, emb, input_mask
            group_dict = {}
            for group, info in self.group_info.items():
                mods = info["mods"]

                # assert that either all modalities in the group are present or none are
                assert all(mod in mod_dict for mod in mods) or all(mod not in mod_dict for mod in mods), \
                    f"Either all modalities in summation group {group} should be present or none should be present."
                # if none of the modalities in the group are present, skip
                if all(mod not in mod_dict for mod in mods):
                    continue

                # If chosen, we drop the absent tokens from the group summation
                mask = lambda mod: (~mod_dict[mod]['input_mask'].unsqueeze(-1) if self.exclude_absent_tokens else 1)

                group_dict[group] = {
                    'x': torch.stack([mod_dict[mod]['x']  *  mask(mod) for mod in mods], 0).sum(0),
                    'emb': torch.stack([mod_dict[mod]['emb'] *  mask(mod) for mod in mods], 0).sum(0),
                    'input_mask': torch.stack([mod_dict[mod]['input_mask'] for mod in mods], 0).all(0),
                    'id': group,
                }

        else:
            # if not summing, each modality is its own group
            for mod in mod_dict.keys():
                mod_dict[mod]['id'] = self.modality_info[mod]['id']
            group_dict = mod_dict

        for group, d in group_dict.items():
            encoder_tokens_all.append(d['x'])
            emb_all.append(d['emb'])
            encoder_mask_all.append(d['input_mask'])
            mod_mask_all.append(torch.full_like(d['input_mask'], d['id'], dtype=torch.int16))
        
        encoder_tokens_all = torch.cat(encoder_tokens_all, dim=1)
        emb_all = torch.cat(emb_all, dim=1)
        encoder_mask_all = torch.cat(encoder_mask_all, dim=1)
        mod_mask_all = torch.cat(mod_mask_all, dim=1)

        return encoder_tokens_all, emb_all, encoder_mask_all, mod_mask_all

    def cat_decoder_tensors(self, mod_dict: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor]:
        """Concatenate decoder tensors from different modalities.
        
        Args:
            mod_dict (dict): A dictionary containing information for each modality.
                             Expected keys for each modality include 'x' (input tokens),
                             'ids' (target IDs), 'emb' (embeddings), 'target_mask', 'decoder_attention_mask', etc.

        
        Returns:
            tuple:
                - decoder_tokens_all (torch.Tensor): Concatenated decoder tokens from all modalities. Shape (B, P, D) where P is the total number of all decoder tokens.
                - emb_all (torch.Tensor): Concatenated decoder embeddings from all modalities. Shape (B, P, D)
                - decoder_mask_all (torch.Tensor): Concatenated boolean masks indicating which tokens are part of the decoder input / target (set to 0 for valid tokens, 1 otherwise). Shape (B, P)
                - target_ids_all (torch.Tensor): Concatenated target IDs from all modalities. Shape (B, P)
                - attention_mask_all (torch.Tensor): Concatenated attention masks in compressed format, needs to be passed to adapt_decoder_attention_mask() to obtain the final attention mask. Shape (B, P)
                - mod_mask_all (torch.Tensor): Concatenated integer mask marking the modality type for each decoder token. Shape (B, P)
        """

        decoder_tokens_all = []
        target_ids_all = []
        emb_all = []
        decoder_mask_all = []
        attention_mask_all = []
        mod_mask_all = []

        # Shuffle order in which modalities are provided (useful for modality causal mask)
        mod_dict = {mod: d for mod, d in random.sample(list(mod_dict.items()), len(mod_dict))}

        for mod, d in mod_dict.items():
            if self.modality_info[mod]['type'] in ['chain_token']: # MLM modeling modalities
                # For MLM we simply feed in the mask token
                decoder_tokens_all.append(torch.zeros_like(d['x']) + self.mask_token)  # Replace x by mask token
                target_ids_all.append(d['ids'])
                emb_all.append(d['emb'])
                decoder_mask_all.append(d['target_mask'])
                attention_mask_all.append(d['decoder_attention_mask'])
                mod_mask_all.append(torch.full_like(d['ids'], self.modality_info[mod]['id'], dtype=torch.int16))
            elif self.modality_info[mod]['type'] in ["text_token_all_targets"]:
                # autoregressive 
                decoder_tokens_all.append(d['x'][:, :-1])
                target_ids_all.append(d['ids'][:, 1:])  # Shifted left
                emb_all.append(d['emb'][:, :-1])
                decoder_mask_all.append(d['target_mask'][:, 1:])  # Shifted left
                attention_mask_all.append(d['decoder_attention_mask'][:, :-1])
                mod_mask_all.append(torch.full_like(d['ids'][:, :-1], self.modality_info[mod]['id'], dtype=torch.int16))


        decoder_tokens_all = torch.cat(decoder_tokens_all, dim=1)
        emb_all = torch.cat(emb_all, dim=1)
        decoder_mask_all = torch.cat(decoder_mask_all, dim=1)
        target_ids_all = torch.cat(target_ids_all, dim=1)
        attention_mask_all = torch.cat(attention_mask_all, dim=1)
        mod_mask_all = torch.cat(mod_mask_all, dim=1)

        return decoder_tokens_all, emb_all, decoder_mask_all, target_ids_all, attention_mask_all, mod_mask_all

    def run_positions(self, x: torch.Tensor):
        """
        x: Long/any dtype tensor of shape [B, T].
        returns: Long tensor of shape [B, T] with positions 0..len(run)-1
        """
        B, T = x.shape
        # Run starts: first position or value different from previous
        starts = torch.ones((B, T), dtype=torch.bool, device=x.device)
        starts[:, 1:] = x[:, 1:] != x[:, :-1]

        # Indices 0..T-1 across time, broadcast to [B, T]
        t = torch.arange(T, device=x.device).expand(B, -1)

        # For non-starts, put -1; then forward-fill the last start index via cummax
        start_idx = torch.where(starts, t, torch.full((B, T), -1, device=x.device))
        last_start, _ = torch.cummax(start_idx, dim=1)   # forward-fill of last start position

        # Position inside the current run = current index - last start index
        pos = t - last_start
        return pos.long()


    def forward_mask_encoder(self, mod_dict: Dict[str, Dict[str, torch.Tensor]], num_encoder_tokens: int) -> Tuple[torch.Tensor]:
        """Concatenates and mask encoder tensors based on provided modality information.

        This function consolidates encoder tokens from multiple modalities, then selects a specified number of them based on modality information (i.e. masking).

        Args:
            mod_dict (dict): Dictionary containing tensors for different modalities. 
                            It is expected to have keys for each modality and values 
                            containing the modalities' associated tensors.
            num_encoder_tokens (int): Number of encoder tokens to retain after masking.

        Returns:
            tuple:
                - encoder_tokens (torch.Tensor): Selected encoder tokens from all modalities. Shape (B, N, D) where N is the number of selected encoder tokens. 
                - encoder_emb (torch.Tensor): Corresponding embeddings for encoder tokens. Shape (B, N, D)
                - encoder_mask (torch.Tensor): A boolean mask indicating which encoder tokens are valid (set to 0 for valid tokens, 1 otherwise). Shape (B, 1, N)
                - mod_mask (torch.Tensor): An integer mask marking the modality type for each encoder token (with -1 indicating unassigned pad tokens). Shape (B, N)

        Notes:
            - If `num_register_tokens` is set and greater than 0, register tokens are added at the beginning of the sequence.
        """
        B = list(mod_dict.values())[0]['tensor'].shape[0]

        encoder_tokens_all, emb_all, encoder_mask_all, mod_mask_all = self.cat_encoder_tensors(mod_dict)
        rel_pos_all = self.run_positions(mod_mask_all)

        # Add arange multiplied by small constant to mask so they get sorted in a deterministic way
        mask_arange = torch.arange(encoder_mask_all.shape[1], device=encoder_mask_all.device).unsqueeze(0) * 1e-6
        ids_shuffle = torch.argsort(encoder_mask_all + mask_arange, dim=1)
        # ids_restore = torch.argsort(ids_shuffle, dim=1)
        num_keep_tokens = min(num_encoder_tokens, (~encoder_mask_all).sum(1).max())
        pos_keep = ids_shuffle[:, :num_keep_tokens]

        encoder_tokens = torch.gather(encoder_tokens_all, dim=1,
                                      index=repeat(pos_keep, "b n -> b n d", d=encoder_tokens_all.shape[2]))
        encoder_emb = torch.gather(emb_all, dim=1, index=repeat(pos_keep, "b n -> b n d", d=emb_all.shape[2]))
        encoder_mask = torch.gather(encoder_mask_all, dim=1, index=pos_keep)
        mod_mask = torch.gather(mod_mask_all, dim=1, index=pos_keep)
        rel_pos_idx = torch.gather(rel_pos_all, dim=1, index=pos_keep)

        if self.num_register_tokens > 0:
            register_tokens = repeat(self.register_tokens, '() n d -> b n d', b=B)
            # We add register tokens at the beginning of the sequence
            encoder_tokens = torch.cat([register_tokens, encoder_tokens], dim=1)
            encoder_emb = torch.cat([torch.zeros_like(register_tokens), encoder_emb], dim=1)
            encoder_mask = torch.cat([torch.zeros((B, register_tokens.shape[1]), dtype=torch.bool, device=encoder_mask.device), encoder_mask], dim=1)
            mod_mask = torch.cat([torch.full((B, register_tokens.shape[1]), self.register_token_id, dtype=torch.int16, device=mod_mask.device), mod_mask], dim=1)
            # add the register tokens to pos_keep
            reg_idx = torch.arange(self.num_register_tokens, device=pos_keep.device).unsqueeze(0).expand(B, -1)
            rel_pos_idx = torch.cat([reg_idx, rel_pos_idx], dim=1)

        # set the rel_pos_idx to -1 wherever encoder_mask is True
        rel_pos_idx[encoder_mask.squeeze(1)] = -1

        encoder_tokens[encoder_mask] = 0.
        encoder_emb[encoder_mask] = 0.
        mod_mask[encoder_mask] = -1
        # Mask could be of shape 'b n1 n2' but not needed for masked_fill
        # This means this mask can then be re-used for decoder cross-attention
        encoder_mask = rearrange(encoder_mask, 'b n2 -> b 1 n2')

        return encoder_tokens, encoder_emb, encoder_mask, mod_mask, rel_pos_idx

    def forward_mask_decoder(self, mod_dict: Dict[str, Dict[str, torch.Tensor]], num_decoder_tokens: int) -> Tuple[torch.Tensor]:
        """Concatenates and mask decoder tensors based on provided modality information.

        This function consolidates decoder tokens from multiple modalities, selects a specified number of them based on modality information, and applies appropriate masking.

        Args:
            mod_dict (dict): Dictionary containing tensors for different modalities.
                            It is expected to have keys for each modality and values 
                            containing the modalities' associated tensors.
            num_decoder_tokens (int): Number of decoder tokens to retain after masking.

        Returns:
            tuple:
                - decoder_tokens (torch.Tensor): Selected decoder tokens from all modalities. Shape (B, M, D) where M is the number of selected decoder tokens.
                - decoder_emb (torch.Tensor): Corresponding embeddings for decoder tokens. Shape (B, M, D)
                - decoder_mask (torch.Tensor): A boolean mask indicating which decoder tokens are valid (set to 0 for valid tokens, 1 otherwise). Shape (B, 1, M)
                - target_ids (torch.Tensor): IDs of the target tokens corresponding to the decoder tokens. Shape (B, M)
                - decoder_attention_mask (torch.Tensor): Mask for the decoder self-attention layers. Shape (B, M, M)
                - mod_mask (torch.Tensor): An integer mask marking the modality type for each decoder token (with -1 indicating unassigned pad tokens). Shape (B, M)
        """

        # decoder_mask and target_mask are equivalent, we rename it here to harmonize with forward_mask_encoder
        decoder_tokens_all, emb_all, decoder_mask_all, target_ids_all, decoder_attention_mask_all, mod_mask_all = self.cat_decoder_tensors(mod_dict)

        # If any row is fully masked, we add a dummy unmasked token to avoid issues
        empty_samples = decoder_mask_all.all(dim=1)  # shape (B,)

        if empty_samples.any():
            logger.error(f"Empty targets detected for {empty_samples.sum().item()} samples. Adding dummy tokens to prevent crash.")

            B = decoder_mask_all.shape[0]
            device = decoder_mask_all.device

            # create the dummy components
            dummy_token = repeat(self.mask_token, '1 1 d -> b 1 d', b=B)
            dummy_emb = torch.zeros_like(dummy_token)

            # Mask is False only for empty rows
            dummy_mask = ~empty_samples.unsqueeze(1)  # shape (B, 1)
            
            dummy_target_ids = torch.full((B, 1), -100, dtype=target_ids_all.dtype, device=device)
            dummy_attn = torch.zeros((B, 1), dtype=decoder_attention_mask_all.dtype, device=device)
            dummy_mod_mask = torch.full((B, 1), -1, dtype=mod_mask_all.dtype, device=device)

            # Concatenate to ensure at least one unmasked token per sample
            decoder_tokens_all = torch.cat([decoder_tokens_all, dummy_token], dim=1)
            emb_all = torch.cat([emb_all, dummy_emb], dim=1)
            decoder_mask_all = torch.cat([decoder_mask_all, dummy_mask], dim=1)
            target_ids_all = torch.cat([target_ids_all, dummy_target_ids], dim=1)
            decoder_attention_mask_all = torch.cat([decoder_attention_mask_all, dummy_attn], dim=1)
            mod_mask_all = torch.cat([mod_mask_all, dummy_mod_mask], dim=1)

        rel_pos_all = self.run_positions(mod_mask_all)

        # Add arange multiplied by small constant to mask so they get sorted in a deterministic way
        mask_arange = torch.arange(decoder_mask_all.shape[1], device=decoder_mask_all.device).unsqueeze(0) * 1e-6
        ids_shuffle = torch.argsort(decoder_mask_all + mask_arange, dim=1)
        # ids_restore = torch.argsort(ids_shuffle, dim=1)

        num_keep_tokens = min(num_decoder_tokens, (~decoder_mask_all).sum(1).max())
        pos_keep = ids_shuffle[:, :num_keep_tokens]

        decoder_tokens = torch.gather(decoder_tokens_all, dim=1, index=repeat(pos_keep, "b n -> b n d", d=decoder_tokens_all.shape[2]))
        decoder_emb = torch.gather(emb_all, dim=1, index=repeat(pos_keep, "b n -> b n d", d=emb_all.shape[2]))
        decoder_mask = torch.gather(decoder_mask_all, dim=1, index=pos_keep)
        target_ids = torch.gather(target_ids_all, dim=1, index=pos_keep)
        decoder_attention_mask = torch.gather(decoder_attention_mask_all, dim=1, index=pos_keep)
        mod_mask = torch.gather(mod_mask_all, dim=1, index=pos_keep)
        rel_pos_idx = torch.gather(rel_pos_all, dim=1, index=pos_keep)

        # set the rel_pos_idx to -1 wherever decoder is True
        rel_pos_idx[decoder_mask.squeeze(1)] = -1

        decoder_tokens[decoder_mask] = 0.
        decoder_emb[decoder_mask] = 0.
        target_ids[decoder_mask] = 0
        decoder_attention_mask = self.adapt_decoder_attention_mask(decoder_attention_mask, mod_mask)
        mod_mask[decoder_mask] = -1

        # This means this mask can then be re-used for decoder cross-attention
        decoder_mask = rearrange(decoder_mask, 'b n2 -> b 1 n2')

        return decoder_tokens, decoder_emb, decoder_mask, target_ids, decoder_attention_mask, mod_mask, rel_pos_idx

    def adapt_decoder_attention_mask(self, decoder_attention_mask: torch.Tensor, mod_mask=Optional[torch.Tensor]) -> torch.Tensor:
        """
        Transforms the compressed decoder attention mask to a full attention mask based on the specified constraints.

        Args:
            decoder_attention_mask (torch.Tensor): Initial attention mask indicating attention constraints. Shape (B, M) where M is the number of the decoder tokens.
            mod_mask (torch.Tensor, optional): Modality mask to separate attention masks per modality. Shape (B, M)

        Returns:
            torch.Tensor: Adapted attention mask. Shape (B, M, M) where M is the number of the decoder tokens.
        """
        B, N = decoder_attention_mask.shape

        if self.decoder_causal_mask:
            # For causal mode, tokens can only attend to preceding tokens and themselves.
            causal_mask = torch.ones((N, N), dtype=torch.bool, device=decoder_attention_mask.device).triu(1)
            causal_mask = repeat(causal_mask, "n1 n2 -> b n1 n2", b=B)
            adapted_attention_mask = causal_mask
        else:
            # Cumulatively sum the attention mask to determine token-wise attention behavior.
            # Examples:
            # Mask [4, 0, 0, 0] -> Cumsum: [4, 4, 4, 4] -> All tokens attend to each other.
            # Mask [1, 1, 1, 1] -> Cumsum: [1, 2, 3, 4] -> Strict autoregressive behavior.
            # Mask [2, 0, 1, 1] -> Cumsum: [2, 2, 3, 4] -> Tokens 1 and 2 attend to each other, token 3 attends to tokens 1-3, and token 4 to all.
            attention_arange = torch.arange(N, device=decoder_attention_mask.device)
            attention_arange = repeat(attention_arange, "n2 -> b n1 n2", b=B, n1=N)
            cumsum_mask = torch.cumsum(decoder_attention_mask, dim=-1)
            cumsum_mask = rearrange(cumsum_mask, "b n -> b n 1")
            adapted_attention_mask = (attention_arange >= cumsum_mask)

        if self.decoder_sep_mask:
            # Separate attention between tokens based on their modality using mod_mask.
            sep_mask = repeat(mod_mask, "b n2 -> b n1 n2", n1=N) != repeat(mod_mask, "b n1 -> b n1 n2", n2=N)
            adapted_attention_mask = adapted_attention_mask | sep_mask

        return adapted_attention_mask

    def forward_loss(self, 
                    y: torch.Tensor, 
                    target_ids: torch.Tensor, 
                    decoder_mod_dict: Dict[str, Any], 
                    decoder_mod_mask: torch.Tensor,
                    tok_count_func: str = 'none', 
                    return_logits: bool = False) -> Dict[str, torch.Tensor]:
        """Computes the token-wise loss.

        Args:
            y (torch.Tensor): Decoder tokens. Shape (B, M, D).
            target_ids (torch.Tensor): Ground truth token IDs. Shape (B, M).
            decoder_mod_dict (dict): Dictionary containing tensor information for each modality in the decoder.
            decoder_mod_mask (torch.Tensor): Mask indicating which tokens belong to which modality. Shape (B, M).
            unused_encoder_mods (list): List of encoder modalities that had no input tokens.
            unused_decoder_mods (list): List of decoder modalities that had no target tokens
            tok_count_func (str): Function to apply to the token count for each modality. Can be 'lin', 'sqrt', 'log', or 'none'.
                                  When 'none', tok_count is set to 1 for each modality that has count > 0.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total token loss and dictionary of loss for each modality.
        """

        mod_loss = {}
        mod_count = {}
        if return_logits: 
            mod_logits = {}
            mod_labels = {}

        for mod, d in decoder_mod_dict.items():
            idx = self.modality_info[mod]["id"]
            logits = self.decoder_embeddings[mod].forward_logits(y[decoder_mod_mask == idx])
            if return_logits:
                mod_logits[mod] = logits
            if logits.numel() == 0:
                # If there are no logits / targets, set mod_loss to 0
                mod_loss[mod] = torch.tensor(0., device=logits.device)
                mod_count[mod] = 0
            else:
                labels = target_ids[decoder_mod_mask == idx]
                if return_logits:
                    mod_labels[mod] = labels
                if "class_balance" in self.modality_info[mod]:
                    weight = 1 / torch.sqrt(torch.tensor(self.modality_info[mod]["class_balance"], device=logits.device))
                    if self.class_balance_max is not None:
                        weight = torch.clamp(weight, max=self.class_balance_max)
                else:
                    weight = None
                loss = F.cross_entropy(logits, labels.long(), reduction='mean', weight=weight)
                mod_loss[mod] = loss
                mod_count[mod] = logits.shape[0]

        # keep an unmodified copy of mod_count for output
        mod_count_orig = mod_count.copy()

        # If tok_count_func is specified, apply it to the mod_count
        if tok_count_func == 'sqrt':
            mod_count = {mod: math.sqrt(count) for mod, count in mod_count.items()}
        elif tok_count_func == 'log':
            mod_count = {mod: math.log(count + 1) for mod, count in mod_count.items()}
        elif tok_count_func == 'lin':
            pass
        elif tok_count_func == 'none':
            mod_count = {mod: 1 if count > 0 else 0 for mod, count in mod_count.items()}
        else:
            raise ValueError(f"Unknown tok_count_func: {tok_count_func}. Supported values are 'lin', 'sqrt', 'log', or 'none'.")
        if total_count := sum(mod_count.values()) > 0 :
            loss = sum(mod_loss[mod] * mod_count[mod] for mod in mod_loss.keys()) / total_count
        else:
            loss = y.sum() * 0.0  # zero loss if no tokens

        output = {'loss': loss, 'mod_loss': mod_loss, 'mod_count':mod_count, 'mod_count_orig':mod_count_orig}
        if return_logits:
            output['mod_logits'] = mod_logits
            output['mod_labels'] = mod_labels
        
        return output


    def forward(self, 
            mod_dict: Dict[str, Dict[str, torch.Tensor]], 
            num_encoder_tokens: int, 
            num_decoder_tokens: int, 
            tok_count_func: str = 'none',
            return_logits: bool = False,
            return_encoder_output: bool = False,
            return_loss: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the model.

        Args:
            mod_dict (Dict[str, Dict[str, torch.Tensor]]): Dictionary containing the tensors, masks, and other info for each modality.
                - mod_dict[modality_name]["tensor_name"]: Shape can vary based on tensor_name and modality.
            num_encoder_tokens (int): Number of tokens to keep for the encoder.
            num_decoder_tokens (int): Number of tokens to keep for the decoder.
            tok_count_func (str, optional): Function to apply to the token count for each modality. Can be 'lin', 'sqrt', 'log', or 'none'.
                When 'none', tok_count is set to 1 for each modality that has count > 0.
            return_logits (bool, optional): If True, return the logits. Default is False.
            return_all_logits (bool, optional): If True, return logits for all tokens in the decoder output. If False, separate logits by modality. Default is True.
        Returns:
            Union[dict, tuple]: 
                - If return_logits is True: Dictionary of logits for each modality.
                - Otherwise: Tuple containing the total loss and dictionary of loss for each modality.
        """

        # Mod dicts
        encoder_mod_dict = {mod: self.encoder_embeddings[mod](d)
                            for mod, d in mod_dict.items()
                            if mod in self.encoder_embeddings}
        encoder_tokens, encoder_emb, encoder_mask, encoder_mod_mask, input_pos_idx = self.forward_mask_encoder(encoder_mod_dict, num_encoder_tokens)
        # assert that the encoder_tokens length is less than the num_input_tokens + num_register_tokens
        assert encoder_tokens.shape[1] <= num_encoder_tokens + self.num_register_tokens, \
            f"Encoder tokens length {encoder_tokens.shape[1]} is greater than the sum of num_encoder_tokens {num_encoder_tokens} and num_register_tokens {self.num_register_tokens}"

        # Encoder
        x = encoder_tokens + encoder_emb
        x = self.encoder(x, encoder_mask=encoder_mask, input_pos_idx = input_pos_idx)
        
        output = {}
        if return_encoder_output:
            output.update({'encoder_output': {'x':x, 'encoder_mask':encoder_mask, 'encoder_mod_mask':encoder_mod_mask, 'encoder_emb':encoder_emb}})
            if not (return_logits or return_loss):
                return output
        
        if self.drop_enc_tokens and self.training:
            # sample a number between the min max drop rates using torch uniform
            drop_rate = torch.rand(1) * (self.drop_enc_rate[1] - self.drop_enc_rate[0]) + self.drop_enc_rate[0]
            # create a binary mask with the same shape as the encoder tokens other than the register tokens
            drop_mask = torch.rand_like(encoder_tokens[:, self.num_register_tokens:, 0]) < drop_rate.item()
            
            # Create the full drop mask for the entire tensor
            full_drop_mask = torch.zeros_like(x, dtype=torch.bool)
            full_drop_mask[:, self.num_register_tokens:] = drop_mask.unsqueeze(-1)
            
            # Replace tokens with drop_enc_mask where drop_mask is True
            # The replacement is done since dropping the tokens would lead to different length sequences.
            x = torch.where(full_drop_mask, self.drop_enc_mask.expand_as(x), x)
            
            # Replace encoder_emb with zeros where drop_mask is True
            encoder_emb = torch.where(full_drop_mask, torch.zeros_like(encoder_emb), encoder_emb)

        # Decoder processing only when there are decoder tokens
        decoder_mod_dict = {mod: self.decoder_embeddings[mod].forward_embed(d)
                            for mod, d in mod_dict.items()
                            if mod in self.decoder_embeddings}
        decoder_tokens, decoder_emb, decoder_mask, target_ids, decoder_attention_mask, decoder_mod_mask, target_pos_idx = self.forward_mask_decoder(decoder_mod_dict, num_decoder_tokens)
        
        # Decoder
        context = self.decoder_proj_context(x + encoder_emb) # Project encoder output to decoder dimension
        y = decoder_tokens + decoder_emb
        y = self.decoder(
            y, context,
            encoder_mask=encoder_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_mask=decoder_mask,
            input_pos_idx=input_pos_idx,
            target_pos_idx=target_pos_idx
        )

        # Loss
        output.update(self.forward_loss(
            y, target_ids, decoder_mod_dict, decoder_mod_mask, tok_count_func=tok_count_func, return_logits=return_logits))

        return output

    def freeze_encoder(self, freeze_embeddings=True):
        self.encoder.freeze()

        if freeze_embeddings:
            for param in self.encoder_embeddings.parameters():
                param.requires_grad = False

    def freeze_encoder_except_specific_embeddings(self, frozen_embedding_domain):
        self.encoder.freeze()
        
        frozen_embedding_domain = frozen_embedding_domain.split('-')

        for name, param in self.encoder_embeddings.named_parameters():
            if name.split('.')[0] in frozen_embedding_domain:
                param.requires_grad = False

    def unfreeze_encoder(self, unfreeze_embeddings=True):
        self.encoder.unfreeze()

        if unfreeze_embeddings:
            for param in self.encoder_embeddings.parameters():
                param.requires_grad = True

    def freeze_decoder(self, freeze_embeddings=True):
        self.decoder.freeze()

        if freeze_embeddings:
            for param in self.decoder_embeddings.parameters():
                param.requires_grad = False

    def freeze_decoder_except_specific_embeddings(self, frozen_embedding_domain):
        self.decoder.freeze()

        frozen_embedding_domain = frozen_embedding_domain.split('-')

        for name, param in self.decoder_embeddings.named_parameters():
            if name.split('.')[0] in frozen_embedding_domain:
                param.requires_grad = False

    def unfreeze_decoder(self, unfreeze_embeddings=True):
        self.decoder.unfreeze()

        if unfreeze_embeddings:
            for param in self.decoder_embeddings.parameters():
                param.requires_grad = True

    def freeze_shared_params(self):
        self.freeze_encoder(freeze_embeddings=False)
        self.freeze_decoder(freeze_embeddings=False)

    def freeze_params_except_specific_embeddings(self, frozen_embedding_domain):
        self.freeze_encoder_except_specific_embeddings(frozen_embedding_domain=frozen_embedding_domain)
        self.freeze_decoder_except_specific_embeddings(frozen_embedding_domain=frozen_embedding_domain)

    def unfreeze_shared_params(self):
        self.unfreeze_encoder(unfreeze_embeddings=False)
        self.unfreeze_decoder(unfreeze_embeddings=False)

    def unfreeze_all(self):
        self.unfreeze_encoder(unfreeze_embeddings=True)
        self.unfreeze_decoder(unfreeze_embeddings=True)

    def llm_embeddings_freeze(self):
        # List of LLM embedding modules to potentially freeze
        llm_modules_potentially_freeze = []
        for mod in self.modality_info:
            if self.modality_info[mod]['type'] =='text_token_all_targets':
                llm_modules_potentially_freeze += [f"encoder_embeddings.tok_{mod}.text_token_emb.0.weight"]
                llm_modules_potentially_freeze += [f"decoder_embeddings.tok_{mod}.text_token_emb.0.weight"]
        
         # Actual freeze
        if self.encoder_embeddings.tok_context.freeze_llm_emb:
            for name, param in self.named_parameters():
                if name in llm_modules_potentially_freeze: # Only freeze LLM embeddings
                    param.requires_grad = False

    # ------------------------------------------------------------------
    # User-facing inference API (input / embed / generate).
    #
    # These previously lived on a separate `EVAInterface` wrapper; folding them
    # onto the model itself makes a single `MIMIC` object fully self-contained:
    #   model.input(samples); reps = model.embed(); out = model.generate(target)
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Device the parameters live on; used to move staged inputs in _model_forward."""
        return next(self.parameters()).device

    def _setup_mods_groups(self):
        """Precompute tokenizer / modality-group lookups from modality_info + group_info."""
        self.all_mods = set(self.modality_info.keys())
        self.tokenizers = {mod: self.modality_info[mod]["tokenizer"] for mod in self.all_mods}
        self.group_names = {gid: info["name"] for gid, info in self.group_info.items()}
        self.group_names[self.register_token_id] = "register"
        self.mod_group_lookup = {mod: gid for gid, info in self.group_info.items() for mod in info["mods"]}

    def _lookup_mod(self, mod: str) -> str:
        """Resolve a user modality name (with or without a `tok_` prefix) to its model key."""
        mod = mod.lower().split("tok_")[-1]
        if f"tok_{mod}" in self.all_mods:
            return f"tok_{mod}"
        raise ValueError(f"Modality {mod} is not in the tokenized modalities.")

    def input(self, batch: Union[dict, Sequence[dict]], bypass_input_length_check: bool = False):
        """Tokenize and stage a batch of samples for embed()/generate().

        Each sample is a dict mapping a modality name to a raw value (tokenized via that
        modality's tokenizer) or, when the key is prefixed with `tok_`, to already-tokenized
        ids. The result is stored on `self.tok_input` / `self.raw_input`.
        """
        batch = [batch] if isinstance(batch, dict) else list(batch)

        tok_input = []
        for sample in batch:
            tok_sample = {}
            for mod, val in sample.items():
                lookup_mod = self._lookup_mod(mod)
                if mod.startswith("tok_"):
                    tok_sample[lookup_mod] = val.tolist() if isinstance(val, np.ndarray) else val
                else:
                    tok_sample[lookup_mod] = self.tokenizers[lookup_mod].tokenize(val)

            if not bypass_input_length_check:
                for mod, data in tok_sample.items():
                    assert len(data) <= self.modality_info[mod]["max_tokens"], \
                        f"Input length {len(data)} for modality {mod} exceeds max_tokens {self.modality_info[mod]['max_tokens']}"

            # Within each summation group, all provided modalities must share a length.
            for group, info in self.group_info.items():
                group_lens = set(len(tok_sample[mod]) for mod in info["mods"] if mod in tok_sample)
                assert len(group_lens) <= 1, \
                    f"Modalities in group {group}:{info['name']} have different lengths: {group_lens}"

            tok_input.append(tok_sample)

        self.raw_input = batch
        self.tok_input = tok_input

    def _create_model_input(self, tok_input, target_mods=None, target_lens=None):
        """Turn staged token lists into a padded, collated model-ready mod_dict."""
        target_mods = target_mods or []
        model_input_list = []
        if not target_lens:
            target_lens = [0] * len(tok_input)

        for target_len, tok_sample in zip(target_lens, tok_input):
            sample_input = {}
            for mod, tensor in tok_sample.items():
                tensor = torch.tensor(tensor)
                sample_input[mod] = {
                    'tensor': tensor,
                    'input_mask': torch.zeros_like(tensor, dtype=torch.bool),
                    'target_mask': torch.ones_like(tensor, dtype=torch.bool),
                    'decoder_attention_mask': torch.zeros_like(tensor, dtype=torch.bool),
                }
                # A token equal to the modality's mask id marks a position to generate.
                mask_locs = tensor == self.tokenizers[mod].mask_token_id
                assert not mask_locs.all(), f"Modality {mod} is completely empty. Remove it from the input."
                if mask_locs.sum().item() > 0:
                    sample_input[mod]['input_mask'][mask_locs] = True
                else:
                    assert mod not in target_mods, f"Target modality {mod} has been fully provided."

            # target_mask / decoder_attention_mask here are placeholders; the generation
            # strategy sets them per step.
            for target_mod in target_mods:
                if target_mod not in sample_input:
                    sample_input[target_mod] = {
                        'tensor': torch.zeros(target_len, dtype=torch.int),
                        'input_mask': torch.ones(target_len, dtype=torch.bool),
                        'target_mask': torch.ones(target_len, dtype=torch.bool),
                        'decoder_attention_mask': torch.zeros(target_len, dtype=torch.bool),
                    }
                else:
                    provided_len = len(sample_input[target_mod]['tensor'])
                    assert provided_len == target_len, \
                        f"Target length {target_len} does not match the length of the partial target modality provided {provided_len}"

            sample_input = fill_incomplete_mod_dict(data=sample_input, modality_info=self.modality_info)
            model_input_list.append(sample_input)

        return pad_and_collate(model_input_list, self.modality_info, self.sum_modality_groups)

    def _find_target_lens(self, target_mods: Set) -> list:
        """Infer per-sample generation length from a co-grouped input modality."""
        target_groups = {self.mod_group_lookup[mod] for mod in target_mods}
        assert len(target_groups) == 1, f"Target modalities {target_mods} are not in the same group. Groups: {target_groups}"
        target_group = target_groups.pop()

        if self.group_info[target_group]['max_seq_len'] == 1:
            return 1
        target_lens = []
        for i, tok_input in enumerate(self.tok_input):
            if (overlap := self.group_info[target_group]['mods'].intersection(tok_input)):
                target_lens.append(len(tok_input[overlap.pop()]))
            else:
                raise ValueError(f"Cannot determine generation length for sample {i}. Please provide target_lens.")
        return target_lens

    def embed(
        self,
        train_mode: bool = False,
        return_full: bool = True,
        return_register: bool = False,
        return_modality: bool = False,
    ) -> dict:
        """Encode the staged input into encoder representations.

        Returns a dict containing only the requested keys:

        - ``return_full`` (default True): ``'full'`` -- the batched encoder output
          ``[B, N, D]`` -- plus ``'mod_ids'`` ``[B, N]``, the per-position modality-group
          id for each token in ``'full'`` (``-1`` for padding).
        - ``return_register``: ``'register'`` -- the leading register-token slice
          ``[B, num_register_tokens, D]`` (omitted if the model has no register tokens).
        - ``return_modality``: ``'modality'`` -- a per-sample dict
          ``{sample_index: {group_name: tokens}}`` splitting ``'full'`` by modality group.
        """
        assert self.tok_input, "No input provided. Please provide an input via input() first."

        self.train(train_mode)
        mod_dict = self._create_model_input(self.tok_input, [], 0)
        with torch.set_grad_enabled(train_mode):
            output = self._model_forward(
                mod_dict, return_encoder_output=True, return_logits=False, return_loss=False
            )['encoder_output']

        encodings = {}
        if return_full:
            encodings['full'] = output['x']
            encodings['mod_ids'] = output['encoder_mod_mask']
        if return_register and self.num_register_tokens > 0:
            encodings['register'] = output['x'][:, :self.num_register_tokens]
        if return_modality:
            per_sample = {}
            for i in range(len(output['x'])):
                all_mod_idx = set(output['encoder_mod_mask'][i].tolist()) - {-1}
                per_sample[i] = {
                    self.group_names[idx]: output['x'][i][output['encoder_mod_mask'][i] == idx]
                    for idx in all_mod_idx
                }
            encodings['modality'] = per_sample

        return encodings

    def _resolve_strategy(self, strategy):
        """Resolve a strategy name (or None) to a generation-strategy instance.

        An instance is returned unchanged. Recognized names: "default"/"ensemble"
        (single-pass soft vote over all target tokens; deterministic at low temperature),
        "one_shot", and "autoregressive".
        """
        from . import strategies as strat
        if isinstance(strategy, strat.BaseGenerationStrategy):
            return strategy
        key = (strategy or "default").lower()
        if key in ("default", "ensemble"):
            return strat.EnsembleGenerationStrategy(
                num_tokens_per_step=self.num_target_tokens, num_passes=1, agg_mode="soft"
            )
        if key in ("one_shot", "oneshot"):
            return strat.OneShotGenerationStrategy()
        if key in ("autoregressive", "ar"):
            return strat.ARGenerationStrategy(num_tokens_per_step=1, sampling="sequential")
        raise ValueError(
            f"Unknown generation strategy {strategy!r}. Pass a BaseGenerationStrategy "
            f"instance or one of: 'default'/'ensemble', 'one_shot', 'autoregressive'."
        )

    def generate(
        self,
        target: Union[str, Sequence[str]],
        strategy=None,
        target_lens: list = None,
        temperature: float = 1e-8,
        seed: int = 42,
        verbose: bool = True,
        detokenize_mode: Literal["argmax", "weighted_mean"] = "argmax",
        detokenize_structure: bool = True,
        return_tokens: bool = False,
        return_logits: bool = False,
        return_probs: bool = False,
        return_sampling_probs: bool = False,
        on_unsupported: Literal["warn", "error", "allow"] = None,
    ) -> dict:
        """Generate one or more target modalities from the staged input.

        `strategy` may be a BaseGenerationStrategy instance or a name
        ("default"/"ensemble", "one_shot", "autoregressive"); default is Ensemble.

        Return shape:
          - By default, returns ``{target_name: preds}`` where ``preds`` is the
            detokenized generation for that modality (a string for sequence/text
            modalities, a float array for scalar tracks like sasa/phylop, a list
            of labels for categorical tracks, or a biotite ``AtomArray`` for
            ``prot_struct`` when ``detokenize_structure=True``).
          - If any of ``return_tokens/return_logits/return_probs/
            return_sampling_probs`` is True, every target's value becomes a dict
            ``{"preds": ..., <requested extras>}`` (as numpy arrays).

        ``detokenize_structure`` (default True) decodes ``prot_struct`` tokens to a
        backbone structure (a biotite ``AtomArray``) via the ESM3 VQVAE decoder. The
        decoder + its weights load/download on first use; pass
        ``detokenize_structure=False`` to skip this, leaving ``prot_struct`` preds
        as ``None`` (use ``return_tokens=True`` to still get the raw tokens).

        Pathway gating: cross-track generations MIMIC was not trained for (e.g.
        protein input -> is_coding) are refused. ``on_unsupported`` controls the
        response -- "error" (default: raise), "warn" (run with a loud warning), or
        "allow" (run silently). The package-wide default is `DEFAULT_ON_UNSUPPORTED`.
        Within-track and text-association pathways are allowed (see
        modality_info.PATHWAY_ALLOWLIST to enable additional cross-track pathways).
        """
        from .modality_info import unsupported_pathways, DEFAULT_ON_UNSUPPORTED

        torch.manual_seed(seed)
        assert self.tok_input, "No input provided. Please provide an input via input() first."

        strategy = self._resolve_strategy(strategy)
        target = [target] if isinstance(target, str) else target
        name_trans_back = {self._lookup_mod(mod): mod for mod in target}
        target_set = set(name_trans_back.keys())

        # Pathway-confidence gate: check every staged sample's inputs against targets.
        on_unsupported = on_unsupported or DEFAULT_ON_UNSUPPORTED
        if on_unsupported != "allow":
            input_mods = set().union(*(s.keys() for s in self.tok_input))
            bad = unsupported_pathways(input_mods, target_set)
            if bad:
                lines = "; ".join(f"{name_trans_back[m]}: {r}" for m, r in bad.items())
                msg = (
                    "Untrusted generation pathway -- MIMIC was not trained to generate "
                    f"these targets from the given conditioning ({lines}). To run it "
                    "anyway pass on_unsupported='allow' (or 'warn' to run with a warning)."
                )
                if on_unsupported == "error":
                    raise ValueError(msg)
                logger.warning(msg)

        target_lens = target_lens if target_lens else self._find_target_lens(target_set)
        target_lens = [target_lens] if isinstance(target_lens, int) else target_lens

        mod_dict = self._create_model_input(self.tok_input, target_set, target_lens)

        output = strategy.generate(
            model_forward=self._model_forward,
            mod_dict=mod_dict,
            modality_info=self.modality_info,
            target_set=target_set,
            max_model_target_tokens=self.num_target_tokens,
            temperature=temperature,
            is_target_autoregr=self.is_target_autoregr,
            verbose=verbose,
        )

        # Which raw strategy outputs to surface alongside the detokenized preds.
        want = {
            "tokens": return_tokens,
            "logits": return_logits,
            "probs": return_probs,
            "sampling_probs": return_sampling_probs,
        }
        any_extra = any(want.values())

        result = {}
        for mod, output_dict in output.items():
            tokens_np = output_dict["tokens"].cpu().numpy()
            tok = self.tokenizers[mod]
            # Structure detok (ESM3 VQVAE decode -> coordinates) loads/downloads the
            # decoder on first use; on by default, disable via detokenize_structure=False.
            if mod == "tok_prot_struct":
                preds = tok.detokenize(tokens_np.tolist(), as_biotite=True) if detokenize_structure else None
            elif detokenize_mode == "weighted_mean":
                preds = tok.detokenize(tokens_np, probs=output_dict["probs"].cpu().type(torch.float32).numpy())
            else:
                preds = tok.detokenize(tokens_np)

            name = name_trans_back[mod]
            if not any_extra:
                # Default: hand back just the detokenized generation.
                result[name] = preds
                continue

            # Rich return: preds plus each explicitly requested extra, as numpy.
            entry = {"preds": preds}
            for key, requested in want.items():
                if not requested:
                    continue
                val = output_dict.get(key)
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy() if key == "tokens" else val.cpu().type(torch.float32).numpy()
                elif isinstance(val, list):
                    val = np.array(val)
                entry[key] = val
            result[name] = entry

        return result

    def _model_forward(self, mod_dict: dict, return_encoder_output: bool,
                       return_logits: bool, return_loss: bool) -> dict:
        """Move a mod_dict to the model device and run forward under inference autocast."""
        device = self.device
        with torch.autocast(
            device_type=device.type,
            dtype=self.autocast_dtype,
            enabled=(self.autocast_dtype != torch.float32 and device.type == "cuda"),
        ):
            mod_dict = {
                modality: {k: v.to(device, non_blocking=True) for k, v in d.items()}
                for modality, d in mod_dict.items()
            }
            output = self(
                mod_dict,
                num_encoder_tokens=self.num_input_tokens + self.num_register_tokens,
                num_decoder_tokens=self.num_target_tokens,
                return_encoder_output=return_encoder_output,
                return_logits=return_logits,
                return_loss=return_loss,
            )
        return output
