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
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from einops import repeat
from . import encoder_embeddings
from .encoder_embeddings import (
    create_biobert_token_embedding,
    create_empty_text_token_embedding,
)


class SequenceDecoderEmbedding(nn.Module):
    """Embedding module for sequence inputs, like captions or a sequence of objects.

    Args:
        vocab_size: Vocabulary size
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        position_code: Position embedding type ('rotary', 'sincos', 'learnable').
                      If 'rotary', no position embeddings are created.
        pad_token_id: Padding index for word embedding
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        dim_tokens: Optional[int] = None,
        position_code: str = 'sincos',
        max_sincos_pos_emb: int = 512,
        pad_token_id: int = 0,
        biobert_model: str = 'dmis-lab/biobert-base-cased-v1.2',
        freeze_llm_emb: bool = True,
        init_text_from_biobert: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.position_code = position_code
        self.pad_token_id = pad_token_id
        self.max_sincos_pos_emb = max_sincos_pos_emb
        self.biobert_model = biobert_model
        self.freeze_llm_emb = freeze_llm_emb
        self.init_text_from_biobert = init_text_from_biobert

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)


    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of embedding module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes

        # mimic 1.0 uses rotary position embeddings (applied inside the transformer
        # attention), so no positional embedding buffer is created here.
        if self.position_code != 'rotary':
            raise ValueError(
                f"mimic ships rotary-only embeddings; got position_code={self.position_code!r}."
            )
        self.pos_emb = None

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding - shared with encoder. Seed from BioBERT only when requested
        # (training); otherwise allocate empty and fill from the checkpoint.
        if encoder_embeddings._SHARED_TEXT_TOKEN_EMB is None:
            if self.init_text_from_biobert:
                encoder_embeddings._SHARED_TEXT_TOKEN_EMB = create_biobert_token_embedding(
                    biobert_model=self.biobert_model,
                    dim_tokens=self.dim_tokens,
                    freeze_llm_emb=self.freeze_llm_emb,
                )
            else:
                encoder_embeddings._SHARED_TEXT_TOKEN_EMB = create_empty_text_token_embedding(
                    vocab_size=self.vocab_size,
                    dim_tokens=self.dim_tokens,
                )
        self.text_token_emb = encoder_embeddings._SHARED_TEXT_TOKEN_EMB


        # Output projection layer
        self.to_logits = nn.Linear(self.dim_tokens, self.vocab_size, bias=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward_embed(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming sequence of ids to sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict, with at least the following keys:
                - 'tensor' (torch.Tensor): Token sequence for each batch. Shape (B, L) where B is the batch size and L is the sequence length.
                - 'target_mask' (torch.Tensor): Mask for valid tokens in the target sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the target sequence. Shape (B, L, D).
                - 'ids' (torch.Tensor): Original token sequence from input dict. Shape (B, L).
        """
        ids = d["tensor"]
        B = ids.shape[0]
        assert (
            self.dim_tokens is not None
        ), "Need to call init(dim_tokens) function first"

        # Map to embedding
        x = self.text_token_emb(ids)

        if self.pos_emb is None:
            # Rotary positional embeddings are handled within the transformer modules
            x_emb = repeat(self.mod_emb, "1 1 d -> b n d", b=B, n=ids.shape[1])
        else:
            # Use positional embeddings (sincos or learnable)
            expanded_pos_emb = repeat(self.pos_emb, "1 n d -> b n d", b=B)

            # Target pos encoding
            target_mask = d["target_mask"]
            target_pos_id = (~target_mask).int().cumsum(dim=1) - 1
            target_pos_id[target_mask] = 0
            # Sometimes target sequence is over max length, it will be truncated in decoder
            target_pos_id[target_pos_id >= self.max_length] = 0
            target_pos_emb = torch.gather(
                expanded_pos_emb,
                dim=1,
                index=repeat(target_pos_id, "b n -> b n d", d=expanded_pos_emb.shape[2]),
            )
            target_pos_emb[target_mask] = 0

            x_emb = target_pos_emb + self.mod_emb

        d["x"] = x
        d["emb"] = x_emb
        d["ids"] = d["tensor"]

        return d

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through output projection layer, transforming sequence of embeddings to logits.

        Args:
            x (torch.Tensor): Output tokens from the decoder. Shape (B, M, D)

        Returns:
            torch.Tensor: Logits for each token in the sequence. Shape (B, M, V)
        """
        logits = self.to_logits(x)
        return logits


class ChainTokenDecoderEmbedding(nn.Module):
    """Embedding module for tokenized chain inputs.

    Args:
        vocab_size: Vocabulary size
        dim_tokens: Dimension of output tokens. Can be set using init method.
        position_code: Position embedding type ('rotary', 'sincos', 'learnable'). 
                      If 'rotary', no position embeddings are created.
        max_length: Maximum chain size. Used to initialize size of positional embeddings.
        pad_token_id: Padding index for the embedding. This is initialized to all zeros without
            gradient so it is easily ignored by attention.
    """

    def __init__(
        self,
        vocab_size: int,
        dim_tokens: Optional[int] = None,
        position_code: str = 'sincos',
        max_length: Union[int] = 1024,
        pad_token_id: int = 0,
    ):

        super().__init__()
        self.vocab_size = vocab_size
        self.dim_tokens = dim_tokens
        self.position_code = position_code
        self.max_length = max_length
        self.pad_token_id = pad_token_id

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # mimic 1.0 uses rotary position embeddings (applied inside the transformer
        # attention), so no positional embedding buffer is created here.
        if self.position_code != 'rotary':
            raise ValueError(
                f"mimic ships rotary-only embeddings; got position_code={self.position_code!r}."
            )
        self.pos_emb = None

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding
        self.token_emb = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.dim_tokens,
            padding_idx=self.pad_token_id,
        )

        # Output projection layer
        self.to_logits = nn.Linear(self.dim_tokens, self.vocab_size, bias=False)

        # NOTE: sharing of embeddings with the decoder to populate .to_logits() has been deprecated

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward_embed(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming chain tokens to a sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict with at least the following key:
                - 'tensor' (torch.Tensor): Input chain tokens for each batch. Shape (B, L) where B is the batch size, and L is the chain length.
                - 'input_mask' (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            Dict[str, torch.Tensor]: Modality dictionary with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, H*W, D).
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, H*W, D).
        """
        ids = d["tensor"]
        B, T = ids.shape
        assert (
            self.dim_tokens is not None
        ), "Need to call init(dim_tokens) function first"

        # Map to embedding
        x = self.token_emb(ids)

        if self.pos_emb is None:
            # Rotary positional embeddings are handled within the transformer modules
            x_emb = repeat(self.mod_emb, "1 1 d -> b n d", b=B, n=T)
        else:
            # Use positional embeddings (sincos or learnable)
            x_emb = repeat(self.pos_emb[:, :T] + self.mod_emb, "1 n d -> b n d", b=B)

        d["x"] = x
        d["emb"] = x_emb
        d["ids"] = ids

        return d

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through output projection layer, transforming sequence of embeddings to logits.

        Args:
            x (torch.Tensor): Output tokens from the decoder. Shape (B, M, D)

        Returns:
            torch.Tensor: Logits for each token in the sequence. Shape (B, M, V)
        """
        logits = self.to_logits(x)
        return logits
