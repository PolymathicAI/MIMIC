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
from loguru import logger
import torch
import torch.nn as nn
from einops import repeat

# BioBERT dimension for the text token embedding (dmis-lab/biobert-base-cased-v1.2).
# The text embedding matrix is [text_vocab_size, LLM_EMB_DIM], projected to the model dim.
LLM_EMB_DIM = 768

# Module-level shared token embedding used when `_SHARED_TEXT_TOKEN_EMB` is True
# Embedding Block will be shared between encoder AND decoder 
# This approach vs others will work even if only we want to load decoder and for instance finetune
# Avois having to create an extra class where SequenceEncoder/DecoderEmbedding is inherited just to share the embedding
_SHARED_TEXT_TOKEN_EMB: Optional[nn.Module] = None


def create_biobert_token_embedding(
    biobert_model: str,
    dim_tokens: int,
    freeze_llm_emb: bool = True,
) -> nn.Module:
    """
    Create a shared BioBERT-based token embedding module with projection and normalization.

    This function creates a sequential module that:
    1. Loads BioBERT pretrained embeddings
    2. Extends them with special tokens (PAD, MASK, UNK)
    3. Projects from BioBERT dimension to target dimension
    4. Applies layer normalization

    IMPORTANT: For FSDP compatibility, embeddings are created UNFROZEN regardless of
    freeze_llm_emb parameter. Actual freezing happens after optimizer creation in
    train_eva.py to ensure parameters are included in optimizer state before freezing.

    Args:
        biobert_model: HuggingFace model identifier for BioBERT
        dim_tokens: Target embedding dimension
        freeze_llm_emb: DEPRECATED - kept for API compatibility but ignored.
                       Freezing is handled in train_eva.py after optimizer creation.
    Returns:
        nn.Sequential module containing [llm_token_emb, projection, layer_norm]
    """
    # Load BioBERT pretrained embeddings. `transformers` is imported lazily here
    # because it is only needed when initializing text embeddings from BioBERT
    # (i.e. for training / re-init). Inference loads these weights from the
    # checkpoint via `create_empty_text_token_embedding` instead.
    from transformers import AutoModel, AutoTokenizer

    biobert = AutoModel.from_pretrained(biobert_model)
    biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_model)

    pretrained_embeddings = biobert.embeddings.word_embeddings.weight.data.clone()
    llm_dim = pretrained_embeddings.shape[1]
    biobert_pad_id = biobert_tokenizer.pad_token_id
    biobert_mask_id = biobert_tokenizer.mask_token_id
    biobert_unk_id = biobert_tokenizer.unk_token_id

    # Get their respective biobert embeddings
    pad_embedding = pretrained_embeddings[biobert_pad_id].clone()
    mask_embedding = pretrained_embeddings[biobert_mask_id].clone()
    unk_embedding = pretrained_embeddings[biobert_unk_id].clone()

    # Concat original embeddings with the native special tokens embeddings at the end
    extended_embeddings = torch.cat([
        pretrained_embeddings,           # Original 28996 rows of embeddings
        pad_embedding.unsqueeze(0),      # Row 28996 biobert PAD
        mask_embedding.unsqueeze(0),     # Row 28997 biobert MASK
        unk_embedding.unsqueeze(0),      # Row 28998 biobert UNK
    ], dim=0)

    # Final embedding matrix
    # IMPORTANT: Always create unfrozen for FSDP compatibility
    # Freezing will be done AFTER optimizer creation in train_eva.py
    llm_token_emb = nn.Embedding.from_pretrained(
        embeddings=extended_embeddings,
        freeze=False,  # Changed from freeze=freeze_llm_emb
    )

    # Free memory
    del biobert

    # Add projection layer to match dimensions and layer norm to
    # avoid norm of LLM embeddings dominating gradient signal
    projection = nn.Linear(llm_dim, dim_tokens)
    layer_norm = nn.LayerNorm(dim_tokens, elementwise_affine=True)

    # Initialize projection to identity and bias to zero
    nn.init.eye_(projection.weight)
    nn.init.constant_(projection.bias, 0.0)

    # Return as sequential module
    return nn.Sequential(llm_token_emb, projection, layer_norm)


def create_empty_text_token_embedding(
    vocab_size: int,
    dim_tokens: int,
    llm_dim: int = LLM_EMB_DIM,
) -> nn.Module:
    """Build the text token embedding with the SAME structure as
    `create_biobert_token_embedding` but WITHOUT loading BioBERT.

    Used at inference: the (trained) weights are loaded from the checkpoint, so
    there is no need to pull `transformers`/BioBERT just to allocate the module.
    Structure: [Embedding(vocab_size, llm_dim), Linear(llm_dim, dim_tokens),
    LayerNorm(dim_tokens)].
    """
    llm_token_emb = nn.Embedding(vocab_size, llm_dim)
    projection = nn.Linear(llm_dim, dim_tokens)
    layer_norm = nn.LayerNorm(dim_tokens, elementwise_affine=True)
    nn.init.eye_(projection.weight)
    nn.init.constant_(projection.bias, 0.0)
    return nn.Sequential(llm_token_emb, projection, layer_norm)


class SequenceEncoderEmbedding(nn.Module):
    """Embedding module for encoding sequence inputs, like captions or a sequence of objects.

    Args:
        vocab_size: Vocabulary size
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        position_code: Position embedding type ('rotary', 'sincos', 'learnable').
                      If 'rotary', no position embeddings are created.
        max_sincos_pos_emb: Maximum allowed length for sin-cos positional embeddings
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
        mask_token_id: int = 1,
        biobert_model: str = 'dmis-lab/biobert-base-cased-v1.2',
        freeze_llm_emb: bool = True,
        init_text_from_biobert: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.position_code = position_code
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.max_sincos_pos_emb = max_sincos_pos_emb
        self.biobert_model = biobert_model
        self.freeze_llm_emb = freeze_llm_emb
        # When False (inference default), the text embedding is allocated empty and
        # filled from the checkpoint; when True (training), it is seeded from BioBERT.
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

        # mimic 1.0 uses rotary position embeddings (applied inside the transformer
        # attention), so no positional embedding buffer is created here.
        if self.position_code != 'rotary':
            raise ValueError(
                f"mimic ships rotary-only embeddings; got position_code={self.position_code!r}."
            )
        self.pos_emb = None

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding: shared module-level embedding across text modalities and
        # across encoder/decoder. Seed from BioBERT only when explicitly requested
        # (training); otherwise allocate empty and fill from the checkpoint.
        global _SHARED_TEXT_TOKEN_EMB
        if _SHARED_TEXT_TOKEN_EMB is None:
            if self.init_text_from_biobert:
                _SHARED_TEXT_TOKEN_EMB = create_biobert_token_embedding(
                    biobert_model=self.biobert_model,
                    dim_tokens=self.dim_tokens,
                    freeze_llm_emb=self.freeze_llm_emb,
                )
            else:
                _SHARED_TEXT_TOKEN_EMB = create_empty_text_token_embedding(
                    vocab_size=self.vocab_size,
                    dim_tokens=self.dim_tokens,
                )
        self.text_token_emb = _SHARED_TEXT_TOKEN_EMB

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming sequence of ids to sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict with at least the following keys:
                - 'tensor' (torch.Tensor): Input token sequence for each batch. Shape (B, L) where B is the batch size and L is the sequence length.
                - 'input_mask' (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, L, D).
        """
        ids = d["tensor"]
        B = ids.shape[0]
        assert (
            self.dim_tokens is not None
        ), "Need to call init(dim_tokens) function first"

        # make a copy of the ids where the places here input_mask == True are replaced by the mask_token_id
        ids = ids.masked_fill(d["input_mask"], self.mask_token_id)

        # Map to embedding
        x = self.text_token_emb(ids)

        if self.pos_emb is None:
            # Rotary positional embeddings are handled within the transformer modules
            x_emb = repeat(self.mod_emb, "1 1 d -> b n d", b=B, n=ids.shape[1])
        else:
            # Use positional embeddings (sincos or learnable)
            expanded_pos_emb = repeat(self.pos_emb, "1 n d -> b n d", b=B)
            # Input pos encoding
            input_mask = d["input_mask"]
            input_pos_id = (~input_mask).int().cumsum(dim=1) - 1
            input_pos_id[input_mask] = 0
            input_pos_emb = torch.gather(
                expanded_pos_emb,
                dim=1,
                index=repeat(input_pos_id, "b n -> b n d", d=expanded_pos_emb.shape[2]),
            )
            input_pos_emb[input_mask] = 0
            x_emb = input_pos_emb + self.mod_emb

        d["x"] = x
        d["emb"] = x_emb
        return d


class ChainTokenEncoderEmbedding(nn.Module):
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
        mask_token_id: int = 1,
    ):

        super().__init__()
        self.vocab_size = vocab_size
        self.dim_tokens = dim_tokens
        self.position_code = position_code
        self.max_chain_len = max_length
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

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

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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

        # make a copy of the ids where the places here input_mask == True are replaced by the mask_token_id
        ids = ids.masked_fill(d["input_mask"], self.mask_token_id)

        # Map to embedding
        x = self.token_emb(ids)

        if self.pos_emb is None:
            # Rotary positional embeddings are handled within the transformer modules
            x_emb = repeat(self.mod_emb, "1 1 d -> b n d", b=B, n=T)
        else:
            # Use positional embeddings (sincos or learnable)
            expanded_pos_emb = repeat(self.pos_emb, "1 n d -> b n d", b=B)
            # Input pos encoding
            input_mask = d["input_mask"]
            input_pos_id = (~input_mask).int().cumsum(dim=1) - 1
            input_pos_id[input_mask] = 0
            input_pos_emb = torch.gather(
                expanded_pos_emb,
                dim=1,
                index=repeat(input_pos_id, "b n -> b n d", d=expanded_pos_emb.shape[2]),
            )
            input_pos_emb[input_mask] = 0
            x_emb = input_pos_emb + self.mod_emb

        d["x"] = x
        d["emb"] = x_emb

        return d
