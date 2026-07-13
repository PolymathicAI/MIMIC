import torch
import torch.nn as nn
from x_transformers.x_transformers import AttentionLayers
import re

from .base import EvaDecoder, EvaEncoder
from .registries import register_encoder, register_decoder

def create_mixed_attention_mask(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    device: torch.device,
    unidir_attention_ratio: float = 0,
    num_register_tokens: int = 0
) -> torch.Tensor:
    """
    Creates mixed attention masks for different head patterns.

    Args:
        batch_size: Batch size.
        num_heads: Number of attention heads.
        seq_len: Sequence length.
        device: Device to create tensors on.
        unidir_attention_ratio: Fraction of heads with unidirectional attention.
        num_register_tokens: Number of register tokens at the beginning of the sequence.

    Returns:
        Attention mask tensor of shape [batch_size, num_heads, seq_len, seq_len]
        where False means positions to mask (ignore).
    """
    # Ensure ratio is valid
    if not (0 <= unidir_attention_ratio <= 1):
        raise ValueError("unidir_attention_ratio must be between 0 and 1")

    # Define the core attention mask (all True, meaning attend everywhere)
    mask = torch.ones(num_heads, seq_len, seq_len, dtype=torch.bool, device=device)

    # Calculate number of heads for each pattern
    num_left_only = num_right_only = int(num_heads * unidir_attention_ratio * 0.5)  # Half of unidirectional heads use causal (attend to past) / anti-causal (attend to future) attention look left/right
    
    # Apply left-only (causal) masking
    if num_left_only > 0:
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        mask[:num_left_only, :, :] = causal_mask

    # Apply right-only masking
    if num_right_only > 0:
        right_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        mask[num_left_only:num_left_only + num_right_only, :, :] = right_mask

    if num_register_tokens > 0:
        # Register tokens attend to all other tokens
        mask[:, :num_register_tokens, :] = True
        # All tokens attend to register tokens
        mask[:, :, :num_register_tokens] = True

    # Expand for batch dimension
    batch_mask = mask.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return batch_mask

class XTransformerEncoder(EvaEncoder):
    def __init__(self, 
                 dim: int, 
                 depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0, # Called ff_mult in x_transformers
                 mlp_bias: bool = True, # Can't individually disable qkv or proj bias in x_transformers
                 drop_path_rate: float = 0.0, # Slightly different implementation than 4m - constant rate for all layers
                 act_layer: nn.Module = nn.SiLU,
                 gated_mlp: bool = False, # Enabled as ff_glu
                 qk_norm: bool = False,
                 attn_flash: bool = True,
                 rotary_pos_emb: bool = True,
                 rotary_emb_ratio: float = 0.5,
                 use_alibi: bool = False,
                 causal: bool = False,
                 # Mixed attention pattern parameters
                 mixed_attention: bool = False,
                 unidir_attention_ratio: float = 0,
                 num_register_tokens: int = 0,
                #  rotary_cross_attn: bool = False # Dummy parameter for compatibility
                 ): 
        super().__init__(dim)
        
        # Store mixed attention parameters
        self.mixed_attention = mixed_attention
        self.unidir_attention_ratio = unidir_attention_ratio
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens

        self.torso = AttentionLayers(
            dim = dim,
            depth = depth,
            heads = num_heads,
            ff_mult = mlp_ratio,
            ff_no_bias = not mlp_bias,
            layer_dropout = drop_path_rate,
            ff_glu = gated_mlp,
            ff_swish = act_layer == nn.SiLU,
            attn_qk_norm = qk_norm,
            attn_flash = attn_flash,
            rotary_pos_emb = rotary_pos_emb,
            rotary_emb_dim = int(dim / num_heads * rotary_emb_ratio) if rotary_pos_emb else None,
            alibi_pos_bias = use_alibi,
            cross_attend = False,
            causal = causal,
        )


    def forward(self, x: torch.Tensor, encoder_mask: torch.Tensor, input_pos_idx: torch.Tensor) -> torch.Tensor:
        # During processing, any place where encoder_mask is equal to 1 is dropped,
        # so in general there should be no place where encoder_mask is equal to 1.
        # The only exception is when batches have different lengths of non-masked tokens,
        # in which case the padding will have encoder_mask = True.
                
        # x_transformers expects mask where True = valid tokens, False = tokens to be ignored
        mask = ~encoder_mask.squeeze(1).bool()
        
        # Create mixed attention mask if enabled
        attn_mask = None
        if self.mixed_attention:
            batch_size, seq_len = x.shape[:2]
            attn_mask = create_mixed_attention_mask(
                batch_size=batch_size,
                num_heads=self.num_heads,
                seq_len=seq_len,
                device=x.device,
                unidir_attention_ratio=self.unidir_attention_ratio,
                num_register_tokens=self.num_register_tokens
            )
        
        return self.torso(x, mask=mask, attn_mask=attn_mask, pos=input_pos_idx)

    def get_num_layers(self) -> int:
        return len(self.torso.layers) // 2 # 2 layers per block (self-attn, mlp)
    
    def get_block_types(self) -> set:
        """Return the set of block types used by this encoder for FSDP and activation checkpointing."""
        return {AttentionLayers}


class XTransformerDecoder(EvaDecoder):
    """XTransformer decoder implementation using attention blocks."""
    
    def __init__(self, 
                 dim: int, 
                 depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 mlp_bias: bool = True,
                 drop_path_rate: float = 0.0,
                 act_layer: nn.Module = nn.SiLU,
                 gated_mlp: bool = False,
                 qk_norm: bool = False,
                 attn_flash: bool = True,
                 rotary_pos_emb: bool = True,
                 rotary_emb_ratio: float = 0.5,
                 rotary_cross_attn: bool = False,
                 use_alibi: bool = False,
                 causal: bool = False,
                 # Mixed attention pattern parameters (for self-attention)
                 mixed_attention: bool = False,
                 unidir_attention_ratio: float = 0,
                 ):
        super().__init__(dim)
        
        # Store mixed attention parameters
        self.mixed_attention = mixed_attention
        self.unidir_attention_ratio = unidir_attention_ratio
        self.num_heads = num_heads
                
        self.torso = AttentionLayers(
            dim = dim,
            depth = depth,
            heads = num_heads,
            ff_mult = mlp_ratio,
            ff_no_bias = not mlp_bias,
            layer_dropout = drop_path_rate,
            ff_glu = gated_mlp,
            ff_swish = act_layer == nn.SiLU,
            attn_qk_norm = qk_norm,
            attn_flash = attn_flash,
            rotary_pos_emb = rotary_pos_emb,
            rotary_emb_dim = int(dim / num_heads * rotary_emb_ratio) if rotary_pos_emb else None,
            alibi_pos_bias = use_alibi,
            cross_attend = True,
            causal = causal
        )

        self.rotary_cross_attn = rotary_cross_attn
            
    def forward(self, y: torch.Tensor, context: torch.Tensor, 
                encoder_mask: torch.Tensor, decoder_attention_mask: torch.Tensor,
                decoder_mask: torch.Tensor = None, input_pos_idx: torch.Tensor = None,
                target_pos_idx: torch.Tensor = None) -> torch.Tensor:

        # x_transformers expects mask where True = valid tokens, False = tokens to be ignored
        # self attention mask
        sa_mask = ~decoder_attention_mask.unsqueeze(1)  # [B, 1, M, M]

        # Build cross-attention mask (encoder pad)
        context_mask = ~encoder_mask.squeeze(1).bool()                      # [B, N], True = valid tokens
        
        # Create mixed attention mask for self-attention if enabled
        if self.mixed_attention:
            batch_size, seq_len = y.shape[:2]
            mixed_attn_mask = create_mixed_attention_mask(
                batch_size=batch_size,
                num_heads=self.num_heads,
                seq_len=seq_len,
                device=y.device,
                unidir_attention_ratio=self.unidir_attention_ratio,
                num_register_tokens=0 # The register tokens are only inserted in the encoder
            )
            # Combine with existing self-attention mask
            # sa_mask is [B, 1, M, M], mixed_attn_mask is [B, H, M, M]
            # Broadcast sa_mask to [B, H, M, M] and combine with OR (True = mask)
            sa_mask_expanded = sa_mask.expand(-1, self.num_heads, -1, -1)
            sa_mask = sa_mask_expanded & mixed_attn_mask
        
        # Decode with proper masks. Input is now context and target is y
        y = self.torso(
            y,
            context=context,
            attn_mask=sa_mask,
            context_mask=context_mask,
            pos=target_pos_idx,
            context_pos=input_pos_idx
        )

        return y    

    def get_num_layers(self) -> int:
        return len(self.torso.layers) // 3 # 3 layers per block (self-attn, cross-attn, mlp)
    
    def get_block_types(self) -> set:
        """Return the set of block types used by this decoder for FSDP and activation checkpointing."""
        return {AttentionLayers}


def _parse_model_name(name: str) -> tuple[str, int, int]:
    """Parse model name like 'xt_encoder_6L_384D' into (model_type, layers, dim)."""
    # Extract model type (encoder/decoder), layers, and dimension
    pattern = r'xt_(encoder|decoder)_(\d+)L_(\d+)D'
    match = re.match(pattern, name)
    if not match:
        raise ValueError(f"Invalid model name format: {name}. Expected format: xt_[encoder|decoder]_<layers>L_<dim>D")
    
    model_type = match.group(1)
    layers = int(match.group(2))
    dim = int(match.group(3))
    
    return model_type, layers, dim

def _create_xtransformer_model(name: str, **kwargs):
    """Factory function to create XTransformer models based on name parsing."""
    model_type, layers, dim = _parse_model_name(name)
    
    # Pass the parsed parameters and any kwargs to the constructors
    # Use kwargs values if provided, otherwise use the parsed values
    if 'dim' in kwargs:
        dim = kwargs.pop('dim')
        
    if 'depth' in kwargs:
        layers = kwargs.pop('depth')

    # Calculate num_heads only if not specified in kwargs
    if 'num_heads' not in kwargs:
        kwargs['num_heads'] = max(1, dim // 64)
    
    # Create the appropriate model based on type
    if model_type == 'encoder':
        return XTransformerEncoder(dim=dim, depth=layers, **kwargs)
    elif model_type == 'decoder':
        return XTransformerDecoder(dim=dim, depth=layers, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Register all encoder models
def register_xt_encoders():
    """Register all XTransformer encoder models."""
    model_configs = [
        "xt_encoder_1L_8D", # smallest size for testing
        "xt_encoder_2L_32D", # Small for gittest
        "xt_encoder_6L_384D", # 10.63M
        "xt_encoder_8L_512D", # 25.20M
        "xt_encoder_10L_640D", # 49.20M
        "xt_encoder_12L_768D", # 85.00M
        "xt_encoder_16L_1536D", # 453.16M
        "xt_encoder_20L_1536D",
        "xt_encoder_20L_2048D",
        "xt_encoder_16L_1024D",
        "xt_encoder_24L_1024D",
        "xt_encoder_24L_1536D",
        "xt_encoder_24L_2048D"
    ]
    
    for name in model_configs:
        def create_encoder(name=name, **kwargs):
            return _create_xtransformer_model(name, **kwargs)
        
        # Register with the appropriate decorator
        register_encoder(name)(create_encoder)

def register_xt_decoders():
    """Register all XTransformer decoder models."""
    model_configs = [
        "xt_decoder_1L_8D", # smallest size for testing
        "xt_decoder_2L_32D",
        "xt_decoder_3L_32D",
        "xt_decoder_3L_48D", # Small for gittest
        "xt_decoder_6L_384D", # 14.17M
        "xt_decoder_8L_512D", # 33.59M
        "xt_decoder_10L_640D", # 65.59M
        "xt_decoder_12L_768D", # 113.32M
        "xt_decoder_12L_1536D",
        "xt_decoder_16L_1536D", # 604.18M
        "xt_decoder_16L_2048D", # 604.18M
        "xt_decoder_24L_1024D",
        "xt_decoder_24L_2048D"
    ]
    
    for name in model_configs:
        def create_decoder(name=name, **kwargs):
            return _create_xtransformer_model(name, **kwargs)
        
        # Register with the appropriate decorator
        register_decoder(name)(create_decoder)

# Call the registration functions
register_xt_encoders()
register_xt_decoders()