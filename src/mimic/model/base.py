import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class EvaEncoder(nn.Module, ABC):
    """Abstract base class for encoder architectures.
    
    All encoder implementations must inherit from this class and implement
    the forward method with the specified signature.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor, encoder_mask: torch.Tensor, input_pos_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass for the encoder.
        
        Args:
            x (torch.Tensor): Encoder input tokens. Shape (B, N, D) where N is the number of encoder tokens.
            encoder_mask (torch.Tensor): Encoder mask indicating which tokens are valid (set to 0 for valid tokens, 1 otherwise). Shape (B, 1, N)
            
        Returns:
            torch.Tensor: Encoder output. Shape (B, N, D)
        """
        pass

    @abstractmethod
    def get_num_layers(self) -> int:
        """Return the number of layers in the encoder."""
        return len(self.encoder) if hasattr(self, 'encoder') else 0
    
    @abstractmethod
    def get_block_types(self) -> set:
        """Return the set of block types used by this encoder/decoder for FSDP and activation checkpointing."""
        pass

    def freeze(self):
        """Freeze the encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze the encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True

class EvaDecoder(nn.Module, ABC):
    """Abstract base class for decoder architectures.

    All decoder implementations must inherit from this class and implement
    the forward method with the specified signature.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @abstractmethod
    def forward(self, y: torch.Tensor, context: torch.Tensor, encoder_mask: torch.Tensor,
                    decoder_attention_mask: torch.Tensor, decoder_mask: torch.Tensor = None,
                     input_pos_idx:torch.Tensor = None, target_pos_idx: torch.Tensor = None, **kwargs) -> torch.Tensor:
            """Forward pass for the decoder.

            Args:
                y (torch.Tensor): Decoder input tokens. Shape (B, M, D) where M is the number of decoder tokens.
                context (torch.Tensor): Context from the encoder. Shape (B, N, D) where N is the number of encoder tokens.
                encoder_mask (torch.Tensor): Encoder mask indicating which tokens are valid (set to 0 for valid tokens, 1 otherwise). Shape (B, 1, N).
                decoder_attention_mask (torch.Tensor): Self-attention mask for the decoder. Shape (B, M, M).
                decoder_mask (torch.Tensor): Decoder mask indicating which tokens are valid (set to 0 for valid tokens, 1 otherwise). Shape (B, 1, M).
                **kwargs: Additional keyword arguments.

            Returns:
                torch.Tensor: Decoder output. Shape (B, M, D)
            """
            pass

    def get_num_layers(self) -> int:
        """Return the number of layers in the decoder."""
        return len(self.decoder) if hasattr(self, 'decoder') else 0
    
    @abstractmethod
    def get_block_types(self) -> set:
        """Return the set of block types used by this encoder/decoder for FSDP and activation checkpointing."""
        pass

    def freeze(self):
        """Freeze the decoder parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self, freeze_embeddings=True):
        """Unfreeze the decoder parameters."""
        for param in self.parameters():
            param.requires_grad = True

