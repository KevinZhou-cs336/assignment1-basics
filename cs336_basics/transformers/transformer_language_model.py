import torch

from cs336_basics.transformers.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.transformers.positionwise_feedforward import SwiGLUFeedForwardNetwork
from cs336_basics.transformers.rms_norm import RMSNorm
from cs336_basics.transformers.rope import RotaryPositionalEmbedding
from cs336_basics.transformers.transformer_block import TransformerBlock


class TransformerLanguage(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
    ):
        # transformer lanaguage moduel parameters
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        # transformer block parameters

        self.blocks = torch.nn.ModuleList(
            [TransformerBlock() for _ in range(num_layers)]
        )
