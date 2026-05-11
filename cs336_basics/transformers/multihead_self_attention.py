import torch

from cs336_basics.transformers.functions import scaled_dot_product_attention
from cs336_basics.transformers.rope import RotaryPositionalEmbedding


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding = None
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # d_k = d_v = d_model / num_heads
        self.rope = rope

    def forward(
        self,
        q_weight: torch.Tensor,  # (num_heads * d_head, d_model) = (d_model, d_model)
        k_weight: torch.Tensor,  # (num_heads * d_head, d_model) = (d_model, d_model)
        v_weight: torch.Tensor,  # (num_heads * d_head, d_model) = (d_model, d_model)
        o_weight: torch.Tensor,  # (d_model, num_heads * d_head) = (d_model, d_model)
        in_features: torch.Tensor,  # (..., seq_len, d_model)
    ) -> torch.Tensor:  # (..., seq_len, d_model)
        # Reshape flat projection weights into per-head view
        # (d_model, d_model) -> (num_heads, d_head, d_model)
        q_weights = torch.reshape(q_weight, (self.num_heads, self.d_head, self.d_model))
        k_weights = torch.reshape(k_weight, (self.num_heads, self.d_head, self.d_model))
        v_weights = torch.reshape(v_weight, (self.num_heads, self.d_head, self.d_model))

        # Project input into Q, K, V for all heads simultaneously
        # in_features: (..., seq_len, d_model)  x: ...xj
        # q_weights:   (num_heads, d_head, d_model)  nhj
        # q_x:         (..., num_heads, seq_len, d_head)  ...nxh
        q_x = torch.einsum("...xj, nhj->...nxh", in_features, q_weights)
        k_x = torch.einsum("...xj, nhj->...nxh", in_features, k_weights)
        v_x = torch.einsum("...xj, nhj->...nxh", in_features, v_weights)

        # Apply RoPE to Q and K (not V); token_position: (seq_len,)
        # RoPE treats num_heads as a batch dimension, shape preserved: (..., num_heads, seq_len, d_head)
        token_position = torch.arange(in_features.shape[-2])
        if self.rope:
            q_x = self.rope.forward(q_x, token_position)
            k_x = self.rope.forward(k_x, token_position)

        # Causal mask: lower-triangular boolean matrix (seq_len, seq_len)
        # mask[i, j] = True means query i can attend to key j
        mask = torch.tril(
            torch.ones(in_features.shape[-2], in_features.shape[-2])
        ).bool()

        # Scaled dot-product attention, applied independently per head
        # Input/output: (..., num_heads, seq_len, d_head)
        multihead_attentions = scaled_dot_product_attention(q_x, k_x, v_x, mask)

        # Rearrange from (..., num_heads, seq_len, d_head)
        #             to (..., seq_len, num_heads, d_head)
        multihead_attentions = multihead_attentions.transpose(-2, -3)

        # Concatenate heads: (..., seq_len, num_heads, d_head) -> (..., seq_len, d_model)
        attentions = multihead_attentions.reshape(
            (*multihead_attentions.shape[:-2], -1)
        )

        # Output projection: W_O applied to concatenated heads
        # o_weight: (d_model, d_model) stored as (out, in) -> einsum treats as (j, i)
        # attentions: (..., seq_len, d_model)  ...ki
        # output:     (..., seq_len, d_model)  ...kj
        return torch.einsum("ji, ...ki-> ...kj", o_weight, attentions)
