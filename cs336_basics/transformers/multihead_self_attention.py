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
        self.d_head = d_model // num_heads
        self.rope = rope

    def forward(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        o_weight: torch.Tensor,
        in_features: torch.Tensor,
    ) -> torch.Tensor:
        # weighs (d_model, d_model)
        q_weights = torch.reshape(q_weight, (self.num_heads, self.d_head, self.d_model))
        k_weights = torch.reshape(k_weight, (self.num_heads, self.d_head, self.d_model))
        v_weights = torch.reshape(v_weight, (self.num_heads, self.d_head, self.d_model))

        # input_feature(..., sequence, d_model)
        # q_x (... num of heads, sequence, d_head)
        q_x = torch.einsum("...xj, nhj->...nxh", in_features, q_weights)
        token_position = torch.arange(in_features.shape[-2])
        if self.rope:
            q_x = self.rope.forward(q_x, token_position)
        k_x = torch.einsum("...xj, nhj->...nxh", in_features, k_weights)
        if self.rope:
            k_x = self.rope.forward(k_x, token_position)
        v_x = torch.einsum("...xj, nhj->...nxh", in_features, v_weights)

        # lower triangle
        mask = torch.tril(
            torch.ones(in_features.shape[-2], in_features.shape[-2])
        ).bool()

        multihead_attentions = scaled_dot_product_attention(q_x, k_x, v_x, mask)
        multihead_attentions = multihead_attentions.transpose(-2, -3)
        attentions = multihead_attentions.reshape(
            (*multihead_attentions.shape[:-2], -1)
        )

        # o_weight: (out_dimension, in_dimension)
        # attentions: (..., sequence, in_dimension)
        return torch.einsum("ji, ...ki-> ...kj", o_weight, attentions)
