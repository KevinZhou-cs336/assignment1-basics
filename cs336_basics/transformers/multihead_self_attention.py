import torch

from cs336_basics.transformers.functions import scaled_dot_product_attention
from cs336_basics.transformers.linear import Linear
from cs336_basics.transformers.rope import RotaryPositionalEmbedding


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding = None
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # d_k = d_v = d_model / num_heads
        self.rope = rope

        # Q/K/V/output projections: each Linear(d_model, d_model), weight shape (d_model, d_model)
        # State dict keys: q_proj.weight, k_proj.weight, v_proj.weight, output_proj.weight
        # The full d_model output is split across num_heads inside forward()
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(
        self,
        in_features: torch.Tensor,  # (..., seq_len, d_model)
        token_positions: torch.Tensor = None,  # (..., seq_len)
    ) -> torch.Tensor:  # (..., seq_len, d_model)
        # Step 1: Project input into Q, K, V then split into per-head tensors
        #
        # einsum "...oi, ni->...on":
        #   in_features:  (..., seq_len, d_model)  o=seq_len, i=d_model_in
        #   proj.weight:  (d_model, d_model)        n=d_model_out, i=d_model_in
        #   → contracts over i, result: (..., seq_len, d_model)
        #
        # reshape → (..., seq_len, num_heads, d_head)  splits d_model into heads
        # transpose(-2,-3) → (..., num_heads, seq_len, d_head)  heads before seq for SDPA
        q_x = (
            torch.einsum("...oi, ni->...on", in_features, self.q_proj.weight)
            .reshape((*in_features.shape[:-1], self.num_heads, self.d_head))
            .transpose(-2, -3)
        )
        k_x = (
            torch.einsum("...oi, ni->...on", in_features, self.k_proj.weight)
            .reshape((*in_features.shape[:-1], self.num_heads, self.d_head))
            .transpose(-2, -3)
        )
        v_x = (
            torch.einsum("...oi, ni->...on", in_features, self.v_proj.weight)
            .reshape((*in_features.shape[:-1], self.num_heads, self.d_head))
            .transpose(-2, -3)
        )
        # q_x, k_x, v_x: (..., num_heads, seq_len, d_head)

        # Step 2: Apply RoPE to Q and K (not V)
        # num_heads is treated as a batch dim; shape unchanged: (..., num_heads, seq_len, d_head)
        if self.rope:
            q_x = self.rope.forward(q_x, token_positions)
            k_x = self.rope.forward(k_x, token_positions)

        # Step 3: Causal mask — lower-triangular boolean (seq_len, seq_len)
        # mask[i, j] = True means query at position i can attend to key at position j
        # device= must match in_features to avoid CPU/GPU mismatch in masked_fill inside SDPA
        mask = torch.tril(
            torch.ones(in_features.shape[-2], in_features.shape[-2], device=in_features.device)
        ).bool()

        # Step 4: Scaled dot-product attention applied independently per head
        # Input/output: (..., num_heads, seq_len, d_head)
        multihead_attentions = scaled_dot_product_attention(q_x, k_x, v_x, mask)

        # Step 5: Merge heads back into d_model
        # transpose(-2,-3): (..., num_heads, seq_len, d_head) → (..., seq_len, num_heads, d_head)
        # reshape:          (..., seq_len, num_heads, d_head) → (..., seq_len, d_model)
        multihead_attentions = multihead_attentions.transpose(-2, -3)
        attentions = multihead_attentions.reshape(
            (*multihead_attentions.shape[:-2], -1)
        )
        # attentions: (..., seq_len, d_model)

        # Step 6: Output projection W_O
        # output_proj.weight: (d_model, d_model)  j=d_model_out, i=d_model_in
        # attentions:         (..., seq_len, d_model)  indices ...ki
        # output:             (..., seq_len, d_model)  indices ...kj
        return torch.einsum("ji, ...ki-> ...kj", self.output_proj.weight, attentions)
