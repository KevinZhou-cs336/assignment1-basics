import torch

from cs336_basics.transformers.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.transformers.positionwise_feedforward import SwiGLUFeedForwardNetwork
from cs336_basics.transformers.rms_norm import RMSNorm
from cs336_basics.transformers.rope import RotaryPositionalEmbedding


class TransformerBlock(torch.nn.Module):
    """Pre-norm Transformer block (Nguyen & Salazar, 2019; Xiong et al., 2020).

    Each block applies two sub-layers with residual connections:

      Sub-layer 1 (attention):
        y = x + MultiHeadSelfAttention(RMSNorm(x))          [eq. 15 in handout]

      Sub-layer 2 (feed-forward):
        output = y + FFN(RMSNorm(y))

    RMSNorm is applied *before* each sub-layer (pre-norm), not after (post-norm).
    This improves gradient flow by keeping a clean residual stream from input to output.

    State dict key structure (matches reference implementation):
        ln1.*   — RMSNorm before attention sub-layer
        attn.*  — MultiHeadSelfAttention (q_proj, k_proj, v_proj, output_proj)
        ln2.*   — RMSNorm before FFN sub-layer
        ffn.*   — SwiGLUFeedForwardNetwork (w1_weights, w2_weights, w3_weights)
    """

    def __init__(
        self,
        attn: MultiHeadSelfAttention,
        ffn: SwiGLUFeedForwardNetwork,
        rms_norm_mha: RMSNorm,
        rms_norm_ffn: RMSNorm,
    ):
        super().__init__()

        self.ln1 = rms_norm_mha   # RMSNorm before attention sub-layer
        self.attn = attn          # MultiHeadSelfAttention (RoPE stored inside attn)
        self.ln2 = rms_norm_ffn   # RMSNorm before FFN sub-layer
        self.ffn = ffn            # SwiGLU feed-forward network

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        # in_features: (batch, seq_len, d_model)

        # --- Sub-layer 1: Multi-Head Self-Attention with pre-norm and residual ---
        # Step 1a: Pre-norm — RMSNorm applied to input before attention
        #   (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        l1_output = self.ln1.forward(in_features)

        # Step 1b: Multi-Head Self-Attention (with RoPE applied inside MHA)
        #   token_positions: (seq_len,) = [0, 1, ..., seq_len-1], created on the same
        #   device as in_features to avoid device mismatch when indexing RoPE sin/cos tables
        #   (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        token_positions = torch.arange(in_features.shape[-2], device=in_features.device)
        l1_output = self.attn.forward(l1_output, token_positions)

        # Step 1c: Residual connection — y = x + MHA(RMSNorm(x))
        #   (batch, seq_len, d_model)
        l1_output += in_features

        # --- Sub-layer 2: Position-wise Feed-Forward Network with pre-norm and residual ---
        # Step 2a: Pre-norm — RMSNorm applied to attention output before FFN
        #   (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        l2_output = self.ln2.forward(l1_output)

        # Step 2b: SwiGLU Feed-Forward Network
        #   (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        l2_output = self.ffn.forward(l2_output)

        # Step 2c: Residual connection — output = y + FFN(RMSNorm(y))
        #   (batch, seq_len, d_model)
        return l1_output + l2_output
