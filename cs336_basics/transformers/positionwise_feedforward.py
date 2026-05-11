import torch

from cs336_basics.transformers.linear import Linear


class SwiGLUFeedForwardNetwork(torch.nn.Module):
    def __init__(
        self,
        d_model: int, # in_features
        d_ff: int, # out_features
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(self.d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, self.d_model)
        self.w3 = Linear(self.d_model, self.d_ff)
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(X) = SwiGLU(X, W_1, W_2, W_3) = W_2 (SiLU(W_1*X) • W_3 * X)
        # SiLU(W_1*X)
        # (d_ff, d_model), (batch, seq_length, d_model) -> (batch, seq_length, d_ff)
        w1_x = torch.einsum("ji,...i ->...j", self.w1.weight, x)
        # (batch, seq_length, d_ff), (batch, seq_length, d_ff) -> (batch, seq_length, d_ff)
        silu_w1x = torch.einsum("...i,...i->...i", w1_x, torch.sigmoid(w1_x))
        # (d_ff, d_model), (batch, seq_length, d_model) -> (batch, seq_length, d_ff)
        w3_x = torch.einsum("ji,...i->...j", self.w3.weight, x)
        # (batch, seq_length, d_ff), (batch, seq_length, d_ff) -> (batch, seq_length, d_ff)
        silu_w1x_dot_w3_x = torch.einsum("...i, ...i->...i", silu_w1x, w3_x)
        # (d_model, d_ff), (batch, seq_length, d_ff) -> (batch, seq_length, d_ff)
        w2_x_silu_w1x_dot_w3_x = torch.einsum(
            "ij, ...j->...i", self.w2.weight, silu_w1x_dot_w3_x
        )

        return w2_x_silu_w1x_dot_w3_x
