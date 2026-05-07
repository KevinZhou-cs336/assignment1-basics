import torch


class SwiGLUFeedForwardNetwork(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = torch.nn.Parameter(torch.randn(self.d_ff, self.d_model))
        self.w2_weight = torch.nn.Parameter(torch.randn(self.d_model, self.d_ff))
        self.w3_weight = torch.nn.Parameter(torch.randn(self.d_ff, self.d_model))

        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(X) = SwiGLU(X, W_1, W_2, W_3) = W_2 (SiLU(W_1*X) • W_3 * X)
        # SiLU(W_1*X)
        # (d_ff, d_model), (batch, seq_length, d_model) -> (batch, seq_length, d_ff)
        w1_x = torch.einsum("ji,...i ->...j", self.w1_weight, x)
        # (batch, seq_length, d_ff), (batch, seq_length, d_ff) -> (batch, seq_length, d_ff)
        silu_w1x = torch.einsum("...i,...i->...i", w1_x, torch.sigmoid(w1_x))
        # (d_ff, d_model), (batch, seq_length, d_model) -> (batch, seq_length, d_ff)
        w3_x = torch.einsum("ji,...i->...j", self.w3_weight, x)
        # (batch, seq_length, d_ff), (batch, seq_length, d_ff) -> (batch, seq_length, d_ff)
        silu_w1x_dot_w3_x = torch.einsum("...i, ...i->...i", silu_w1x, w3_x)
        # (d_model, d_ff), (batch, seq_length, d_ff) -> (batch, seq_length, d_ff)
        w2_x_silu_w1x_dot_w3_x = torch.einsum(
            "ij, ...j->...i", self.w2_weight, silu_w1x_dot_w3_x
        )

        return w2_x_silu_w1x_dot_w3_x
