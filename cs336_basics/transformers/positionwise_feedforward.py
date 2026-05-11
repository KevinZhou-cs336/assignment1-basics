import torch

from cs336_basics.transformers.linear import Linear


class SwiGLUFeedForwardNetwork(torch.nn.Module):
    """Position-wise Feed-Forward Network with SwiGLU activation (Shazeer, 2020).

    FFN(x) = W_2 · (SiLU(W_1 x) ⊙ W_3 x)

    W_1, W_3 up-project from d_model → d_ff (gate and value streams).
    W_2 down-projects from d_ff → d_model.
    SiLU(z) = z · σ(z) acts as a gating mechanism on the W_1 branch.

    State dict keys: w1.weight (d_ff, d_model), w2.weight (d_model, d_ff), w3.weight (d_ff, d_model)
    """

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
        # w1, w3: up-projections (gate and value), weight shape (d_ff, d_model)
        # w2: down-projection,                     weight shape (d_model, d_ff)
        self.w1 = Linear(self.d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, self.d_model)
        self.w3 = Linear(self.d_model, self.d_ff)
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)  — arbitrary leading batch/sequence dims

        # Step 1: Gate stream — W_1 x, then apply SiLU (= z * sigmoid(z))
        # einsum "ji,...i->...j": w1.weight (d_ff, d_model) × x (..., d_model) → (..., d_ff)
        w1_x = torch.einsum("ji,...i ->...j", self.w1.weight, x)
        # SiLU element-wise: "...i,...i->...i" is equivalent to w1_x * sigmoid(w1_x)
        # silu_w1x: (..., d_ff)
        silu_w1x = torch.einsum("...i,...i->...i", w1_x, torch.sigmoid(w1_x))

        # Step 2: Value stream — W_3 x
        # einsum "ji,...i->...j": w3.weight (d_ff, d_model) × x (..., d_model) → (..., d_ff)
        w3_x = torch.einsum("ji,...i->...j", self.w3.weight, x)

        # Step 3: Element-wise gate — SiLU(W_1 x) ⊙ W_3 x
        # "...i,...i->...i": element-wise product along d_ff dim
        # silu_w1x_dot_w3_x: (..., d_ff)
        silu_w1x_dot_w3_x = torch.einsum("...i, ...i->...i", silu_w1x, w3_x)

        # Step 4: Down-project back to d_model — W_2 · (gated)
        # einsum "ij,...j->...i": w2.weight (d_model, d_ff) × (..., d_ff) → (..., d_model)
        # output: (..., d_model)
        return torch.einsum("ij, ...j->...i", self.w2.weight, silu_w1x_dot_w3_x)
