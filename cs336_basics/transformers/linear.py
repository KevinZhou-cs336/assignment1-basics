import math

import torch


class Linear(torch.nn.Module):
    """Bias-free linear (fully connected) layer: output = x @ weight.T

    A linear layer multiplies every input vector by a learned weight matrix,
    projecting it from one dimension to another. In a Transformer this is used
    for Q/K/V projections, the output projection, and the final lm_head.

    No bias term — following the LLaMA / PaLM convention that RMSNorm (applied
    before each sub-layer) makes the bias redundant and wastes parameters.

    Why is weight stored as (out_features, in_features) instead of (in_features, out_features)?
        The mathematical convention for a linear map is y = W x, where W is
        (out, in). Storing it this way means each OUTPUT neuron's weights are
        a contiguous row, making row-wise operations cache-friendly and matching
        the convention used in PyTorch's own nn.Linear.

    Initialisation — Glorot (Xavier) truncated normal:
        std = sqrt(2 / (in_features + out_features))
        Random weights must have the right scale so that activations neither
        vanish (all zeros) nor explode (all infinity) as they pass through many
        layers. Glorot init keeps the variance of activations roughly constant
        across layers by accounting for both the fan-in and fan-out.
        Clipping at ±3σ removes extreme outliers that could destabilise early training.

    State dict key: weight  shape (out_features, in_features)

    Args:
        in_features:  Dimension of each input vector (D_in).
        out_features: Dimension of each output vector (D_out).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        # weight: (out_features, in_features) — each row is one output neuron's weights.
        # Glorot-style truncated normal: std = sqrt(2/(in+out)), clipped at ±3σ.
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        std = math.sqrt(2 / (out_features + in_features))
        torch.nn.init.trunc_normal_(
            self.weight, mean=0, std=std, a=-3 * std, b=3 * std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:      (..., in_features)   input vectors; leading dims are batch/sequence axes
        # weight: (out_features, in_features)  indices ji  (j=out, i=in)
        # einsum "...i,ji->...j" contracts over i (the shared in_features axis),
        # producing (..., out_features) — equivalent to x @ weight.T
        return torch.einsum("...i,ji->...j", x, self.weight)
