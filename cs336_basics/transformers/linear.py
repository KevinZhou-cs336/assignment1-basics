import math

import torch


class Linear(torch.nn.Module):
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
        # weight stored row-major: (out_features, in_features)
        # Glorot-style truncated normal init: std = sqrt(2/(in+out)), clipped at ±3σ
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        std = math.sqrt(2 / (out_features + in_features))
        torch.nn.init.trunc_normal_(
            self.weight, mean=0, std=std, a=-3 * std, b=3 * std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:      (..., in_features)
        # weight: (out_features, in_features)  indices ji
        # einsum contracts over i, producing (..., out_features)
        return torch.einsum("...i,ji->...j", x, self.weight)
