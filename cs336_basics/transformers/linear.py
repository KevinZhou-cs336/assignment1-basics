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
        # Row-majoring matrix d_out * d_in
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        std = math.sqrt(2 / (out_features + in_features))
        torch.nn.init.trunc_normal_(
            self.weight, mean=0, std=std, a=-3 * std, b=3 * std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...i,ji->...j", x, self.weight)
