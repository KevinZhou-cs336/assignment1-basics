import torch


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Normalizes each token vector by its RMS, then scales by a learned weight:
        RMSNorm(x) = (x / RMS(x)) * weight,   RMS(x) = sqrt(mean(x²) + eps)

    Unlike LayerNorm, there is no mean-centering (re-centering) step, which
    reduces computation while preserving normalization stability.

    weight (gain) shape: (d_model,) — one learnable scale per feature dimension.
    Computation is done in float32 regardless of input dtype to avoid overflow.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        # Learnable per-dimension scale, initialized to 1 (identity at start of training)
        # State dict key: weight, shape (d_model,)
        self.weight = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)  — arbitrary leading batch/sequence dims
        in_dtype = x.dtype

        # Cast to float32 for numerical stability during normalization
        x = x.to(torch.float32)

        # Compute RMS along the last dim (d_model), keeping dim for broadcasting
        # pow → sum → divide by d_model gives mean of squares; +eps prevents division by zero
        # rms_x: (..., 1)  — one RMS value per token vector
        rms_x = torch.sqrt(
            torch.pow(x, 2).sum(dim=-1, keepdim=True) / self.d_model + self.eps
        )

        # Normalize then apply learned scale
        # (x / rms_x): (..., d_model)  — unit-RMS vectors
        # * self.weight: (..., d_model)  — per-feature rescaling; weight broadcasts over batch/seq
        result = (x / rms_x) * self.weight

        # Cast back to original dtype (e.g. bfloat16 for mixed-precision training)
        return result.to(in_dtype)
