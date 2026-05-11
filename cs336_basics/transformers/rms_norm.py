import torch


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Normalizes each token vector by its RMS, then applies a learned per-feature scale:
        RMSNorm(x) = (x / RMS(x)) * weight
        RMS(x)     = sqrt( mean(x²) + eps )
                   = sqrt( sum(x²) / d_model + eps )

    Unlike LayerNorm, there is no mean-centering step, reducing computation while
    preserving normalization stability. Computation is promoted to float32 regardless
    of input dtype to avoid numerical overflow in mixed-precision training.

    State dict key: weight  shape (d_model,)  — initialized to ones (identity)
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
        # Learned per-dimension scale; initialized to 1 so RMSNorm starts as identity
        # State dict key: weight, shape (d_model,)
        self.weight = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)  — arbitrary leading batch/sequence dims
        in_dtype = x.dtype

        # Promote to float32 to avoid overflow in pow/sqrt under bfloat16/float16
        x = x.to(torch.float32)

        # Compute per-token RMS along the feature axis (dim=-1)
        # pow(x, 2): (..., d_model)  element-wise square
        # .sum(dim=-1, keepdim=True): (..., 1)  sum of squares per token
        # / d_model + eps:            (..., 1)  mean of squares + stability term
        # sqrt:                       (..., 1)  RMS value per token
        rms_x = torch.sqrt(
            torch.pow(x, 2).sum(dim=-1, keepdim=True) / self.d_model + self.eps
        )

        # Normalize then apply learned scale
        # x / rms_x:      (..., d_model)  unit-RMS vectors (rms_x broadcasts over d_model)
        # * self.weight:   (..., d_model)  per-feature rescaling (weight broadcasts over batch/seq)
        result = (x / rms_x) * self.weight

        # Cast back to original dtype (e.g. bfloat16 for mixed-precision training)
        return result.to(in_dtype)
