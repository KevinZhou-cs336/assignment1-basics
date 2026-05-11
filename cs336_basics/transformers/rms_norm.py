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
        # x: (..., d_model)
        #   Leading dims (...) can be any combination of batch and sequence axes,
        #   e.g. (batch_size, seq_len, d_model) or just (seq_len, d_model).
        #   The last axis (dim=-1, size d_model) holds the feature vector for one token.
        #
        # Key design choice: normalization is PER-TOKEN and INDEPENDENT across tokens.
        #   Each token's d_model-dimensional vector is normalized by its own RMS scalar.
        #   No information flows between tokens — this is purely an elementwise rescaling
        #   along the feature axis (dim=-1), not across the sequence axis.
        in_dtype = x.dtype

        # Promote to float32 to avoid overflow in pow/sqrt under bfloat16/float16
        x = x.to(torch.float32)

        # Compute one RMS scalar per token by reducing over the feature axis (dim=-1):
        #   pow(x, 2)              : (..., d_model)  — square every feature value
        #   .sum(dim=-1, keepdim=True) : (..., 1)    — sum of squares within each token's vector
        #   / self.d_model         : (..., 1)        — mean of squares (= variance without centering)
        #   + self.eps             : (..., 1)        — numerical stability guard against division by zero
        #   sqrt(...)              : (..., 1)        — RMS scalar, one per token
        rms_x = torch.sqrt(
            torch.pow(x, 2).sum(dim=-1, keepdim=True) / self.d_model + self.eps
        )

        # Normalize: divide each token's feature vector by that token's RMS scalar.
        #   rms_x broadcasts from (..., 1) to (..., d_model) along the feature axis.
        #   Result: every token vector now has RMS = 1.
        #
        # Rescale: multiply by the learned weight vector (shape: d_model,).
        #   weight broadcasts over all leading dims (...), applying a per-feature scale.
        #   This lets the model learn to restore amplitude on a per-dimension basis.
        result = (x / rms_x) * self.weight

        # Cast back to original dtype (e.g. bfloat16 for mixed-precision training)
        return result.to(in_dtype)
