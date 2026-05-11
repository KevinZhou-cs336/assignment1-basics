import torch


class RotaryPositionalEmbedding(torch.nn.Module):
    """Rotary Positional Embedding (RoPE) — Su et al., 2021.

    Encodes absolute position by rotating Q and K vectors; their dot product
    then depends only on the *relative* position offset between query and key tokens.

    Each dimension pair (x_{2k}, x_{2k+1}) is rotated by angle θ_{i,k} = i / Θ^(2k/d_k),
    where i is the token position and Θ (theta) is the base frequency. Lower-index
    dimension pairs rotate faster (higher frequency), higher-index pairs rotate slower.

    Sin/cos lookup tables are pre-computed at init and NOT saved to checkpoints
    (persistent=False) — they are fully re-derivable from (theta, d_k, max_seq_len).

    Args:
        theta:       Base frequency Θ; controls rotation speed decay across dimensions.
        d_k:         Head embedding dimension (= d_model // num_heads); must be even.
        max_seq_len: Maximum sequence length; sets the size of pre-computed tables.
    """

    def __init__(
        self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None
    ):
        super().__init__()

        self.theta = theta
        assert d_k % 2 == 0, "d_k must be an even number"
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # Pre-computed lookup tables: shape (max_seq_len, d_k/2)
        # Not saved to checkpoint (persistent=False) — re-derived from hyperparams at load time
        self.register_buffer(
            "sin_matrices", torch.zeros(max_seq_len, d_k // 2), persistent=False
        )
        self.register_buffer(
            "cos_matrices", torch.zeros(max_seq_len, d_k // 2), persistent=False
        )
        self.sin_matrices, self.cos_matrices = self._build_rotary_position_matrices(
            theta, d_k, max_seq_len
        )

    def _build_rotary_position_matrices(
        self, theta: float, d_k: int, max_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # i_matrices: absolute position indices, shape (max_seq_len, 1)
        # unsqueeze(1) prepares for broadcasting against the (1, d_k/2) frequency vector
        i_matrices = torch.arange(0, max_seq_len).unsqueeze(1)

        # Build rotation angles: theta_matrices[i, k] = i / Θ^(2k/d_k)
        #   torch.arange(0, d_k, 2) / d_k  → (d_k/2,)  exponents 0, 2/d_k, 4/d_k, ...
        #   torch.pow(theta, ...)           → (d_k/2,)  denominators Θ^0, Θ^(2/d_k), ...
        #   .unsqueeze(0)                   → (1, d_k/2)  broadcast-ready
        #   i_matrices / ...               → (max_seq_len, d_k/2)  angle[pos, dim-pair]
        theta_matrices = i_matrices / torch.pow(
            theta, torch.arange(0, d_k, 2) / d_k
        ).unsqueeze(0)

        # sin_matrices, cos_matrices: (max_seq_len, d_k/2)
        return torch.sin(theta_matrices), torch.cos(theta_matrices)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x:               (..., seq_len, d_k)  — Q or K; num_heads is a leading batch dim
        # token_positions: (seq_len,) or (..., seq_len)  — absolute position of each token

        # Step 1: Split last dim into even- and odd-indexed dimensions
        # x_even_pos: (..., seq_len, d_k/2)  — dims 0, 2, 4, ...  (x_{2k})
        # x_odd_pos:  (..., seq_len, d_k/2)  — dims 1, 3, 5, ...  (x_{2k+1})
        x_even_pos = x[..., 0::2]
        x_odd_pos = x[..., 1::2]

        # Step 2: Look up pre-computed sin/cos for the requested token positions
        # Advanced indexing with token_positions broadcasts over all leading (...) dims
        # sin_matrices, cos_matrices: (..., seq_len, d_k/2)
        sin_matrices = self.sin_matrices[token_positions]
        cos_matrices = self.cos_matrices[token_positions]

        # Step 3: Apply 2D rotation to each dimension pair (x_{2k}, x_{2k+1}):
        #   rotated_{2k}   = x_{2k}   * cos(θ_{i,k}) - x_{2k+1} * sin(θ_{i,k})
        #   rotated_{2k+1} = x_{2k}   * sin(θ_{i,k}) + x_{2k+1} * cos(θ_{i,k})
        # positional_x_even, positional_x_odd: (..., seq_len, d_k/2)
        positional_x_even = x_even_pos * cos_matrices - x_odd_pos * sin_matrices
        positional_x_odd = x_even_pos * sin_matrices + x_odd_pos * cos_matrices

        # Step 4: Interleave even and odd results back into original shape
        # stack([even, odd], dim=-1) → (..., seq_len, d_k/2, 2)  pairs along new last dim
        # reshape                    → (..., seq_len, d_k)        restores original shape
        stacked_positional_x = torch.stack(
            [positional_x_even, positional_x_odd], dim=-1
        )
        # output: (..., seq_len, d_k)
        return torch.reshape(stacked_positional_x, (*x.shape[:-1], -1))
