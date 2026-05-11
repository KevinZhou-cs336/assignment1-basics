import torch


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None
    ):
        super().__init__()

        self.theta = theta
        assert d_k % 2 == 0, "d_k must be an even number"
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # Pre-computed sin/cos tables; not saved to checkpoint (derivable from theta/d_k/max_seq_len)
        # sin_matrices, cos_matrices: (max_seq_len, d_k // 2)
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
    ) -> torch.Tensor:
        # Position indices: (max_seq_len, 1)
        i_matrices = torch.arange(0, max_seq_len).unsqueeze(1)

        # Frequency for each dimension pair k: theta^(2k / d_k), k in [0, d_k/2 - 1]
        # torch.arange(0, d_k, 2) / d_k: (d_k/2,)  -> unsqueeze -> (1, d_k/2)
        # angle[i, k] = i / theta^(2k/d_k)  = position i × frequency k
        # theta_matrices: (max_seq_len, d_k/2)
        theta_matrices = i_matrices / torch.pow(
            theta, torch.arange(0, d_k, 2) / d_k
        ).unsqueeze(0)

        # sin_matrices, cos_matrices: (max_seq_len, d_k/2)
        sin_matrices = torch.sin(theta_matrices)
        cos_matrices = torch.cos(theta_matrices)

        return sin_matrices, cos_matrices

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x:               (..., seq_len, d_k)
        # token_positions: (seq_len,) or (..., seq_len)

        # Split x into even-indexed and odd-indexed dimensions along last axis
        # x_even_pos, x_odd_pos: (..., seq_len, d_k/2)
        x_even_pos = x[..., 0::2]
        x_odd_pos = x[..., 1::2]

        # Look up pre-computed sin/cos for the given positions
        # sin_matrices[token_positions]: (..., seq_len, d_k/2)  (index broadcasts over leading dims)
        sin_matrices = self.sin_matrices[token_positions]
        cos_matrices = self.cos_matrices[token_positions]

        # Apply 2D rotation to each (x_{2k}, x_{2k+1}) pair:
        # x_new[2k]   = x[2k]   * cos(θ_{i,k}) - x[2k+1] * sin(θ_{i,k})
        # x_new[2k+1] = x[2k]   * sin(θ_{i,k}) + x[2k+1] * cos(θ_{i,k})
        # positional_x_even, positional_x_odd: (..., seq_len, d_k/2)
        positional_x_even = x_even_pos * cos_matrices - x_odd_pos * sin_matrices
        positional_x_odd = x_even_pos * sin_matrices + x_odd_pos * cos_matrices

        # Interleave even and odd back: stack along new last dim then flatten
        # stack -> (..., seq_len, d_k/2, 2)
        # reshape -> (..., seq_len, d_k)  with order [even_0, odd_0, even_1, odd_1, ...]
        stacked_positional_x = torch.stack(
            [positional_x_even, positional_x_odd], dim=-1
        )

        return torch.reshape(stacked_positional_x, (*x.shape[:-1], -1))
