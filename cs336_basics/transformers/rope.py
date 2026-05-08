import torch


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None
    ):
        super().__init__()

        self.theta = theta
        # d_k: dimension of query and key vectors
        assert d_k % 2 == 0, "d_k must be an even number"
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
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
        # token pos matrix , dimension(max_seq, 1)
        i_matrices = torch.arange(0, self.max_seq_len).unsqueeze(1)
        # theta matrix , dimension(1, d_k / 2) i * 1 / theta^(2k/d) whee k in [0, d/2 -1]
        theta_matrices = i_matrices / torch.pow(
            self.theta, torch.arange(0, self.d_k, 2) / self.d_k
        ).unsqueeze(0)
        # sin_matix, dimension (max_seq, d_/2)
        sin_matrices = torch.sin(theta_matrices)
        cos_matrices = torch.cos(theta_matrices)

        return sin_matrices, cos_matrices

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x, dimenstion (seq_len, d_k)
        x_even_pos = x[..., 0::2]
        x_odd_pos = x[..., 1::2]
        # sin/cos matix, dimension (seq_len, d_k/2)
        sin_matrices = self.sin_matrices[token_positions]
        cos_matrices = self.cos_matrices[token_positions]
        # x encoded with positional information
        # x_new[2k] = x_old[2k] * cos - x_old[2k+1] * sin
        positional_x_even = x_even_pos * cos_matrices - x_odd_pos * sin_matrices
        # x_new[2k+1] = x_old[2k] * sin + x_old[2k+1] * cos
        positional_x_odd = x_even_pos * sin_matrices + x_odd_pos * cos_matrices

        # stack positional x even and old
        # x_i even [0 2 4 ...]
        # x_i odd  [1 3 5 ...]
        # stack_x [[0, 1], [2, 3], [4, 5]]

        stacked_positional_x = torch.stack(
            [positional_x_even, positional_x_odd], dim=-1
        )

        return torch.reshape(stacked_positional_x, (*x.shape[:-1], -1))
