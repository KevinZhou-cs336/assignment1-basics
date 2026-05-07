import torch


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.gain = torch.nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # input dims (batch_size, seq_length, d_model) , we are only calculating along side the d_model
        # which means for every token in the sequence, we are calculating the value for d_model parameters
        # for the token only
        rms_x = torch.sqrt(
            torch.pow(x, 2).sum(dim=-1, keepdim=True) / self.d_model + self.eps
        )
        result = (x / rms_x) * self.gain

        return result.to(in_dtype)
