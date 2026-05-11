import torch

from cs336_basics.transformers.linear import Linear


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,  # size of the vocabulary
        embedding_dim: int,  # dimension of the embedding vectors
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):  
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
        self.device = device
        self.dtype = dtype

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (...)  — arbitrary shape, integer indices in [0, vocab_size)
        # weight:    (vocab_size, d_model)
        # output:    (..., d_model)  — each token ID replaced by its embedding row
        return self.weight[token_ids]
