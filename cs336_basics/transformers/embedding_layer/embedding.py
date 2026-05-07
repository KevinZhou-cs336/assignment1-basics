from numpy import sqrt
import torch


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
        self.embedding_matrices = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        torch.nn.init.trunc_normal_(self.embedding_matrices, mean=0, std=1, a=-3, b=3)
        self.device = device
        self.dtype = dtype

    def forward(self, token_ids: torch.Tensor)->torch.Tensor:
        return self.embedding_matrices[token_ids]
