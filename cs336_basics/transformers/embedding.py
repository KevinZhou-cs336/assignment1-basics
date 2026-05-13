import torch


class Embedding(torch.nn.Module):
    """Learnable token embedding table.

    A language model cannot operate on raw integer token IDs — neural networks
    require continuous, differentiable inputs. The embedding table solves this by
    mapping each token ID to a dense vector of floating-point numbers (its "embedding").

    Concretely, the table is a matrix of shape (vocab_size, embedding_dim):
        - Each ROW corresponds to one vocabulary token.
        - The forward pass is a simple row lookup: weight[token_id] extracts the
          corresponding row, replacing each integer ID with its embedding vector.

    Why learned embeddings instead of one-hot vectors?
        A one-hot vector for a vocabulary of 50,000 tokens would be 50,000-dimensional
        and completely sparse. Learned embeddings are dense and low-dimensional
        (e.g., 512 dimensions), and encode semantic relationships: tokens with similar
        meanings end up with similar embedding vectors, enabling the model to generalise.

    Initialisation — truncated normal (mean=0, std=1, clipped at ±3):
        Random initialisation breaks symmetry so each token starts with a different
        embedding. Clipping at ±3σ prevents extreme outliers from destabilising
        the first few training steps.

    State dict key: weight  shape (num_embeddings, embedding_dim)

    Args:
        num_embeddings: Vocabulary size V — number of distinct token IDs.
        embedding_dim:  Embedding dimension D — length of each token vector.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # weight: (num_embeddings, embedding_dim) — the full embedding table.
        # Stored as nn.Parameter so PyTorch tracks it for gradient updates.
        self.weight = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        # Truncated normal init: mean=0, std=1, clipped to [-3, 3]
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
        self.device = device
        self.dtype = dtype

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (...)           integer indices in [0, num_embeddings)
        #                            can be any shape: (B,), (B, T), etc.
        # weight:    (V, D)          the full embedding table, V rows × D columns
        # output:    (..., D)        each integer ID is replaced by its D-dimensional row
        #
        # weight[token_ids] is PyTorch advanced indexing: for each element in token_ids,
        # fetch the corresponding row from weight. No matrix multiply — just a table lookup.
        # Because weight is an nn.Parameter, gradients flow back through this lookup,
        # updating each token's embedding based on how often it appears and what loss it caused.
        return self.weight[token_ids]
