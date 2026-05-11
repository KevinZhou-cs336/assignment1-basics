import torch

from cs336_basics.transformers.embedding import Embedding
from cs336_basics.transformers.linear import Linear
from cs336_basics.transformers.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.transformers.positionwise_feedforward import SwiGLUFeedForwardNetwork
from cs336_basics.transformers.rms_norm import RMSNorm
from cs336_basics.transformers.rope import RotaryPositionalEmbedding
from cs336_basics.transformers.transformer_block import TransformerBlock


class TransformerLanguageModel(torch.nn.Module):
    """Decoder-only Transformer language model (pre-norm variant).

    Architecture (per forward pass):
        token_ids → Embedding → [TransformerBlock × num_layers] → RMSNorm → Linear → logits

    State dict key structure:
        token_embeddings.weight         (vocab_size, d_model)
        layers.{i}.ln1.weight           (d_model,)
        layers.{i}.attn.q_proj.weight   (d_model, d_model)
        layers.{i}.attn.k_proj.weight   (d_model, d_model)
        layers.{i}.attn.v_proj.weight   (d_model, d_model)
        layers.{i}.attn.output_proj.weight  (d_model, d_model)
        layers.{i}.ln2.weight           (d_model,)
        layers.{i}.ffn.w1.weight        (d_ff, d_model)
        layers.{i}.ffn.w2.weight        (d_model, d_ff)
        layers.{i}.ffn.w3.weight        (d_ff, d_model)
        ln_final.weight                 (d_model,)
        lm_head.weight                  (vocab_size, d_model)
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Token embedding table: maps token IDs → d_model-dimensional vectors
        # weight shape: (vocab_size, d_model)
        self.token_embeddings = Embedding(vocab_size, d_model)

        # Shared RoPE instance: pre-computes sin/cos tables of shape (context_length, d_head)
        # d_head = d_model // num_heads; same rope object is passed into every attention layer
        self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length)

        # Stack of num_layers identical TransformerBlocks; ModuleList auto-names them 0..N-1
        # State dict prefix for block i: layers.{i}.*
        self.layers = torch.nn.ModuleList(
            [self._make_transformer_block() for _ in range(num_layers)]
        )

        # Final layer norm applied to the output of the last transformer block
        # weight shape: (d_model,)
        self.ln_final = RMSNorm(self.d_model)

        # Unembedding / language model head: projects d_model → vocab_size (unnormalized logits)
        # weight shape: (vocab_size, d_model)
        self.lm_head = Linear(self.d_model, self.vocab_size)

    def _make_transformer_block(self) -> TransformerBlock:
        # Each block gets its own independent attn/ffn/norm parameters;
        # all blocks share the same rope (no learnable params in rope)
        attn = MultiHeadSelfAttention(self.d_model, self.num_heads, self.rope)
        ffn = SwiGLUFeedForwardNetwork(self.d_model, self.d_ff)
        rms_norm_attn = RMSNorm(self.d_model)
        rms_norm_ffn = RMSNorm(self.d_model)
        return TransformerBlock(attn, ffn, rms_norm_attn, rms_norm_ffn)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch_size, seq_len)  integer token indices in [0, vocab_size)

        # Step 1: Embed token IDs into continuous vectors
        # (batch_size, seq_len) → (batch_size, seq_len, d_model)
        x = self.token_embeddings.forward(token_ids)

        # Step 2: Pass through each TransformerBlock sequentially
        # shape unchanged throughout: (batch_size, seq_len, d_model)
        for block in self.layers:
            x = block.forward(x)

        # Step 3: Final RMSNorm before the language model head
        # (batch_size, seq_len, d_model) → (batch_size, seq_len, d_model)
        x = self.ln_final(x)

        # Step 4: Project to vocabulary logits (unnormalized)
        # (batch_size, seq_len, d_model) → (batch_size, seq_len, vocab_size)
        return self.lm_head(x)
