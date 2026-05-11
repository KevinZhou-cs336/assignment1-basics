import math

import torch


def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    """Apply softmax along a single dimension.

    Converts raw scores into a probability distribution: all values along the
    specified axis fall in (0, 1] and sum to 1.

    Args:
        in_features: Input tensor of arbitrary shape (...).
        dim: The axis along which softmax is computed. Elements along this axis
            compete against each other for probability mass.

            Example — attention score matrix of shape (seq_len_q, seq_len_k):
              - Each row holds one query's scores against all keys.
              - softmax(dim=-1) normalizes along the key axis, so each query's
                scores sum to 1: the query distributes 100% of its attention
                across all keys. This is the correct choice for attention.
              - softmax(dim=0) would instead normalize along the query axis,
                making each key's weights across all queries sum to 1 — not
                meaningful for attention.

            All dimensions other than `dim` are treated as independent batch
            dimensions and are left unchanged.

    Returns:
        Tensor of the same shape as `in_features`. The slice along `dim` at
        every position forms a valid probability distribution.
    """
    normalized_in_features = (
        in_features - torch.max(in_features, dim, keepdim=True).values
    )

    return torch.exp(normalized_in_features) / torch.sum(
        torch.exp(normalized_in_features), dim, keepdim=True
    )


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention (Vaswani et al., 2017).

    Computes  Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V,
    optionally zeroing out attention to masked-out key positions.

    Args:
        query: Query tensor of shape (..., seq_len_q, d_k).
            The leading dimensions (...) can be any combination of batch and
            head dimensions and are treated as independent batch axes.
            `seq_len_q` is the number of query tokens.
            `d_k` is the query/key head dimension.

        key: Key tensor of shape (..., seq_len_k, d_k).
            Must share the same leading dimensions and `d_k` as `query`.
            `seq_len_k` is the number of key tokens (equals `seq_len_q` for
            self-attention).

        value: Value tensor of shape (..., seq_len_k, d_v).
            Must share the same leading dimensions and `seq_len_k` as `key`.
            `d_v` is the value head dimension (often equal to `d_k`).

        mask: Optional boolean tensor of shape (seq_len_q, seq_len_k).
            mask[i, j] == True  → query i is allowed to attend to key j.
            mask[i, j] == False → query i is blocked from attending to key j;
                the pre-softmax score at that position is set to -inf so the
                corresponding attention weight becomes 0.
            Pass a lower-triangular True mask for causal (autoregressive)
            attention. If None, all positions attend to all positions.

    Returns:
        Output tensor of shape (..., seq_len_q, d_v). Each query position
        receives a weighted sum of value vectors, where the weights are the
        softmax-normalized, scale-adjusted dot products with the keys.
    """
    # Step 1: Compute raw attention scores Q K^T
    # einsum "...qj,...kj->...qk":
    #   query: (..., seq_len_q, d_k)  indices ...qj
    #   key:   (..., seq_len_k, d_k)  indices ...kj  (shared j=d_k contracts)
    #   → (..., seq_len_q, seq_len_k)  score[..., q, k] = dot(query[q], key[k])
    query_key = torch.einsum("...qj,...kj->...qk", query, key)

    # Step 2: Scale by 1/sqrt(d_k) to prevent dot products from growing large
    # (large values push softmax into near-zero gradient regions)
    # query_key: (..., seq_len_q, seq_len_k)  unchanged shape
    d_k = query.shape[-1]
    qk_d_k = query_key / math.sqrt(d_k)

    # Step 3: Apply causal mask — set forbidden positions to -inf before softmax
    # mask[i, j] == True → query i may attend to key j (keep score)
    # mask[i, j] == False → query i blocked from key j → score → -inf → weight → 0
    # qk_d_k: (..., seq_len_q, seq_len_k)  unchanged shape
    if mask is not None:
        qk_d_k = qk_d_k.masked_fill(mask == False, float('-inf'))

    # Step 4: Softmax over the key axis — each query gets a probability distribution over keys
    # softmax_qk: (..., seq_len_q, seq_len_k)  rows sum to 1
    softmax_qk = softmax(qk_d_k, -1)

    # Step 5: Weighted sum of value vectors
    # einsum "...nm,...mv->...nv":
    #   softmax_qk: (..., seq_len_q, seq_len_k)  indices ...nm  (n=seq_len_q, m=seq_len_k)
    #   value:      (..., seq_len_k, d_v)         indices ...mv  (m=seq_len_k contracts, v=d_v)
    #   → (..., seq_len_q, d_v)  output[..., q, :] = weighted sum of value rows
    return torch.einsum("...nm,...mv->...nv", softmax_qk, value)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Numerically stable cross-entropy loss averaged over all token positions.

    Computes: loss = mean_{all positions} -log p(target_token | context)

    Naive approach (NOT used here):
        prob = softmax(logits)[target]  # exp(logit) can overflow to inf
        loss = -log(prob)              # if prob underflows to 0, log(0) = -inf

    Instead we use the log-sum-exp identity to bypass those intermediate values:
        -log(softmax(o)[t]) = -log( exp(o[t]) / Σ exp(o[k]) )
                            = -o[t] + log( Σ exp(o[k]) )
    The numerator o[t] is a raw logit — exp and log cancel algebraically so
    we never compute them, eliminating the underflow → log(0) risk.
    The denominator still calls exp, but subtracting max(o) first keeps every
    argument ≤ 0, bounding exp ∈ (0, 1] and preventing overflow.

    Args:
        logits:  (..., vocab_size)  Raw (pre-softmax) scores for every vocabulary
                 token. Leading dims can be any mix of batch/sequence axes.
        targets: (...)             Integer token indices in [0, vocab_size).
                 Shape must match the leading dims of logits.

    Returns:
        Scalar — mean cross-entropy loss over all token positions.
    """
    in_dtype = logits.dtype

    # Promote to float32: bfloat16/float16 exp/log are prone to overflow.
    logits = logits.to(torch.float32)

    # Flatten all leading (batch/sequence) dims into one so 2-D indexing works
    # for any input shape (batch, seq_len, vocab_size) or (batch, vocab_size).
    #   logits:  (N, vocab_size)  where N = total number of token predictions
    #   targets: (N,)             one correct token index per prediction
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)

    # Subtract per-row max before exp (log-sum-exp stability trick).
    # Shifting by a constant c satisfies: exp(o-c)/Σexp(o_k-c) = exp(o)/Σexp(o_k)
    # because c cancels in numerator and denominator.
    # After shifting, all arguments to exp are ≤ 0, so exp ∈ (0, 1] — no overflow.
    # Shape: (N, vocab_size), unchanged.
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values

    # Numerator: the shifted logit at the target token position for each row.
    # From the identity above, the -o[t] term is just the raw (shifted) logit —
    # no exp is computed here, so there is no underflow → log(0) risk.
    # Shape: (N,)
    logits_numerator = logits[torch.arange(targets.shape[0]), targets]

    # Denominator: log(Σ exp(shifted_logits)) along the vocab axis.
    # Because we subtracted the max, the sum ≥ exp(0) = 1, so log(sum) ≥ 0
    # and we never hit log(0). Individual exp terms may underflow to 0 for very
    # negative logits, but those tokens have negligible probability anyway.
    # Shape: (N,)
    logits_denominator = torch.log(torch.sum(torch.exp(logits), dim=-1))

    # Per-token loss = -o[target] + log(Σ exp(o[k]))  (shape: (N,))
    # Sum over all N positions then divide by N to get the mean loss scalar.
    results = torch.sum(-logits_numerator + logits_denominator) / logits.shape[0]

    return results.to(in_dtype)