import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Numerically stable cross-entropy loss averaged over all token positions.

    Computes: loss = mean_{all positions} -log p(target_token | context)

    Naive approach (NOT used here):
        prob = softmax(logits)[target]  # exp(logit) can overflow to inf
        loss = -log(prob)              # if prob underflows to 0, log(0) = -inf

    Two distinct numerical hazards exist, each requiring its own fix:

    Hazard 1 — Overflow in the denominator (exp → inf):
        Fixed by subtracting max(o) before exp. After shifting, all arguments
        to exp are ≤ 0, so exp ∈ (0, 1] and the sum is always finite.
        Note: subtracting max does NOT fix the numerator underflow below.

    Hazard 2 — Underflow in the numerator (exp → 0 → log(0) = -inf):
        Even after subtracting max, the target token may have a logit far below
        the maximum (e.g., the model confidently predicts the wrong token):
            shifted logits: [0, -1, -2, ..., -1500]  ← target is -1500
            exp(-1500) underflows to 0.0 in float32
            softmax → 0.0 → log(0.0) = -inf
        Fixed by the log-sum-exp identity, which avoids exp on the target entirely:
            -log(softmax(o)[t]) = -log( exp(o[t]) / Σ exp(o[k]) )
                                = -o[t] + log( Σ exp(o[k]) )
        The numerator is just the raw logit o[t] — no exp, no underflow possible.

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
    # The log-sum-exp identity reduces the numerator to just o[t] — no exp call.
    # This is the critical protection against Hazard 2: even though subtracting
    # max prevents denominator overflow, it does NOT prevent the target token's
    # exp from underflowing to 0 when the target logit is far below the max.
    # By never calling exp on the target, underflow → log(0) is impossible here.
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
