from collections.abc import Iterable
import math

import torch


def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine annealing learning rate schedule with linear warmup (LLaMA-style).

    Training is divided into three phases:

    Phase 1 — Linear warmup (0 ≤ it < warmup_iters):
        The learning rate rises linearly from 0 to max_learning_rate.
        Why: At the very start of training, the model's weights are random and the
        gradients are noisy and unreliable. Jumping straight to a large learning rate
        would cause wild, destructive updates. Warming up slowly lets the optimizer
        build up stable gradient estimates before taking big steps.

    Phase 2 — Cosine decay (warmup_iters ≤ it ≤ cosine_cycle_iters):
        The learning rate decreases smoothly from max_learning_rate to min_learning_rate
        following a cosine curve. `progress` tracks how far through this phase we are
        (0.0 = just started decay, 1.0 = reached the end).
        Why cosine and not linear? At the start of decay we want to stay near the
        maximum a little longer (the model is still improving quickly); at the end we
        want to slow down gradually rather than stopping abruptly. The cosine shape
        gives exactly this: it starts nearly flat, then drops steeply in the middle,
        then flattens again near min_learning_rate.

    Phase 3 — Post-annealing (it > cosine_cycle_iters):
        The learning rate stays fixed at min_learning_rate.
        Why: Fine-tuning at a tiny constant rate lets the model settle into a sharp
        minimum without overshooting.

    Args:
        it:                current training step (starts at 0).
        max_learning_rate: peak learning rate, reached at end of warmup.
        min_learning_rate: floor learning rate, held after cosine decay ends.
        warmup_iters:      number of steps for the linear warmup phase.
        cosine_cycle_iters: step at which cosine decay reaches min_learning_rate.

    Returns:
        The learning rate to use at step `it`.
    """
    if it < warmup_iters:
        # Phase 1: ramp linearly from 0 → max_learning_rate
        return max_learning_rate * it / warmup_iters

    elif it <= cosine_cycle_iters:
        # Phase 2: cosine decay from max_learning_rate → min_learning_rate.
        # progress = 0.0 at start of decay, 1.0 at end.
        # cos(π × 0) = 1  → returns max_learning_rate  (start of decay)
        # cos(π × 1) = -1 → returns min_learning_rate  (end of decay)
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (
            max_learning_rate - min_learning_rate
        )

    else:
        # Phase 3: hold at minimum learning rate
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """Clip gradients so their global L2 norm does not exceed max_l2_norm.

    ── Why do we need gradient clipping? ──────────────────────────────────────

    During backpropagation, each parameter gets a gradient — a number that says
    "nudge this parameter in this direction by this much." Normally these gradients
    are small and the model makes steady progress.

    But sometimes, especially in deep models or unlucky batches, the gradients
    can become explosively large ("exploding gradients"). When the optimizer then
    multiplies these huge gradients by the learning rate and applies them, the
    parameters get shoved so far from where they were that the model essentially
    forgets everything it has learned. Loss spikes, training diverges.

    Gradient clipping prevents this by saying: "if the total magnitude of all
    gradients combined is too large, shrink them all down proportionally."

    ── What is the "global L2 norm"? ──────────────────────────────────────────

    Imagine every gradient in the model — every single number in every parameter's
    gradient tensor — laid out in a single long row. The global L2 norm is the
    length of that row, computed the same way as Euclidean distance in geometry:

        global_norm = sqrt(g₁² + g₂² + g₃² + ... + g_N²)

    where g₁, g₂, ..., g_N are all N gradient values across all parameters.
    It captures "how large are all the gradients together."

    ── How does clipping work? ─────────────────────────────────────────────────

    If global_norm ≤ max_l2_norm: gradients are already small enough — do nothing.

    If global_norm > max_l2_norm: scale every gradient down by the same factor
        scale = max_l2_norm / global_norm

    After scaling, the new global norm equals exactly max_l2_norm.

    ── Proof that the new norm is exactly max_l2_norm ─────────────────────────

    Let g₁, g₂, …, gₙ be all gradient values. Before scaling:
        global_norm = √(g₁² + g₂² + … + gₙ²)

    After multiplying every value by scale = max_l2_norm / global_norm:
        new_norm = √((scale·g₁)² + (scale·g₂)² + … + (scale·gₙ)²)
                 = √(scale² · (g₁² + g₂² + … + gₙ²))
                 = scale · √(g₁² + g₂² + … + gₙ²)
                 = scale · global_norm
                 = (max_l2_norm / global_norm) · global_norm
                 = max_l2_norm  ✓

    Concrete example — suppose the model has only 3 gradient values:
        g = [3, 4, 0],  max_l2_norm = 2

        global_norm = √(3² + 4² + 0²) = √25 = 5
        scale       = 2 / 5 = 0.4

        scaled g    = [3×0.4, 4×0.4, 0×0.4] = [1.2, 1.6, 0.0]
        new_norm    = √(1.2² + 1.6² + 0²) = √(1.44 + 2.56) = √4 = 2  ✓

    Crucially, every gradient is multiplied by the same scale, so their relative
    proportions are preserved — the "direction" of the update is unchanged, only
    its magnitude is reduced. Think of it as shrinking an arrow: the direction
    stays the same, only the length changes.

    ── Why add epsilon to the denominator? ────────────────────────────────────

    If every gradient happens to be zero, global_norm = 0. Dividing by zero would
    cause NaN. Adding a tiny epsilon (1e-6) prevents this edge case.

    Args:
        parameters:   iterable of nn.Parameter (e.g., model.parameters()).
                      Only parameters with a non-None .grad are considered.
        max_l2_norm:  maximum allowed global L2 norm of the gradient vector.
    """
    epsilon = 1e-6

    # Materialise into a list so we can iterate twice without exhausting a generator.
    parameters = list(parameters)

    # Collect only the gradient tensors that actually received a gradient this step.
    # (Parameters that were not involved in the forward pass have grad = None.)
    grads = [p.grad for p in parameters if p.grad is not None]

    # Compute the global L2 norm efficiently:
    # sum each gradient tensor's squared norm, then take the square root of the total.
    # This avoids allocating one giant concatenated vector.
    global_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads))

    if global_norm <= max_l2_norm:
        return  # already within budget — nothing to do

    # Scale every gradient down by the same factor so the global norm becomes
    # exactly max_l2_norm (epsilon in denominator guards against zero norm).
    scale = max_l2_norm / (global_norm + epsilon)
    for p in parameters:
        if p.grad is None:
            continue
        p.grad *= scale

