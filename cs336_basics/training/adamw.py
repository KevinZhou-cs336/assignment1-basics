import math
from typing import Callable, Optional

import torch


class AdamWOptimizer(torch.optim.Optimizer):
    """AdamW optimizer (Loshchilov & Hutter, 2019).

    Implements Algorithm 1 from the paper, which decouples weight decay from
    the adaptive gradient update. The full per-step update rule is:

        Algorithm 1 — AdamW (runs once per training step, t starts at 1):
        ──────────────────────────────────────────────────────────────────
        given: θ  (model parameters, arbitrary shape)
               m = 0  (first moment,  same shape as θ, initialized to zero)
               v = 0  (second moment, same shape as θ, initialized to zero)

        for t = 1, 2, ..., T:
            g      ← ∇_θ ℓ(θ; B_t)                    # gradient, shape (*)
            α_t    ← α · √(1 − β₂ᵗ) / (1 − β₁ᵗ)      # bias-corrected lr, scalar
            θ      ← θ − α · λ · θ                      # weight decay (line 8)
            m      ← β₁ · m + (1 − β₁) · g             # 1st moment update (line 9)
            v      ← β₂ · v + (1 − β₂) · g²            # 2nd moment update (line 10)
            θ      ← θ − α_t · m / (√v + ε)            # parameter update  (line 11)
        ──────────────────────────────────────────────────────────────────

    Key design choices vs vanilla Adam:
      - Weight decay is applied directly to θ (line 8), BEFORE the moment
        updates, and uses the BASE learning rate α (not the bias-corrected α_t).
        This decouples regularization from the adaptive gradient scaling.
      - m and v are stored per-parameter in self.state[p] so each parameter
        has its own independent moment history across steps.

    Args:
        params:       Parameters or param_groups to optimize. Can be a flat
                      iterable of nn.Parameter objects (all share the same
                      hyperparameters) or a list of dicts, each specifying
                      "params" and optionally overriding any hyperparameter.
        lr (α):       Base learning rate. Multiplied by the bias-correction
                      factor √(1−β₂ᵗ)/(1−β₁ᵗ) each step. Must be ≥ 0.
        betas (β₁,β₂):Exponential decay rates for the 1st and 2nd moment
                      estimates. Typical values: (0.9, 0.999) for general use,
                      (0.9, 0.95) for large language models.
                        β₁ controls how much the gradient history influences m.
                        β₂ controls how much the squared-gradient history
                          influences v (and thus the per-parameter step size).
        eps (ε):      Small constant added inside √v for numerical stability,
                      preventing division by zero when v is near 0. Typical: 1e-8.
        weight_decay (λ): L2 regularization coefficient. Pulls parameters
                      toward zero each step by subtracting α·λ·θ. Set to 0
                      to disable. Typical: 0.01–0.1.
    """

    def __init__(
        self,
        params,
        lr: float,
        betas: list[float, float],
        eps: float,
        weight_decay: float,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform one AdamW update step across all parameter groups.

        For each parameter p with a gradient, applies Algorithm 1 in order:
          1. Bias-correct the learning rate for this step t.
          2. Apply weight decay to p (line 8).
          3. Update first moment m (line 9).
          4. Update second moment v (line 10).
          5. Apply the moment-adjusted gradient update (line 11).

        Per-parameter state stored in self.state[p]:
          "t"  — int,   step counter, starts at 1, incremented each call.
          "m"  — tensor same shape as p (*),  first moment estimate (EMA of g).
          "v"  — tensor same shape as p (*),  second moment estimate (EMA of g²),
                 always non-negative.

        Args:
            closure: Optional callable that recomputes the loss (for line-search
                     compatibility with the torch.optim.Optimizer API).

        Returns:
            The loss value returned by closure, or None.
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            # Hyperparameters for this param group (may differ across groups)
            lr           = group["lr"]           # α: base learning rate, scalar
            betas        = group["betas"]        # (β₁, β₂): moment decay rates
            eps          = group["eps"]          # ε: numerical stability constant
            weight_decay = group["weight_decay"] # λ: weight decay coefficient

            for p in group["params"]:
                if p.grad is None:
                    continue  # parameter received no gradient this step — skip

                # p:    tensor of shape (*)  — the parameter being updated
                # grad: tensor of shape (*)  — ∇_θ ℓ(θ; B_t), same shape as p
                grad = p.grad.data

                # Retrieve per-parameter state (persists across step() calls).
                # On the first step, t=1 and m/v default to scalar 0, which
                # broadcasts correctly against grad when computing the updates.
                state = self.state[p]
                t = state.get("t", 1)   # step index, starts at 1 per Algorithm 1
                m = state.get("m", 0)   # first moment:  shape (*) after first step
                v = state.get("v", 0)   # second moment: shape (*) after first step

                # Algorithm 1, line 7: bias-corrected learning rate (scalar).
                # Early in training β₁ᵗ and β₂ᵗ are close to 1, so this factor
                # is large, compensating for m and v being biased toward 0.
                cur_lr = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)

                # Algorithm 1, line 8: weight decay — pull θ toward zero.
                # Uses BASE lr α (not bias-corrected α_t) to decouple
                # regularization from the adaptive gradient scaling.
                # p: shape (*) → shape (*), in-place.
                p.data -= lr * weight_decay * p.data

                # Algorithm 1, line 9: first moment (mean of gradients).
                # Exponential moving average of g; shape (*).
                m = betas[0] * m + (1 - betas[0]) * grad

                # Algorithm 1, line 10: second moment (mean of squared gradients).
                # Exponential moving average of g²; shape (*), always ≥ 0.
                v = betas[1] * v + (1 - betas[1]) * grad * grad

                # Save updated moments back to state for the next step.
                state["m"] = m
                state["v"] = v

                # Algorithm 1, line 11: moment-adjusted parameter update.
                # m / (√v + ε) normalizes the gradient by its historical RMS,
                # giving larger steps for parameters with small, consistent
                # gradients and smaller steps for noisy or large-gradient ones.
                # All tensors are shape (*); cur_lr and ε are scalars.
                p.data -= cur_lr * m / torch.sqrt(v + eps)

                # Increment step counter for bias correction next call.
                state["t"] = t + 1

        return loss
