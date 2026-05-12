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

    param_groups structure:
      self.param_groups is a list of dicts. Each dict contains:
        "params"       — list of nn.Parameter tensors belonging to this group
        "lr"           — learning rate α (may differ across groups)
        "betas"        — (β₁, β₂) moment decay rates
        "eps"          — ε numerical stability constant
        "weight_decay" — λ regularization coefficient
      Groups that omit a key inherit the value from `defaults` (set in __init__).
      Common use: give embedding tables a different lr or zero weight_decay
      compared to projection weights.

    Parameter shapes in a GPT-2 XL Transformer (V=50257, D=1600, F≈6400, L=48):
      Each parameter p is a leaf tensor; its shape identifies its role:
        token_embeddings.weight            (V, D)  — vocabulary embedding table
        position_embeddings.weight         (T, D)  — absolute positional embeddings
        layers.i.attn.q_proj.weight        (D, D)  — query projection (all heads packed)
        layers.i.attn.k_proj.weight        (D, D)  — key projection
        layers.i.attn.v_proj.weight        (D, D)  — value projection
        layers.i.attn.output_proj.weight   (D, D)  — output projection (heads → model dim)
        layers.i.ln1.weight                (D,)    — RMSNorm scale before attention
        layers.i.ffn.w1.weight             (F, D)  — FFN up/gate projection (SwiGLU w1)
        layers.i.ffn.w2.weight             (D, F)  — FFN down projection
        layers.i.ffn.w3.weight             (F, D)  — FFN gate projection (SwiGLU w3)
        layers.i.ln2.weight                (D,)    — RMSNorm scale before FFN
        ln_final.weight                    (D,)    — final RMSNorm scale
        lm_head.weight                     (V, D)  — output projection to vocab logits
      m and v in self.state[p] have the SAME shape as p (not scalars), so each
      individual element of each parameter gets its own adaptive learning rate.

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

                # p:    tensor of shape p.shape  — the parameter being updated.
                #       e.g. (V, D) for an embedding table, (D, D) for a
                #       projection weight, (D,) for an RMSNorm scale vector.
                # grad: tensor of the same shape as p — ∇_θ ℓ(θ; B_t).
                #       Every element of p has exactly one gradient element.
                grad = p.grad.data

                # Retrieve per-parameter state (persists across step() calls).
                state = self.state[p]
                t = state.get("t", 1)   # int scalar — step index, starts at 1
                # m and v start as scalar 0 on the first step. After the first
                # EMA update they become tensors of the same shape as p (via
                # broadcasting), and are stored that way in state from step 2 on.
                m = state.get("m", 0)   # same shape as p after step 1 (scalar 0 at init)
                v = state.get("v", 0)   # same shape as p after step 1 (scalar 0 at init)

                # Algorithm 1, line 7: bias-corrected learning rate.
                # scalar — a single number shared by every element of p this step.
                # Early in training β₁ᵗ and β₂ᵗ are close to 1, so this factor
                # is large, compensating for m and v being biased toward 0.
                cur_lr = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)

                # Algorithm 1, line 8: weight decay — pull θ toward zero.
                # Uses BASE lr α (not bias-corrected cur_lr) to decouple
                # regularization from the adaptive gradient scaling.
                # p.data: same shape as p, updated in-place.
                p.data -= lr * weight_decay * p.data

                # Algorithm 1, line 9: first moment (mean of gradients).
                # Result has same shape as grad (and p) — one EMA value per element.
                m = betas[0] * m + (1 - betas[0]) * grad

                # Algorithm 1, line 10: second moment (mean of squared gradients).
                # grad * grad is element-wise squaring, same shape as grad.
                # v is always ≥ 0 since it is an EMA of non-negative values.
                v = betas[1] * v + (1 - betas[1]) * grad * grad

                # Save updated moments back to state for the next step.
                state["m"] = m
                state["v"] = v

                # Algorithm 1, line 11: moment-adjusted parameter update.
                # All of m, v, p.data have the same shape as p.
                # cur_lr and ε are scalars that broadcast over every element.
                # m / (√v + ε) gives each element its own adaptive step size:
                # large consistent gradients → large v → small step;
                # small noisy gradients → small v → relatively larger step.
                p.data -= cur_lr * m / torch.sqrt(v + eps)

                # Increment step counter for bias correction next call.
                state["t"] = t + 1

        return loss
