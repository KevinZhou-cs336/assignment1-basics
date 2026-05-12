import math
from collections.abc import Callable
from typing import Optional

import torch


class SGD(torch.optim.Optimizer):
    """SGD optimizer with inverse-square-root learning rate decay (equation 20).

    Implements the update rule:
        θ_{t+1} = θ_t - (α / √(t+1)) · ∇L(θ_t; B_t)

    where t is the per-parameter step count (starting at 0) and α is the
    initial learning rate. The effective learning rate shrinks as 1/√(t+1),
    taking successively smaller steps over the course of training.

    This implementation is provided directly in the assignment (section 4.2.1)
    as a worked example of how to subclass torch.optim.Optimizer. It shows:
      - Passing hyperparameters through `defaults` to the base class.
      - Iterating over param_groups and params inside step().
      - Reading and writing per-parameter state via self.state[p].
      - Updating parameters in-place via p.data rather than p itself.
    """

    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Perform a single SGD update step.

        Args:
            closure: Optional callable that recomputes the loss. Included to
                     comply with the torch.optim.Optimizer API; not used in
                     typical training loops.

        Returns:
            The loss value returned by closure, or None if no closure given.
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]          # learning rate α for this parameter group

            for p in group["params"]:
                if p.grad is None:
                    continue          # skip parameters that received no gradient

                state = self.state[p]      # per-parameter state dict (persists across steps)
                t = state.get("t", 0)      # step count for this parameter, starting at 0
                grad = p.grad.data         # ∇L(θ_t ; B_t), detached from autograd graph

                # Decayed update: θ_{t+1} = θ_t - (α / √(t+1)) · g
                # p.data is modified in-place so the autograd graph is not affected.
                p.data -= lr / math.sqrt(t + 1) * grad

                state["t"] = t + 1         # increment step counter for the next call

        return loss
