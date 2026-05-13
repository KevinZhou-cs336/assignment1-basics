import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """Serialize model, optimizer, and iteration counter to disk.

    Training large models takes days or weeks. If the process crashes or the
    machine is preempted, all progress is lost without checkpointing. Saving
    periodically means we only lose the work since the last checkpoint.

    Three things are saved — each is necessary for a complete resume:

    1. model_state (model.state_dict()):
       All learnable parameter tensors (weights). This is the "knowledge" the
       model has accumulated. Without it, we would start from random weights.

    2. optimizer_state (optimizer.state_dict()):
       The AdamW moment estimates m and v for every parameter, plus the step
       counter t. These are NOT the weights — they are the optimizer's memory
       of how gradients have been moving. Without them, AdamW resets to its
       initial state and the adaptive learning rates are lost, often causing a
       loss spike at the resumption point.

    3. iteration_count (iteration):
       The training step number. Needed so the learning rate schedule
       (learning_rate_schedule(it=...)) resumes at the correct point rather
       than restarting from the warmup phase.

    Args:
        model:     The Transformer whose weights to save.
        optimizer: The AdamW optimizer whose moment state to save.
        iteration: Current training step number.
        out:       File path or file-like object to write the checkpoint to.
    """
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    checkpoint_object = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "iteration_count": iteration,
    }
    # torch.save uses pickle under the hood to serialise the dict of tensors.
    torch.save(checkpoint_object, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model, optimizer, and iteration counter from a checkpoint file.

    Reverses save_checkpoint exactly: deserialises the dict and calls
    load_state_dict on the model and optimizer to overwrite their current state.

    Args:
        src:       Path or file-like object to the saved checkpoint.
        model:     Model whose weights will be overwritten with the saved weights.
        optimizer: Optimizer whose moment state will be overwritten with saved state.

    Returns:
        The iteration count stored in the checkpoint, so the training loop
        can resume its step counter and LR schedule from the correct position.
    """
    checkpoint_object = torch.load(src)

    # Overwrite current model parameters with the saved weights.
    model.load_state_dict(checkpoint_object["model_state"])

    # Overwrite optimizer moment tensors (m, v) and step counters.
    # Without this, AdamW would treat the next step as step 1, losing its
    # accumulated gradient history and producing incorrect adaptive lr scaling.
    optimizer.load_state_dict(checkpoint_object["optimizer_state"])

    return checkpoint_object["iteration_count"]
