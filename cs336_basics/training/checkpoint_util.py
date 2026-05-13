import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    checkpoint_object = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "iteration_count": iteration,
    }

    torch.save(checkpoint_object, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint_object = torch.load(src)
    model.load_state_dict(checkpoint_object["model_state"])
    optimizer.load_state_dict(checkpoint_object["optimizer_state"])

    iteration_count = checkpoint_object["iteration_count"]

    return iteration_count
