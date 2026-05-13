import numpy as np
import numpy.typing as npt
import torch


# Naive implementation using a Python for loop — kept for reference.
# Functionally correct but slow: builds B separate lists and calls torch.tensor
# once per list, which involves repeated Python-level allocation and copying.
# The vectorised version below replaces the loop with NumPy index broadcasting,
# computing all B windows in a single array operation.
#
# def get_batch(
#     dataset: npt.NDArray, batch_size: int, context_length: int, device: str
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     start_positions = torch.randint(0, len(dataset) - context_length, (batch_size,))
#     input_tokens = [dataset[i : i + context_length] for i in start_positions]
#     target_tokens = [dataset[i + 1 : i + 1 + context_length] for i in start_positions]
#     return torch.tensor(input_tokens, device=device), torch.tensor(target_tokens, device=device)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) token sequences from the dataset.

    Background — how language model training works:
        The dataset is one long sequence of integer token IDs, e.g.:
            [12, 450, 7, 903, 21, 6, ...]

        At each training step we want to show the model B independent windows
        into this sequence (a "batch"), teach it to predict the next token at
        every position inside each window.

        For a window starting at position i with length T:
            input   = tokens[i],   tokens[i+1], ..., tokens[i+T-1]   (T tokens)
            targets = tokens[i+1], tokens[i+2], ..., tokens[i+T]     (same T tokens, shifted by 1)

        At position k inside the window, the model sees tokens[i..i+k] and must
        predict tokens[i+k+1]. Packaging the shift into a parallel "targets"
        tensor lets PyTorch compute all T predictions in a single forward pass.

    Args:
        dataset:        1-D array of integer token IDs (the full tokenised corpus).
        batch_size:     Number of independent sequences to sample (B).
        context_length: Length of each sequence window (T).
        device:         PyTorch device string, e.g. "cpu", "cuda", or "mps".

    Returns:
        input_tokens:  LongTensor of shape (B, T) — the input token sequences.
        target_tokens: LongTensor of shape (B, T) — the target token sequences,
                       each identical to the corresponding input row but shifted
                       one position to the right.
    """
    # Sample B random starting positions uniformly from [0, len(dataset) - T).
    # The upper bound excludes the last T positions so that every window of
    # length T+1 (input + one extra target token) fits within the dataset.
    #
    # Shape (batch_size, 1): the extra dimension of size 1 is intentional —
    # it allows NumPy broadcasting when we add the offsets below.
    start_positions = torch.randint(0, len(dataset) - context_length, (batch_size, 1))

    # Create a row vector [0, 1, 2, ..., context_length - 1].
    # When added to start_positions (shape batch_size×1), broadcasting produces
    # a (batch_size, context_length) index matrix where row b is:
    #   [start_positions[b], start_positions[b]+1, ..., start_positions[b]+T-1]
    offsets = torch.arange(context_length)

    # Index into the dataset with the (batch_size, context_length) index matrix.
    # input_tokens[b, k]  = dataset[start_positions[b] + k]
    # target_tokens[b, k] = dataset[start_positions[b] + k + 1]
    # Both arrays have shape (batch_size, context_length).
    input_tokens = dataset[start_positions + offsets]
    target_tokens = dataset[start_positions + offsets + 1]

    # Convert from NumPy arrays to PyTorch tensors and move to the target device.
    # dtype is inferred from the dataset array (typically int64 / torch.long).
    return (
        torch.tensor(input_tokens, device=device),
        torch.tensor(target_tokens, device=device),
    )
