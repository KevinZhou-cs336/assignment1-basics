"""Training script for the Transformer LM (Problem: training_together, Section 5).

Usage examples:

  # TinyStories small run (CPU / Apple Silicon):
  python -m cs336_basics.training.train \
      --train-data data/tinystories_train.npy \
      --val-data   data/tinystories_val.npy   \
      --vocab-size 10000                       \
      --context-length 256                     \
      --d-model 512                            \
      --num-heads 16                           \
      --num-layers 4                           \
      --batch-size 32                          \
      --total-steps 5000                       \
      --checkpoint-dir checkpoints/tinystories

  # Resume from a checkpoint:
  python -m cs336_basics.training.train ... --resume-from checkpoints/tinystories/step_1000.pt
"""

import argparse

from wandb.util import np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Transformer language model from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    data = parser.add_argument_group("data")
    data.add_argument(
        "--train-data",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the memory-mapped NumPy array of training token IDs (.npy).",
    )
    data.add_argument(
        "--val-data",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the memory-mapped NumPy array of validation token IDs (.npy).",
    )

    # ── Model architecture ────────────────────────────────────────────────────
    model = parser.add_argument_group("model architecture")
    model.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
        help="Vocabulary size (must match the tokenizer used to produce the data).",
    )
    model.add_argument(
        "--context-length",
        type=int,
        default=256,
        help="Maximum sequence length (T). Determines the RoPE sin/cos buffer size.",
    )
    model.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Hidden / embedding dimension (D).",
    )
    model.add_argument(
        "--num-heads",
        type=int,
        default=16,
        help="Number of attention heads (H). Must divide d-model evenly.",
    )
    model.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of stacked Transformer blocks (L).",
    )
    model.add_argument(
        "--d-ff",
        type=int,
        default=None,
        help=(
            "FFN intermediate dimension (F). "
            "Defaults to the nearest multiple of 64 to (8/3) * d-model."
        ),
    )
    model.add_argument(
        "--rope-theta",
        type=float,
        default=10_000.0,
        help="RoPE base frequency theta (Θ).",
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    opt = parser.add_argument_group("optimiser (AdamW)")
    opt.add_argument(
        "--max-lr",
        type=float,
        default=1e-3,
        help="Peak learning rate (α_max), reached at the end of warmup.",
    )
    opt.add_argument(
        "--min-lr",
        type=float,
        default=1e-4,
        help="Minimum learning rate (α_min), held after cosine decay ends.",
    )
    opt.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of linear warm-up steps before cosine decay begins.",
    )
    opt.add_argument(
        "--cosine-cycle-steps",
        type=int,
        default=None,
        help=(
            "Step at which cosine decay reaches min-lr. "
            "Defaults to total-steps (decay runs for the full training run)."
        ),
    )
    opt.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="AdamW weight decay coefficient (λ).",
    )
    opt.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="AdamW β₁ (first-moment decay rate).",
    )
    opt.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="AdamW β₂ (second-moment decay rate). 0.95 is typical for LLMs.",
    )
    opt.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="AdamW ε (numerical stability constant in the denominator).",
    )
    opt.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Maximum global L2 norm for gradient clipping. Set to 0 to disable.",
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    loop = parser.add_argument_group("training loop")
    loop.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of sequences per training step (B).",
    )
    loop.add_argument(
        "--total-steps",
        type=int,
        default=5_000,
        help="Total number of gradient update steps.",
    )
    loop.add_argument(
        "--val-interval",
        type=int,
        default=500,
        help="Evaluate validation loss every this many steps.",
    )
    loop.add_argument(
        "--val-steps",
        type=int,
        default=20,
        help="Number of batches to average for each validation loss estimate.",
    )
    loop.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log training loss to stdout every this many steps.",
    )
    loop.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='PyTorch device string: "cpu", "cuda", "cuda:0", "mps", etc.',
    )

    # ── Checkpointing ─────────────────────────────────────────────────────────
    ckpt = parser.add_argument_group("checkpointing")
    ckpt.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        metavar="DIR",
        help="Directory to write checkpoint files into.",
    )
    ckpt.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1_000,
        help="Save a checkpoint every this many steps.",
    )
    ckpt.add_argument(
        "--resume-from",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a checkpoint file to resume training from.",
    )

    # ── Logging (Weights & Biases) ────────────────────────────────────────────
    wb = parser.add_argument_group("Weights & Biases logging (optional)")
    wb.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name. If not set, W&B logging is disabled.",
    )
    wb.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name. If not set, W&B auto-generates one.",
    )

    args = parser.parse_args()

    # ── Derived defaults ──────────────────────────────────────────────────────
    # Fill in arguments whose defaults depend on other arguments.
    if args.d_ff is None:
        # (8/3) * d_model rounded up to the nearest multiple of 64.
        raw = (8 * args.d_model) / 3
        args.d_ff = int(raw / 64 + 0.5) * 64

    if args.cosine_cycle_steps is None:
        # By default, decay learning rate over the full training run.
        args.cosine_cycle_steps = args.total_steps

    return args


def main():
    args = parse_args()

    # TODO: load training and validation datasets with np.load(..., mmap_mode='r')
    dataset = np.load
    # TODO: instantiate TransformerLanguageModel with args.vocab_size, args.context_length,
    #       args.d_model, args.num_heads, args.num_layers, args.d_ff, args.rope_theta
    transformer_lm = TransformerLanguageModel()
    # TODO: instantiate AdamWOptimizer with args.max_lr, (args.beta1, args.beta2),
    #       args.eps, args.weight_decay

    # TODO: if args.resume_from is not None, call load_checkpoint(...)

    # TODO: training loop — for step in range(start_step, args.total_steps):
    #   - sample a batch with get_batch(...)
    #   - forward pass + cross_entropy loss
    #   - loss.backward()
    #   - gradient_clipping(model.parameters(), args.grad_clip) if args.grad_clip > 0
    #   - set learning rate with learning_rate_schedule(...) on the optimiser
    #   - optimizer.step() then optimizer.zero_grad()
    #   - log / validate / checkpoint at the appropriate intervals

    pass


if __name__ == "__main__":
    main()
