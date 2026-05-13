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
import os

import numpy as np
import torch

from cs336_basics.training import cross_entropy
from cs336_basics.training.adamw import AdamWOptimizer
from cs336_basics.training.checkpoint_util import load_checkpoint, save_checkpoint
from cs336_basics.training.data_util import get_batch
from cs336_basics.training.learning_utils import (
    gradient_clipping,
    learning_rate_schedule,
)
from cs336_basics.transformers.transformer_language_model import (
    TransformerLanguageModel,
)


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
    print(f"[Training Loop]: args are {args}")

    # Create checkpoint directory upfront so save_checkpoint never fails on a missing path.
    print(f"[Preparation]: make checkpoint dir at {args.checkpoint_dir}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # mmap_mode='r' memory-maps the file: the OS loads pages on demand rather than
    # reading the entire dataset into RAM. Essential for datasets larger than available memory.
    train_dataset = np.load(args.train_data, mmap_mode="r")
    validation_dataset = np.load(args.val_data, mmap_mode="r")

    transformer_lm = TransformerLanguageModel(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    # Move all parameters to the target device before creating the optimizer.
    # If we moved them after, the optimizer's m/v tensors (created on first step())
    # would still be on CPU while parameters are on GPU, causing a device mismatch.
    transformer_lm.to(args.device)

    # Initialize optimizer once and keep it alive for the entire training run.
    # AdamW accumulates per-parameter moment estimates (m, v) across steps in
    # self.state[p]. Creating a new optimizer each step would discard that history,
    # turning AdamW into plain gradient descent with no adaptive learning rates.
    adamw_optimizer = AdamWOptimizer(
        params=transformer_lm.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # When resuming, restore model weights, optimizer state (m/v moments), and the
    # step counter so the LR schedule continues from where training left off.
    iteration_start = 0
    if args.resume_from is not None:
        iteration_start = load_checkpoint(
            args.resume_from, transformer_lm, adamw_optimizer
        )

    for step in range(iteration_start, args.total_steps):
        # Sample B random windows from the training corpus.
        # input_tokens and target_tokens are both (B, T); targets are inputs shifted right by 1.
        input_tokens, target_tokens = get_batch(
            dataset=train_dataset,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
        )

        # Forward pass: causal attention ensures position i only attends to positions ≤ i,
        # so logits[b, i, :] predicts the distribution over the next token after position i.
        # PyTorch records every operation into a computation graph for backward().
        logits = transformer_lm(input_tokens)

        # Cross-entropy averages -log(p_correct) over all B×T positions.
        # The result is a scalar tensor still attached to the computation graph.
        loss = cross_entropy(logits, target_tokens)

        # Log before backward so loss.item() detaches the value without disturbing the graph.
        if (step - iteration_start + 1) % args.log_interval == 0:
            print(
                f"[Training] step {step - iteration_start + 1} | loss {loss.item():.4f} | lr {adamw_optimizer.param_groups[0]['lr']:.2e}"
            )

        # Backpropagate: walk the computation graph in reverse, applying chain rule at each op.
        # Each parameter's .grad is filled with ∂loss/∂p — the average gradient over B×T positions.
        loss.backward()

        # If the global L2 norm of all gradients exceeds grad_clip, scale them down proportionally.
        # Prevents "exploding gradients" that would shove parameters far from a good region.
        if args.grad_clip > 0:
            gradient_clipping(transformer_lm.parameters(), args.grad_clip)

        # Compute the learning rate for this step (warmup → cosine decay → min_lr floor).
        # The LR lives in param_groups, not inside the optimizer itself, so we update it
        # manually before each step(). The optimizer has no built-in schedule awareness.
        lr = learning_rate_schedule(
            it=step,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_steps,
            cosine_cycle_iters=args.cosine_cycle_steps,
        )
        for group in adamw_optimizer.param_groups:
            group["lr"] = lr

        # Apply the AdamW update: use .grad, m, v, and the bias-corrected lr to nudge each parameter.
        adamw_optimizer.step()

        # Reset all .grad tensors to zero. PyTorch accumulates gradients by default —
        # without this, the next backward() would add onto this step's gradients.
        adamw_optimizer.zero_grad()

        # Periodically snapshot model weights + optimizer state so training can be resumed
        # after a crash or preemption without starting from scratch.
        if (
            args.checkpoint_interval
            and (step - iteration_start + 1) % args.checkpoint_interval == 0
        ):
            save_checkpoint(
                transformer_lm,
                adamw_optimizer,
                step,
                f"{args.checkpoint_dir}/training_{step}.ckp",
            )

        # Run validation only every val_interval steps — skip otherwise.
        if (step - iteration_start + 1) % args.val_interval != 0:
            continue

        # Switch to eval mode: disables dropout so all neurons are active and outputs are
        # deterministic. Pair with torch.no_grad() which skips building the computation
        # graph, saving the activation memory that backward() would otherwise need.
        transformer_lm.eval()
        validation_loss_sum = 0
        for _ in range(args.val_steps):
            validation_input_tokens, validation_target_tokens = get_batch(
                dataset=validation_dataset,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
            )
            with torch.no_grad():
                logits = transformer_lm(validation_input_tokens)
                loss = cross_entropy(logits, validation_target_tokens)
                validation_loss_sum += loss

        # Average over val_steps batches for a more stable estimate than a single batch.
        avg_val_loss = validation_loss_sum.item() / args.val_steps
        print(f"[Validation] step {step - iteration_start + 1} | avg loss {avg_val_loss:.4f}")

        # Restore training mode so dropout is active again for the next training step.
        transformer_lm.train()


if __name__ == "__main__":
    main()
