# AdamW Resource Accounting (Problem: adamw_accounting)

**Assignment reference:** Section 4.3, Problem `adamw_accounting` — "Resource accounting for training with AdamW (2 points)"
**PDF location:** CS336 Assignment 1 Basics, page 32–33

All tensors are float32 (4 bytes each).

---

## Notation

| Symbol | Meaning | GPT-2 XL value |
|--------|---------|----------------|
| B | batch\_size | (variable) |
| T | context\_length | 1,024 |
| D | d\_model | 1,600 |
| H | num\_heads | 25 |
| d = D/H | d\_head (per-head dim) | 64 |
| F | d\_ff = ⌈(8/3)·D⌉ rounded to mult of 64 | 4,288 |
| L | num\_layers | 48 |
| V | vocab\_size | 50,257 |
| P | total parameter count | 1,640,452,800 |

Parameter count P (from `transformer_accounting`):
```
P = 2VD + D + L × (2D + 4D² + 3DF)
  = 2·50257·1600 + 1600 + 48 × (3200 + 10,240,000 + 20,582,400)
  = 160,822,400 + 1,600 + 48 × 30,825,600
  = 1,640,452,800
```

---

## (a) Peak Memory Breakdown

### Parameters
Each of the P parameters is stored as a float32 value.

```
Memory_params = 4P bytes
```

### Gradients
After the backward pass, each parameter has a gradient of the same shape.

```
Memory_grads = 4P bytes
```

### Optimizer State (AdamW)
AdamW maintains two moment tensors per parameter — the first moment **m** and second moment **v** — both the same shape as the parameter tensor.

```
Memory_optimizer = 2 × 4P = 8P bytes
```

### Activations
During the forward pass, intermediate tensors must be retained for the backward pass. We account for the output of each listed sub-operation.

**Per Transformer block (×L):**

| Sub-operation | Output shape | Elements |
|---------------|-------------|---------|
| RMSNorm before attention | (B, T, D) | BTD |
| RMSNorm before FFN | (B, T, D) | BTD |
| Q projection | (B, T, D) | BTD |
| K projection | (B, T, D) | BTD |
| V projection | (B, T, D) | BTD |
| QKᵀ scores | (B, H, T, T) | BHT² |
| Softmax weights | (B, H, T, T) | BHT² |
| Weighted sum of V | (B, T, D) | BTD |
| Output projection | (B, T, D) | BTD |
| FFN W1 output (gate, before SiLU) | (B, T, F) | BTF |
| FFN SiLU output | (B, T, F) | BTF |
| FFN W3 output (value branch) | (B, T, F) | BTF |
| FFN element-wise product | (B, T, F) | BTF |
| FFN W2 output | (B, T, D) | BTD |

Per-block total: **8BTD + 2BHT² + 4BTF** elements

**Global (computed once):**

| Component | Output shape | Elements |
|-----------|-------------|---------|
| Final RMSNorm | (B, T, D) | BTD |
| Output embedding / LM head (logits) | (B, T, V) | BTV |
| Cross-entropy (uses logits in-place) | scalar | — |

Global total: **BTD + BTV** elements

**Total activation memory:**
```
Memory_activations = 4 × [L(8BTD + 2BHT² + 4BTF) + BTD + BTV]  bytes
```

With F = (8/3)D substituted symbolically (4BTF = (32/3)BTD):
```
= 4 × [L((56/3)BTD + 2BHT²) + BTD + BTV]  bytes
```

### Total Peak Memory

```
Memory_total = Memory_params + Memory_grads + Memory_optimizer + Memory_activations
             = 4P + 4P + 8P + 4[L(8BTD + 2BHT²+ 4BTF) + BTD + BTV]
             = 16P + 4L(8BTD + 2BHT² + 4BTF) + 4BTD + 4BTV  bytes
```

---

## (b) GPT-2 XL Instantiation — Maximum Batch Size on 80 GB

Substituting the GPT-2 XL values (T=1024, D=1600, H=25, F=4288, L=48, V=50257, P=1,640,452,800):

### Static memory (independent of batch size)

```
16P = 16 × 1,640,452,800 = 26,247,244,800 bytes  ≈ 24.45 GB
```

Breakdown:
- Parameters: 4P ≈ 6.11 GB
- Gradients:  4P ≈ 6.11 GB
- Optimizer (m + v): 8P ≈ 12.22 GB

### Activation memory per batch element (B = 1)

Per-block elements:
```
8 × T × D           = 8 × 1024 × 1600         = 13,107,200
2 × H × T × T       = 2 × 25 × 1024 × 1024    = 52,428,800
4 × T × F           = 4 × 1024 × 4288          = 17,563,648
                                                 ─────────────
Per block total:                                  83,099,648
```

All 48 blocks: `48 × 83,099,648 = 3,988,783,104`

Global:
```
T × D = 1024 × 1600   =  1,638,400
T × V = 1024 × 50257  = 51,463,168
                        ──────────
Global total:           53,101,568
```

Total activation elements (B=1): `3,988,783,104 + 53,101,568 = 4,041,884,672`

Activation bytes per batch element: `4 × 4,041,884,672 = 16,167,538,688 bytes ≈ 15.06 GB`

> The attention score matrices (2BHT² = 52M elements/block × 48 blocks ≈ 2.5B elements) dominate
> because H × T² ≫ T × D for these dimensions (ratio ≈ 16×).

### Expression

```
Memory_total (bytes) = 16,167,538,688 × batch_size + 26,247,244,800
                     ≈ 15.06 × B  +  24.45  GB
```

### Maximum batch size within 80 GB

```
15.06 × B + 24.45 ≤ 80
15.06 × B ≤ 55.55
B ≤ 3.69
```

**Maximum batch size = 3**

---

## (c) FLOPs for One AdamW Step

AdamW applies five elementwise operations per parameter (see Algorithm 1):

| Operation | Ops per element |
|-----------|----------------|
| Bias-corrected lr α_t (scalar, amortized ≈ 0/element) | ~0 |
| Weight decay: θ ← θ − α·λ·θ | 2 (1 mul, 1 sub) |
| m update: m ← β₁·m + (1−β₁)·g | 3 (2 mul, 1 add) |
| v update: v ← β₂·v + (1−β₂)·g² | 4 (1 sq, 2 mul, 1 add) |
| θ update: θ ← θ − α_t·m/(√v + ε) | 5 (1 sqrt, 1 add, 1 div, 1 mul, 1 sub) |
| **Total** | **14 ops** |

Since all operations are elementwise over P parameters:

```
FLOPs_AdamW ≈ 14P
```

For GPT-2 XL:
```
14 × 1,640,452,800 ≈ 22.97 GFLOPs ≈ 23 GFLOPs
```

**Comparison to training step:**
- Forward pass: 3.52 TFLOPs
- Backward pass (2× forward): 7.04 TFLOPs
- AdamW step: 0.023 TFLOPs

AdamW accounts for only ~0.2% of total training-step FLOPs; it is dominated by the matrix multiplies in forward/backward.

---

## (d) Training Time on a Single H100 at 50% MFU

**Setup:**
- Steps: 400,000
- Batch size: B = 1,024
- H100 peak throughput (float32/TF32): 495 TFLOPs/s
- MFU: 50% → effective throughput = 247.5 TFLOPs/s
- Backward pass FLOPs = 2 × forward pass FLOPs (per problem assumption)

**FLOPs per training step:**

Each step processes B = 1,024 sequences, each of length T = 1,024 tokens.

```
Forward FLOPs/step  = B × 3,519,984,435,200  ≈ 3,604.5 TFLOPs
Backward FLOPs/step = 2 × 3,604.5            ≈ 7,208.9 TFLOPs
AdamW FLOPs/step    ≈ 0.023 TFLOPs           (negligible)
─────────────────────────────────────────────────────────────
Total/step          ≈ 10,813.4 TFLOPs
```

**Time per step:**
```
10,813.4 TFLOPs / 247.5 TFLOPs/s ≈ 43.7 seconds
```

**Total training time:**
```
400,000 steps × 43.7 s/step = 17,480,000 s
                             ÷ 3,600 s/hr
                             ≈ 4,856 hours
                             ≈ 202 days
```

**Deliverable:** Training GPT-2 XL for 400K steps with batch size 1,024 at 50% MFU on a single H100 would take approximately **4,856 hours (~202 days)**. This is because each step processes 1M tokens and requires ~10.8 PFLOPs (forward + backward), while the H100 delivers only ~247.5 TFLOPs/s at 50% utilization.
