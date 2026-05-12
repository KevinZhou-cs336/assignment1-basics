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

**What is an activation?**
Every intermediate tensor produced during the forward pass that must be kept in memory for the backward pass is called an activation. The rule is simple: if operation `Y = f(X)` is non-trivial, then `X` must be saved so that the backward pass can compute gradients. The output `Y` is also saved if any later operation needs it as input.

**What dimensions appear and what do they mean?**

| Symbol | What it counts | Intuition |
|--------|---------------|-----------|
| B | batch size | how many independent text sequences we process simultaneously |
| T | sequence length (context\_length = 1024) | how many tokens are in each sequence |
| D | d\_model = 1600 | each token is represented as a vector of 1600 numbers |
| H | num\_heads = 25 | attention is split into 25 independent "heads" |
| d = D/H | per-head dim = 64 | each head works with a 64-number slice of the D-dim vector |
| F | d\_ff = 4288 ≈ (8/3)D | FFN intermediate width, wider than D to add capacity |
| V | vocab\_size = 50257 | number of possible tokens; the final output has one score per token |

---

#### Per Transformer block (×L = 48 blocks)

Every block receives a tensor of shape `(B, T, D)` — B sequences, each T tokens long, each token a D-dimensional vector — and produces the same shape at the end. Inside the block, several intermediate tensors are created.

---

##### RMSNorm before attention — output shape: **(B, T, D)**

```
input:  (B, T, D)   — residual stream coming into the block
output: (B, T, D)   — normalized version; same shape because RMSNorm
                       only rescales each token vector, does not change its size
```

Why saved? The backward pass needs this output as the input to the Q/K/V projection weight-gradient computation:
`grad_W_Q = RMSNorm_output.T @ grad_Q`

Elements: **B × T × D**

---

##### Q projection — output shape: **(B, T, D)**

```
operation:  Q = RMSNorm_output @ W_Q
W_Q shape:  (D, D)  — D inputs → D query outputs per token
output:     (B, T, D)
```

Why `(B, T, D)`?
- B: one query matrix per sequence in the batch
- T: one query vector per token (each token asks its own question)
- D: the query vector has D dimensions (it is later split into H heads of d=D/H each)

Why saved? The backward pass for computing `grad_K` during `QKᵀ` needs Q:
`grad_K = grad_scores.T @ Q`

Elements: **B × T × D**

---

##### K projection — output shape: **(B, T, D)**

```
operation:  K = RMSNorm_output @ W_K
output:     (B, T, D)
```

Same reasoning as Q. Every token produces a key vector that other tokens will compare their query against.

Why saved? Backward of `QKᵀ` needs K to compute `grad_Q`:
`grad_Q = grad_scores @ K`

Elements: **B × T × D**

---

##### V projection — output shape: **(B, T, D)**

```
operation:  V = RMSNorm_output @ W_V
output:     (B, T, D)
```

Each token produces a value vector — the content it contributes to other tokens that attend to it.

Why saved? Backward of the weighted sum `softmax_weights @ V` needs V to compute `grad_softmax_weights`:
`grad_softmax_weights = grad_attn_output @ V.T`

Elements: **B × T × D**

---

##### QKᵀ attention scores — output shape: **(B, H, T, T)**

```
operation: scores = Q @ K.T      (after reshaping Q and K into H heads)
Q reshaped: (B, H, T, d)         — B sequences, H heads, T queries,  d=D/H dims each
K reshaped: (B, H, T, d)         — B sequences, H heads, T keys,     d=D/H dims each
output:     (B, H, T, T)
```

Why `(B, H, T, T)`?
- B: one attention matrix per sequence
- H: each of the 25 heads computes its own attention pattern independently
- first T: one row per query token (the token that is "looking")
- second T: one column per key token (the token being "looked at")

Entry `[b, h, i, j]` is the raw dot-product score: "in sequence b, head h, how relevant is key token j to query token i?"

Why saved? It is the input to softmax; softmax backward does not need it (only the softmax output is needed), but in standard PyTorch autograd this tensor is retained as the input to the softmax node.

Elements: **B × H × T × T**

For GPT-2 XL (B=1): 1 × 25 × 1024 × 1024 = **26,214,400** — this single tensor is already 16× larger than one (T×D) tensor.

---

##### Softmax attention weights — output shape: **(B, H, T, T)**

```
operation: weights = softmax(scores / sqrt(d), dim=-1)
output:    (B, H, T, T)   — same shape as scores; each row now sums to 1
```

Why same shape? Softmax converts raw scores into probabilities but does not add or remove dimensions.

Why saved? The softmax backward needs the softmax output to compute `grad_scores`:
`grad_scores[i] = weights[i] * (grad_weights[i] - sum(grad_weights[i] * weights[i]))`

Elements: **B × H × T × T**

---

##### Weighted sum of values (attention output) — output shape: **(B, T, D)**

```
operation: attn_output = weights @ V     (per head, then concatenate)
weights:   (B, H, T, T)
V:         (B, H, T, d)
per-head:  (B, H, T, d)   →  concatenated → (B, T, H×d) = (B, T, D)
```

Why `(B, T, D)`?
- Each of the T query tokens receives a weighted mixture of all T value vectors.
- After concatenating H heads (each of size d=64), the result is H×d = D = 1600 per token.

Why saved? It is the input to the output projection; backward of output projection needs it to compute `grad_W_O`:
`grad_W_O = attn_output.T @ grad_output_proj`

Elements: **B × T × D**

---

##### Output projection — output shape: **(B, T, D)**

```
operation:  out = attn_output @ W_O
W_O shape:  (D, D)
output:     (B, T, D)
```

This linearly mixes information across the H heads back into a single D-dimensional token vector.

Why saved? It is added to the residual stream; the tensor flows to RMSNorm2 which needs it as its input.

Elements: **B × T × D**

---

##### RMSNorm before FFN — output shape: **(B, T, D)**

```
input:  (B, T, D)  — residual after attention (block_input + output_proj)
output: (B, T, D)  — normalized; same shape
```

Why saved? The FFN weight-gradient computations (for W1, W3) need this as their input:
`grad_W1 = RMSNorm2_output.T @ grad_W1_output`

Elements: **B × T × D**

---

##### FFN W1 gate projection — output shape: **(B, T, F)**

```
operation:  gate = RMSNorm2_output @ W1.T
W1 shape:   (F, D)   — maps D dims → F dims
output:     (B, T, F)
```

Why `(B, T, F)`?
- F = 4288 ≈ 2.68 × D: the FFN first expands each token's representation to a wider space (more capacity for computation).
- The `F` dimension is this wider hidden space.

Why saved? It is the input to SiLU; SiLU backward needs the pre-activation value to compute the derivative:
`SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))`

Elements: **B × T × F**

---

##### FFN SiLU output — output shape: **(B, T, F)**

```
operation:  silu_out = SiLU(gate)   — elementwise; SiLU(x) = x * sigmoid(x)
output:     (B, T, F)               — same shape as gate
```

Why same shape? SiLU is applied independently to each of the B×T×F elements.

Why saved? It is one of the two inputs to the element-wise product (SwiGLU gate). Backward of the product needs both inputs to compute each gradient:
`grad_W3_output = grad_product * silu_out`

Elements: **B × T × F**

---

##### FFN W3 value projection — output shape: **(B, T, F)**

```
operation:  value = RMSNorm2_output @ W3.T
W3 shape:   (F, D)   — same shape as W1, maps D → F
output:     (B, T, F)
```

This is the second branch of SwiGLU: it produces the "value" that will be gated.

Why saved? Backward of the element-wise product needs it:
`grad_silu_out = grad_product * value`

Elements: **B × T × F**

---

##### FFN element-wise product (SwiGLU gate) — output shape: **(B, T, F)**

```
operation:  product = silu_out * value    — elementwise multiplication
output:     (B, T, F)
```

This is the SwiGLU gating: the value branch is multiplied element-by-element by the gated activation branch.

Why saved? It is the input to W2; backward of W2 needs it to compute `grad_W2`:
`grad_W2 = product.T @ grad_W2_output`

Elements: **B × T × F**

---

##### FFN W2 down projection — output shape: **(B, T, D)**

```
operation:  ffn_output = product @ W2.T
W2 shape:   (D, F)   — maps F dims back down to D dims
output:     (B, T, D)
```

Why `(B, T, D)`? We compress the F-dimensional representation back to D so the FFN output can be added to the residual stream (which has shape B×T×D).

Why saved? It is added to the residual stream; the result is the block's output tensor, which is the next block's input (and is counted there).

Elements: **B × T × D**

---

##### Per-block total

| Tensor | Shape | Elements |
|--------|-------|---------|
| RMSNorm1 output | (B,T,D) | BTD |
| Q | (B,T,D) | BTD |
| K | (B,T,D) | BTD |
| V | (B,T,D) | BTD |
| QKᵀ scores | (B,H,T,T) | BHT² |
| Softmax weights | (B,H,T,T) | BHT² |
| Weighted sum of V | (B,T,D) | BTD |
| Output projection | (B,T,D) | BTD |
| RMSNorm2 output | (B,T,D) | BTD |
| W1 gate output | (B,T,F) | BTF |
| SiLU output | (B,T,F) | BTF |
| W3 value output | (B,T,F) | BTF |
| Element-wise product | (B,T,F) | BTF |
| W2 FFN output | (B,T,D) | BTD |

**Per-block total: 8BTD + 2BHT² + 4BTF elements**

(The 8 BTD tensors are: RMSNorm1, Q, K, V, weighted-sum, output-proj, RMSNorm2, W2-output)

---

#### Global components (computed once, not ×L)

---

##### Final RMSNorm — output shape: **(B, T, D)**

```
input:  (B, T, D)  — last block's output
output: (B, T, D)  — normalized; same shape
```

This is the normalization applied to the final hidden states before the LM head.

Why saved? The LM head (W_lm) backward needs it:
`grad_W_lm = final_rms_output.T @ grad_logits`

Elements: **B × T × D**

---

##### LM head / output embedding — output shape: **(B, T, V)**

```
operation:  logits = final_rms_output @ W_lm.T
W_lm shape: (V, D)   — maps D dims to V vocab scores
output:     (B, T, V)
```

Why `(B, T, V)`?
- B: one output per sequence in the batch
- T: at each of the T token positions, the model predicts the next token
- V = 50,257: the model outputs one raw score (logit) for every possible next token in the vocabulary

This is by far the widest tensor per token: V = 50,257 vs D = 1,600.

Why saved? Cross-entropy backward needs the logits to compute the softmax probabilities and then `grad_logits`.

Elements: **B × T × V**

---

##### Cross-entropy — no new large tensor

```
input:  logits (B, T, V)  — already saved above
        targets (B, T)    — token IDs, tiny
output: scalar loss
```

The backward of cross-entropy only needs the logits (already in memory) and the targets (tiny). No new large tensor is allocated.

---

#### Global total: BTD + BTV elements

---

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
