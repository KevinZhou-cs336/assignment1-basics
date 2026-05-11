# Transformer Accounting (Problem: transformer_accounting)

## Model Configuration (GPT-2 XL–sized)

| Hyperparameter | Full meaning | Value | Symbol used below |
|----------------|--------------|-------|-------------------|
| vocab_size | Total number of distinct tokens the model can recognize | 50,257 | V |
| context_length | Number of tokens processed in one forward pass (sequence length) | 1,024 | T (for "Time steps") |
| num_layers | Number of stacked Transformer blocks (depth) | 48 | L |
| d_model | Hidden dimension: the size of the vector representing each token | 1,600 | D |
| num_heads | Number of parallel attention heads in Multi-Head Attention | 25 | H |
| d_head = D / H | Dimension assigned to each attention head (D split evenly across H heads) | 64 | d_head |
| d_ff | FFN intermediate dimension: input is expanded from D to F then compressed back | 4,288 | F |

> **Symbol quick-reference:**
> - **V** = vocab size (number of tokens in vocabulary)
> - **T** = sequence / context length (number of tokens per forward pass)
> - **L** = number of Transformer layers
> - **D** = d_model (hidden / embedding dimension per token)
> - **H** = number of attention heads
> - **d_head** = D/H (dimension per attention head)
> - **F** = d_ff (feed-forward network intermediate dimension)
> - **FLOPs** = Floating Point Operations (measure of compute cost)
> - **TFLOPs** = 10¹² FLOPs (tera floating point operations)

---

## (a) Trainable Parameters and Memory

### Parameter count by component

| Component | Description | Shape | Count |
|-----------|-------------|-------|-------|
| `token_embeddings.weight` | Embedding table: maps each integer token ID to a D-dim vector | (V=50257, D=1600) | 80,411,200 |
| Per block ×48: `ln1.weight` | Learned per-dimension scale for RMSNorm before attention | (D=1600,) | 1,600 |
| Per block ×48: `attn.q_proj.weight` | Query projection W_Q: projects input into query vectors | (D=1600, D=1600) | 2,560,000 |
| Per block ×48: `attn.k_proj.weight` | Key projection W_K: projects input into key vectors | (D=1600, D=1600) | 2,560,000 |
| Per block ×48: `attn.v_proj.weight` | Value projection W_V: projects input into value vectors | (D=1600, D=1600) | 2,560,000 |
| Per block ×48: `attn.output_proj.weight` | Output projection W_O: merges all heads back to D dims | (D=1600, D=1600) | 2,560,000 |
| Per block ×48: `ln2.weight` | Learned per-dimension scale for RMSNorm before FFN | (D=1600,) | 1,600 |
| Per block ×48: `ffn.w1.weight` | SwiGLU gate up-projection (gate stream: D → F) | (F=4288, D=1600) | 6,860,800 |
| Per block ×48: `ffn.w2.weight` | SwiGLU down-projection (F → D, back to d_model) | (D=1600, F=4288) | 6,860,800 |
| Per block ×48: `ffn.w3.weight` | SwiGLU value up-projection (value stream: D → F) | (F=4288, D=1600) | 6,860,800 |
| `ln_final.weight` | Final RMSNorm before the language model head | (D=1600,) | 1,600 |
| `lm_head.weight` | Language model head: projects D-dim hidden state to V-dim logits | (V=50257, D=1600) | 80,411,200 |

**Per-block subtotal:**
- 2 RMSNorm scales: 2 × D = 2 × 1,600 = 3,200
- 4 attention weight matrices (Q/K/V/O): 4 × D² = 4 × 2,560,000 = 10,240,000
- 3 FFN weight matrices (W1/W2/W3): 3 × D×F = 3 × 6,860,800 = 20,582,400
- **Subtotal per block: 30,825,600**

**Total:**
```
token_embeddings + lm_head  +  L × per_block  +  ln_final
= 2 × (V × D)  +  L × (2D + 4D² + 3DF)  +  D
= 2 × 80,411,200  +  48 × 30,825,600  +  1,600
= 160,822,400  +  1,479,628,800  +  1,600
= 1,640,452,800
```

**Total trainable parameters: ~1.64 billion**

### Memory (single-precision float32, 4 bytes/param)

```
1,640,452,800 × 4 bytes = 6,561,811,200 bytes ≈ 6.11 GB
```

**Deliverable:** The GPT-2 XL–sized model has approximately 1.64 billion trainable parameters. Storing them in single-precision float32 (4 bytes each) requires about 6.1 GB of memory.

---

## (b) FLOPs for One Forward Pass

**Counting rule:** For a matrix multiply A ∈ ℝ^{m×n} · B ∈ ℝ^{n×p}, the result has m×p entries, each requiring n multiplications and n additions (= 2n ops), so the total cost is **2mnp FLOPs**.

Only matrix multiplies are counted — elementwise operations (RMSNorm, SiLU, softmax, residual add) are negligible in comparison.
Token embedding lookup is an index/gather operation with 0 FLOPs.

> **Dimension reminder:** Each Transformer block receives input of shape **(T, D) = (1024, 1600)**,
> meaning T=1024 tokens, each represented by a D=1600-dimensional vector.

### Per transformer block (repeated × L=48)

#### Multi-Head Self-Attention (MHA)

Each block has 4 weight matrices (W_Q, W_K, W_V, W_O), each of shape (D, D),
plus the attention score computation QKᵀ and the weighted sum Attn·V across all H heads.

| Operation | What it does | Dimensions (m × n × p) | FLOPs = 2mnp |
|-----------|--------------|------------------------|--------------|
| Q projection: x · W_Qᵀ | Map each of T tokens (dim D) to a query vector (dim D) | T × D × D = 1024 × 1600 × 1600 | 5,242,880,000 |
| K projection: x · W_Kᵀ | Map each token to a key vector | T × D × D = 1024 × 1600 × 1600 | 5,242,880,000 |
| V projection: x · W_Vᵀ | Map each token to a value vector | T × D × D = 1024 × 1600 × 1600 | 5,242,880,000 |
| QKᵀ (all H heads) | Compute pairwise attention scores; summed over H heads = T²×D total | H×(T × d_head × T) = 1024 × 1600 × 1024 | 3,355,443,200 |
| Attn · V (all H heads) | Weighted sum of value vectors; summed over H heads = T²×D total | H×(T × T × d_head) = 1024 × 1024 × 1600 | 3,355,443,200 |
| Output projection: x · W_Oᵀ | Merge all heads back to D-dim output | T × D × D = 1024 × 1600 × 1600 | 5,242,880,000 |
| **MHA subtotal** | | | **27,682,406,400** |

#### SwiGLU Feed-Forward Network (FFN)

Each FFN has 3 weight matrices: W1 and W3 up-project D→F, W2 down-projects F→D.
FFN(x) = W2 · (SiLU(W1·x) ⊙ W3·x)

| Operation | What it does | Dimensions (m × n × p) | FLOPs = 2mnp |
|-----------|--------------|------------------------|--------------|
| W1 gate up-projection: x · W1ᵀ | Expand each token from D to F dims (gate stream) | T × D × F = 1024 × 1600 × 4288 | 14,073,241,600 |
| W3 value up-projection: x · W3ᵀ | Expand each token from D to F dims (value stream) | T × D × F = 1024 × 1600 × 4288 | 14,073,241,600 |
| W2 down-projection: (SiLU⊙) · W2ᵀ | Compress gated result from F back to D dims | T × F × D = 1024 × 4288 × 1600 | 14,073,241,600 |
| **FFN subtotal** | | | **42,219,724,800** |

**Per-block total: 27,682,406,400 + 42,219,724,800 = 69,902,131,200 FLOPs**

### Across all L=48 blocks

```
48 × 69,902,131,200 = 3,355,302,297,600 FLOPs
```

### LM head (final unembedding — runs once, not ×48)

The LM head projects the final hidden state of every token from D dims to V-dim logits (one score per vocabulary entry).

| Operation | What it does | Dimensions (m × n × p) | FLOPs = 2mnp |
|-----------|--------------|------------------------|--------------|
| lm_head: hidden → logits | Project T token vectors from D to V dims | T × D × V = 1024 × 1600 × 50,257 | 164,682,137,600 |

### Grand total

```
L × (MHA + FFN)  +  LM head
= 3,355,302,297,600 + 164,682,137,600
= 3,519,984,435,200 FLOPs
≈ 3.52 × 10¹² FLOPs  (3.52 TFLOPs)
```

### Summary table

| Component | FLOPs | % of total |
|-----------|-------|------------|
| 48 × MHA (Q/K/V/O projections + QKᵀ + Attn·V) | 1,328,755,507,200 | 37.7% |
| 48 × SwiGLU FFN (W1, W2, W3) | 2,026,546,790,400 | 57.6% |
| LM head (×1, not repeated per layer) | 164,682,137,600 | 4.7% |
| **Total** | **3,519,984,435,200** | **100%** |

**Deliverable:** The forward pass requires 9 families of matrix multiplies per Transformer block (3 Q/K/V projections + output projection + QKᵀ + Attn·V for MHA, plus W1/W2/W3 for SwiGLU FFN) and one LM head projection, totaling approximately **3.52 TFLOPs**. The SwiGLU FFN layers account for ~58%, MHA for ~38%, and the LM head for ~5%.

---

## (c) Which Component Requires the Most FLOPs?

The 48 SwiGLU FFN layers dominate, consuming ~2.03 TFLOPs (58% of total). Each block runs three matrix multiplies scaled by the expanded intermediate dimension F=4,288 ≈ 2.68×D, making each FFN ~1.5× more expensive than the full MHA in the same block. The MHA layers are second at ~1.33 TFLOPs (38%). The LM head is a distant third at ~165 GFLOPs (5%): although it involves the large vocabulary dimension V=50,257, it runs only once — not 48 times like the block operations.

**Deliverable:** The SwiGLU FFN sublayers require the most FLOPs (~58% of total), because each of the 48 blocks performs three matrix multiplies expanded to the d_ff=4,288 intermediate dimension; the LM head contributes only ~5% despite the large vocabulary size, because it is not repeated per layer.
