# Technical Note: DeepSeek-V4.1-Flash 40-Layer Pallas Decode Megakernel & `DSpark` (`mtp.0..2`) on `TPU v6e-16`

This technical note documents the exact decode equations, checkpoint weight layouts, `TPU v6e-16` (`4×4` 2D mesh) hardware measurements, lowered HLO structure, and `DSpark` (`mtp.0..2`) speculative verification invariants implemented in [`megakernel/`](megakernel/).

---

## 1. Checkpoint Verification & Weight Inventory (`gs://dsv4-flash-jawadamin-asia-ne1/deepseek-v4.1-flash`)

The checkpoint consists of `48` `safetensors` shards (`475 GiB` total: shards `00001..00046` hold the `40` decoder layers, embedding, final norm, LM head, and `3` `mtp.0..2` layers; shards `00047` and `00048` hold the `L1` and `L14` Engram tables at `101.5 GB` each).

On every load ([`megakernel/load.py`](megakernel/load.py)), six sampled tensors across the checkpoint are verified by SHA-256 against [`results/ckpt_hashes.json`](results/ckpt_hashes.json):

| Tensor Name | Safetensors Shard | Shape | Dtype | Verified SHA-256 |
|---|---|---|---|---|
| `embed.weight` | `model-00002-of-00048.safetensors` | `[129280, 5120]` | `BF16` | `3b14dd76fa6c0c0aa2a528404ec5bfc471bf93c41b961a08e1955c4c712716c9` (`head4M_tail1M`) |
| `layers.0.attn.wq_a.weight` | `model-00003-of-00048.safetensors` | `[1280, 5120]` | `F8_E4M3` | `d75dd17b5fd33533ab8269412e23c01a44314fd719b8e0d274cd437b3f125af7` |
| `layers.0.ffn.experts.0.w1.weight` | `model-00003-of-00048.safetensors` | `[2304, 2560]` | `I8` (packed `MXFP4`) | `84c55fa21fcbf91d97372c49ab1556220d50fd613219b9fe975da493e071d9a4` |
| `layers.14.attn.compressor.wkv.weight` | `model-00017-of-00048.safetensors` | `[512, 5120]` | `BF16` | `42318c7477d6948b6ee6b9f9fa4c2b0377344fbcf11a9f3832dc26d3f5ab5888` |
| `layers.39.ffn.gate.weight` | `model-00042-of-00048.safetensors` | `[384, 5120]` | `BF16` | `8ff7c88eb8478365a88697edb79f436c17db765d753c7a1e7959c365f06e46a9` |
| `norm.weight` | `model-00043-of-00048.safetensors` | `[5120]` | `BF16` | `9cd3b57cd9513541b9771bf66c9b356bf1a7b20ff050ed69f7e97cff9fedd428` |

### Quantization Formats Handled
1. **Dense & Shared-Expert `FP8` (`32 × 32` Block-Scaled `E4M3` + `UE8M0`):**
   Weight tensors (`wq_a`, `wq_b`, `wo_a`, `wo_b`, `shared_experts.w1/w2/w3`) are stored in `float8_e4m3fn` with a `uint8` `UE8M0` exponent scale per `32 × 32` tile (`scale = 2^(u8 - 127)`).
2. **Routed-Expert `MXFP4` (`E2M1` Nibbles + `UE8M0` Block-32 Scales):**
   Each of the `384` routed experts per backbone layer (`128` per `DSpark` layer) stores two `E2M1` 4-bit codes per `int8`/`uint8` byte (`low nibble = even column`, `high nibble = odd column`) with one `UE8M0` exponent byte per `1 × 32` input block (`K = 5120` → `160` scale blocks for `w1/w3`; `K = 2304` → `72` scale blocks for `w2`).

---

## 2. Exact Per-Layer Decode Dataflow

### 2.1 Manifold-Constrained Hyper-Connections (`mHC`, `hc_mult = 4`, `20` Sinkhorn Iterations)
The hidden state is carried across all 40 layers as 4 parallel streams $S \in \mathbb{R}^{B \times 4 \times 5120}$. Before each sublayer (`attn` or `ffn`), [`decode_megakernel.py`](megakernel/decode_megakernel.py) normalizes the flattened stream vector $S_{\text{flat}} \in \mathbb{R}^{B \times 20480}$ (`eps = 1e-20`), projects dynamic adjustments via `hc_proj` (`[20480, 24]`) scaled by `hc_scale` (`float32`), adds `hc_base` (`[6, 4]`), and splits the result into:
- $\alpha_{\text{pre}} = \sigma(H_0) \in \mathbb{R}^{B \times 4}$ (`sigmoid`)
- $\alpha_{\text{post}} = 2\sigma(H_1) \in \mathbb{R}^{B \times 4}$ (`2 * sigmoid`)
- $M_0 = H_{2:6} \in \mathbb{R}^{B \times 4 \times 4}$, refined via **20 Sinkhorn-Knopp iterations** in `float32`:
  $$A^{(0)} = \exp(M_0 - \max(M_0)), \qquad A^{(t+1)} = \text{ColNorm}\!\left(\text{RowNorm}\!\left(A^{(t)}\right)\right)$$
The sublayer input is $x_{\text{in}} = \text{RMSNorm}\!\left(\sum_{s=0}^{3} \alpha_{\text{pre}, s} S_s\right)$, and the post-sublayer stream update is:
$$S \leftarrow M_{20} S + \alpha_{\text{post}}^\top \odot y_{\text{sublayer}}$$

### 2.2 `Engram` Memory Lookup at Layers `1` and `14`
Before attention at layers `1` and `14`, the last 4 causal token IDs form 2-gram and 3-gram windows that hash into 8 prime-sized sub-tables (`total rows = 384,006,168`, `dim = 256` per head, `16` heads = `4096` total embedding dim). After gathering and dequantizing the FP8 rows on host (`engram_host`), the kernel projects Keys and Values per stream, applies RMSNorm + SiLU-gated dot-product scaling, runs a causal depthwise `conv1d` (`kernel_size = 4`) with RMSNorm + SiLU residual, and adds the result directly into the 4 `mHC` streams $S$.

> [!NOTE]
> **In-Kernel vs. Host-DRAM `Engram` Execution Scope (`enable_engram`):** Inside `_megakernel_40l_body`, the Layer 1 and Layer 14 `Engram` projections, RMSNorm, SiLU gate, causal `conv1d`, and 4-stream `mHC` residual update **execute unconditionally on every step**. In Stage 1 ([`verify_reference_engine.py`](scripts/verify_reference_engine.py)) and Stage 2 Checks 1 & 2 ([`verify_pallas_megakernel.py`](scripts/verify_pallas_megakernel.py)), the full `203.1 GB` (`shards 47–48`) host-DRAM `mmap` tables are loaded (`load_engram_tables=True`, `enable_engram=True`), verifying both end-to-end golden parity (`98.24%` / `97.36%` tie-aware top-1), the `engram_table_row_L1` negative control (`max_abs_logit_diff = 36.07`), and `~10.37 ms/step` end-to-end step time including synchronous Python host-CPU `mmap` gather (`82` steps in `0.85 s` on `p6_lin_alg`). In Stage 2 Check 4, Stage 3 ([`verify_accuracy_and_tpot.py`](scripts/verify_accuracy_and_tpot.py)), and Stage 4 ([`verify_dspark_speculative.py`](scripts/verify_dspark_speculative.py)), `enable_engram=False` (`zero_eng_rows` passed into the active Layer 1 & Layer 14 in-kernel Engram blocks) is used to avoid holding `203.1 GB` of host `mmap` buffers alongside multi-bucket Pallas + `DSpark` (`mtp.0..2`) compilation and to isolate on-device TPU step time (`8.49 ms` at `C=1`, `5.57 ms` with `DSpark`, and `12/16 = 75.0%` GSM8K/STEM accuracy matching `vLLM`'s `12/16 = 75.0%` with host `Engram` enabled).

### 2.3 Multi-Variant `CSA + SWA` Decode Attention
- **Query & Local SWA KV:** $q_a = \text{RMSNorm}(x_{\text{in}} W_{q,a})$, $q = q_a W_{q,b} \in \mathbb{R}^{B \times 64 \times 512}$ (`448` nope + `64` Yarn RoPE). The local KV projection $kv_{\text{local}} = x_{\text{in}} W_{kv} \in \mathbb{R}^{B \times 512}$ (`448` nope + `64` Yarn RoPE) writes into the 128-slot SWA ring buffer.
- **KV Compressor (`L2, L8, L14` at `ratio = 2`; `L20` at `ratio = 1`):**
  - At `L2, L8, L14`, every 2 consecutive tokens `[t_even, t_odd]` in `comp_buf` (`[B, 2, 5120]`) are concatenated (`10,240` dim) and projected through `compressor.wkv_a` → RMSNorm → `compressor.wkv_b` (`[512]`), with `compress_rope_theta` applied to the 64 RoPE dims at position $p_{\text{comp}} = \lfloor t / 2 \rfloor$.
  - Consumer layers share compressed KV from `kv_source_layer_ids = [2, 8, 14, 20]` (`L0..L1` are SWA-only).
- **Two-Level Sparse Indexer (`L2, 8, 14, 20, 24, 28, 32, 36`):**
  Computes 32-head `128`-dim indexer queries (`wq_b`) and keys (`wk`), scores compressed KV entries with head weights (`weights_proj` / $\sqrt{128}$), and selects up to `top-512` compressed KV entries (with `L20` also producing `candidate_topk_blocks` for `L24, 28, 32, 36`).
- **Joint Softmax & Grouped Output Projection:**
  Compressed KV logits and SWA logits are combined under a single numerically stable online softmax (`softmax_scale = 512^-0.5 * mscale^2`), followed by 8-group output LoRA (`wo_a` `[8, 8, 512, 128]` → `wo_b` `[8, 128, 5120]`).

### 2.4 Top-6 of 384 `MXFP4` Routed MoE + Shared Expert
The router computes $g = \text{sqrtsoftplus}(x_{\text{ffn}} W_{\text{gate}})$, selects the top-6 experts using $g + b_{\text{score\_bias}}$ (`noaux_tc`), normalizes the top-6 raw scores $g$, and multiplies by `routed_scaling_factor = 1.5`. Both routed and shared experts apply SwiGLU with clamping (`swiglu_limit = 10.0` on both gate and up activations).

---

## 3. Single-`pallas_call` HLO & Per-Layer Parity Verification

From [`results/pallas_megakernel_report.json`](results/pallas_megakernel_report.json):
- **Lowered HLO Inspection (`check3_single_pallas_call_hlo`):**
  - `tpu_custom_call_count = 1` (single persistent 40-layer Pallas kernel)
  - `all_gather_count = 1` (single post-kernel 16-way `lax.all_gather` collecting the `[8080]` per-chip LM-head vocab shard into `[129280]` logits inside `jax.shard_map`)
  - `hlo_bytes = 6,406,704` (`6.2 MiB`)
- **All-40-Layer Parity vs. Pure-JAX Reference Engine (`check1_per_layer_parity`, `enable_engram=True`):**
  - **`B = 1`:** `40 / 40` layers pass (`min_cos_b1 = 0.999984`, `max_abs_diff_b1 = 0.0625`).
  - **`B = 8`:** `40 / 40` layers pass (`min_cos_b8 = 0.999647`, `max_abs_diff_b8 = 0.1250`).
- **End-to-End Golden Parity (`check2_end_to_end_golden`, `enable_engram=True`):**
  - `tie_aware_prompt_top1_agreement = 97.36%` (`strict = 92.96%` vs. `94.13%` `vLLM` `C=1` vs. `C=8` batch-composition floor).
  - `8 / 8` greedy continuations match the pure-JAX reference engine token-for-token.

---

## 4. `DSpark` (`mtp.0..2`) Group-Packed Architecture & Speculative State Machine

DeepSeek-V4.1-Flash's three Multi-Token Prediction layers (`mtp.0`, `mtp.1`, `mtp.2`) operate in parallel inside a single `DSpark` draft block (`dspark_block_size = 5`, producing `4` draft tokens per step):
1. **Target Feature Concatenation (`L37, L38, L39`):**
   During the megakernel pass, each token captures `RMSNorm(S.mean(axis=1))` after layers `37`, `38`, and `39`, concatenated into $h_{\text{target}} \in \mathbb{R}^{15360}$.
2. **8-Head Group-Packed `main_proj` (`5120 → 1280`):**
   Given the anchor token embedding $\text{RMSNorm}(E(x_0))$ and $h_{\text{target}}$, each `mtp.k` layer computes:
   $$z_k = \text{concat}(\text{RMSNorm}(E(x_k)), h_{\text{target}}) W_{\text{eh}} \in \mathbb{R}^{5120}$$
   In `mtp.0..2`, `attn.main_proj` (`[1280, 5120]`) replaces both `wq_a` (`1280`) and `wkv` (`512`). Crucially, the checkpoint packs `wq_b` (`[36864, 1280]`) into **8 groups (`g = c // 2` for `c = 0..15`) of `160` input channels (`128` nope + `32` rope)**, where each group of 8 query heads (`4608` output dims = `2 × (4 × 512 Q) + 512 K`) expects:
   - `q_nope_even` (`4 × 448 = 1792`) + `q_rope_even` (`4 × 64 = 256` with RoPE)
   - `q_nope_odd` (`4 × 448 = 1792`) + `q_rope_odd` (`4 × 64 = 256` with RoPE)
   - `k_nope` (`448`) + `k_rope` (`64` with RoPE)
   Summing the 8 groups' `k_nope` and `k_rope` yields the shared MQA key/value `kv_t` (`512` dim) written to `dspark_kv_cache`, while `mtp.1` and `mtp.2` add a rank-256 Markov transition term (`markov_q_proj` × `markov_k_proj`) before the `o_groups = 8` output projection and top-3 of 128 `MXFP4` MoE block.
3. **Lossless Causal Verification & Zero-Copy Pointer Rollback:**
    Given anchor token $t_0$ and 4 draft tokens $[\hat{t}_1, \hat{t}_2, \hat{t}_3, \hat{t}_4]$, [`verify_speculative_step`](megakernel/decode_megakernel.py) runs all 5 tokens through the 40-layer Pallas megakernel in causal order, producing target greedy predictions $[t_1^*, t_2^*, t_3^*, t_4^*, t_5^*]$. If the first mismatch occurs at $j \in \{1..4\}$ ($\hat{t}_j \neq t_j^*$), the step emits $[t_1^*, \dots, t_j^*]$ (`j` tokens) and rewinds `swa_pos`, `comp_buf_len`, and `comp_num_entries` to their state after token $j-1$ (`O(1)` scalar rollback). If all 4 draft tokens match, all 5 tokens $[t_1^*, t_2^*, t_3^*, t_4^*, t_5^*]$ are emitted.
