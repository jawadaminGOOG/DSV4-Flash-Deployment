# Technical Note: DeepSeek-V4.1-Flash 40-Layer Pallas Decode Megakernel, Vectorized `Engram`, & `DSpark` (`mtp.0..2`) on `TPU v6e-16`

This technical note documents the exact decode equations, checkpoint weight layouts, `TPU v6e-16` (`4×4` 2D mesh) hardware measurements, lowered HLO structure, vectorized `/dev/shm` `Engram` host-to-VMEM pipeline, and `DSpark` (`mtp.0..2`) speculative verification invariants implemented in [`megakernel/`](megakernel/).

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
   Weight tensors (`wq_a`, `wq_b`, `wo_a`, `wo_b`, `shared_experts.w1/w2/w3`, `engram.wkv`) are stored in `float8_e4m3fn` with a `uint8` `UE8M0` exponent scale per `32 × 32` tile (`scale = 2^(u8 - 127)`).
2. **Routed-Expert `MXFP4` (`E2M1` Nibbles + `UE8M0` Block-32 Scales):**
   Each of the `384` routed experts per backbone layer (`128` per `DSpark` layer) stores two `E2M1` 4-bit codes per `int8`/`uint8` byte (`low nibble = even column`, `high nibble = odd column`) with one `UE8M0` exponent byte per `1 × 32` input block (`K = 5120` → `160` scale blocks for `w1/w3`; `K = 2304` → `72` scale blocks for `w2`).

---

## 2. Exact Per-Layer Decode Dataflow

### 2.1 Manifold-Constrained Hyper-Connections (`mHC`, `hc_mult = 4`, `20` Sinkhorn Iterations)
The hidden state is carried across all 40 layers in VMEM as 4 parallel streams $S \in \mathbb{R}^{4 \times B_{\text{tile}} \times 5120}$ along with the delayed pre-mix vector $\alpha_{\text{pre}} \in \mathbb{R}^{4 \times B_{\text{tile}} \times 128}$. Before each sublayer (`attn` or `ffn`), [`decode_megakernel.py`](megakernel/decode_megakernel.py) normalizes the flattened stream vector $S_{\text{flat}} \in \mathbb{R}^{B_{\text{tile}} \times 20480}$ (`eps = 1e-20`), projects dynamic adjustments via `hc_fn` (`[5120, 128]` per stream) scaled by `hc_scale` (`float32`), adds `hc_base`, and computes:
- $\alpha_{\text{pre, next}} = \sigma(H_0) + 10^{-6} \in \mathbb{R}^{B_{\text{tile}} \times 4}$ (delayed pre-mix consumed by the next sublayer)
- $\alpha_{\text{post}} = 2\sigma(H_1) \in \mathbb{R}^{B_{\text{tile}} \times 4}$
- $M_0 = \text{softmax}(H_{2:6}) + 10^{-6} \in \mathbb{R}^{B_{\text{tile}} \times 4 \times 4}$, refined via **20 Sinkhorn-Knopp column/row normalization iterations** in `float32`.
The sublayer input is $x_{\text{in}} = \text{RMSNorm}\!\left(\sum_{s=0}^{3} \alpha_{\text{pre}, s} S_s\right)$, and the post-sublayer stream update is:
$$S \leftarrow M_{20} S + \alpha_{\text{post}}^\top \odot y_{\text{sublayer}}$$

### 2.2 Vectorized `200.5 GB` `Engram` Lookup & In-Kernel Gated Cross-Attention (`Layers 1 & 14`)
At layers `1` and `14`, the last 4 causal token IDs $[w_0, w_1, w_2, w_3]$ (`[t, t-1, t-2, t-3]`) hash into `24` prime-sized sub-tables (`8` 2-gram heads + `8` 3-gram heads + `8` 4-gram heads, `384,006,168` total rows × `256` `FP8 E4M3` cols + `12,000,193 × 256` `UE8M0` scales per layer = `200.5 GB` total across shards `47–48`):
1. **Vectorized Host `/dev/shm` Gather (`160.8 µs` for both `L1` and `L14`):**
   Each of the 4 hosts maps its 6 local `Engram` columns (`cols [6*h .. 6*h + 5]`, `~47.2 GiB/host`) in `/dev/shm`. On every step, [`EngramHostTables.fast_gather_both_chips`](megakernel/load.py) evaluates the exact uint64 XOR-multiply rolling hash (`prime * (prev ^ tok)`) for the host's 6 local columns, slices the `/dev/shm` memory maps,dequantizes `FP8 E4M3` via a 256-entry `float32` lookup table (`_fp8_lut[u8] * exp2(scale_u8 - 127)`), and packs the `[2, B_tile, 384]` `bfloat16` shards onto the 4 local TPU chips in **`160.8 µs` (`B=1`) to `185.5 µs` (`B=5`)**.
2. **In-Kernel `pallas_engram_sublayer` (`decode_megakernel.py:570-645`):**
   Inside `_megakernel_40l_body`, `pallas_engram_sublayer` executes on every step at `L1` (`eng_idx = 0`) and `L14` (`eng_idx = 1`):
   - Collects the 4 host groups' `[B_tile, 1536]` slices across the `4×4` mesh into the full `[B_tile, 6144]` `Engram` embedding in VMEM.
   - Applies the block-32 `FP8` linear projection `engram.wkv` (`[25600, 6144]`, sharded across `TP=16` as `[5, 384, 5120]` per chip and summed across each 4-chip row) to produce `4` Key streams $K \in \mathbb{R}^{4 \times B_{\text{tile}} \times 5120}$ and `1` shared Value stream $V \in \mathbb{R}^{B_{\text{tile}} \times 5120}$.
   - Computes the RMS-normalized per-stream dot product with `engram.q_weight` and `engram.k_weight` (`[4, 5120]`):
     $$d_s = \frac{\langle S_s \odot w_{q,s},\; K_s \odot w_{k,s} \rangle}{\text{RMS}(S_s)\,\text{RMS}(K_s)\,\sqrt{5120}}, \qquad g_s = \sigma\!\left(\text{sign}(d_s)\sqrt{\max(|d_s|, 10^{-6})}\right)$$
   - Updates all 4 `mHC` streams in VMEM before attention: $S_s \leftarrow S_s + g_s \odot V$.

### 2.3 Multi-Variant `CSA + SWA` Decode Attention
- **Query & Local SWA KV:** $q_a = \text{RMSNorm}(x_{\text{in}} W_{q,a})$, $q = q_a W_{q,b} \in \mathbb{R}^{B_{\text{tile}} \times 64 \times 512}$ (`448` nope + `64` RoPE). The local KV projection $kv_{\text{local}} = \text{RMSNorm}_{\text{nope}}(x_{\text{in}} W_{kv}) \in \mathbb{R}^{B_{\text{tile}} \times 512}$ writes into the 128-slot SWA ring buffer.
- **KV Compressor (`L2, L8, L14` at `ratio = 2`; `L20` at `ratio = 1`):**
   - At `L2, L8, L14`, every 2 consecutive tokens `[t_even, t_odd]` in `comp_tail` (`[B_{\text{tile}}, 1024]`) are weighted via `compressor.wgate`, projected through `compressor.wkv`, RMS-normalized (`compressor.norm`), and rotated by YaRN RoPE (`compress_rope_theta = 160000.0`) at compressed slot $p_{\text{comp}} = \lfloor t / 2 \rfloor$.
   - Consumer layers share compressed KV from `kv_source_layer_ids = [2, 8, 14, 20]` (`L0..L1` are SWA-only).
- **Two-Level Sparse Indexer (`L2, 8, 14, 20, 24, 28, 32, 36`):**
   Computes 32-head `128`-dim indexer queries (`idx_wqb`) and keys (`idx_wk`), scores compressed KV entries with head weights (`idx_wproj` / $\sqrt{128}$), and selects up to `top-512` compressed KV entries.
- **Combined Attention & Grouped Output Projection:**
   Compressed KV logits and 128-window SWA logits (plus the learned `attn_sink` logit per head) are combined under a single softmax (`softmax_scale = 512^-0.5 * mscale^2`), followed by 8-group output LoRA (`wo_a` → `wo_b`) and 2D mesh all-reduce (`pallas_allreduce_16`).

### 2.4 Top-6 of 384 `MXFP4` Routed MoE + Shared Expert
Inside `_megakernel_40l_body`, the router computes $g = \text{sqrtsoftplus}(x_{\text{ffn}} W_{\text{gate}})$, selects the top-6 of 384 experts using $g + b_{\text{gate}}$ (`noaux_tc`) via 6 argmax passes in VMEM, normalizes the top-6 scores, multiplies by `routed_scaling_factor = 1.5`, and builds the sparse routing weight matrix `p_w` (`[B_tile, 24]`) for the `24` experts (`384 / 16`) owned by `chip_id`. Each chip streams its `24` local `MXFP4` experts in 3 groups of 8 through VMEM (`_mxfp4_group8_T` + `swiglu_limit = 10.0` clamped SwiGLU), adds the TP-sharded `FP8` shared expert output, and reduces across all 16 chips via `pallas_allreduce_16`.

---

## 3. Single-`pallas_call` HLO, Per-Layer Parity, & Live `Engram` Performance

From [`results/pallas_megakernel_report.json`](results/pallas_megakernel_report.json) and [`results/accuracy_and_tpot_report.json`](results/accuracy_and_tpot_report.json):
- **Lowered HLO Inspection (`check3_single_pallas_call_hlo`):**
  - `tpu_custom_call_count = 1` (single persistent 40-layer Pallas kernel)
  - `all_gather_count = 1` (single post-kernel 16-way `lax.all_gather` collecting the `[8080]` per-chip LM-head vocab shard into `[129280]` logits inside `jax.shard_map`)
  - `hlo_bytes = 6,406,704` (`6.2 MiB`)
- **All-40-Layer Parity vs. Pure-JAX Reference Engine (`check1_per_layer_parity`, `enable_engram=True`):**
  - **`B = 1`:** `40 / 40` layers pass (`min_cos_b1 = 0.999984`, `max_abs_diff_b1 = 1.0` at Layer 39 matching the `2.0` two-XLA-config BF16 floor; Layer 1 `Engram` `cos = 0.999999`, Layer 14 `Engram` `cos = 1.000000`).
  - **`B = 8`:** `40 / 40` layers pass (`min_cos_b8 = 0.999647`, `max_abs_diff_b8 = 2.0` at Layer 39 matching the `2.0` two-XLA-config BF16 floor; Layer 1 `Engram` `cos = 0.999992`, Layer 14 `Engram` `cos = 0.999998`).
- **Live `enable_engram=True` Step Latency (`check4_decode_step_latency` & Stage 3 Concurrency Sweep):**
  - **`1,024` Context:** `B = 1` median **`9.16 ms`** (`9.30 ms` / `107.6 tok/s` in the Stage 3 concurrency ladder, `4.82×` faster than `vLLM`'s `44.78 ms`); `B = 8` median **`16.83 ms`** (`20.69 ms` / `386.6 tok/s` with distinct request positions in Stage 3).
  - **`4,096` Context:** `B = 1` median **`10.25 ms`**; `B = 8` median **`17.79 ms`**.
  - **16-Question GSM8K/STEM Accuracy (`enable_engram=True`):** **`16 / 16` (`100.0%`)** vs. `12 / 16` (`75.0%`) on `vLLM` raw completion.

---

## 4. `DSpark` (`mtp.0..2`) Architecture & Lossless Speculative State Machine (`enable_engram=True`)

DeepSeek-V4.1-Flash's three Multi-Token Prediction layers (`mtp.0`, `mtp.1`, `mtp.2`) operate inside a single `DSpark` draft block (`dspark_block_size = 5`, `dspark_noise_token_id = 128799`, proposing `4` draft tokens per step):
1. **Target Feature Fusion (`L37, L38, L39`):**
   During the 40-layer Pallas megakernel pass, each verified row captures the 4-stream mean across target layers `dspark_target_layer_ids = [37, 38, 39]`, concatenated into $h_{\text{target}} \in \mathbb{R}^{15360}$. In `mtp.0`, `main_proj` (`[5120, 15360]` FP8) + `main_norm` (`[5120]` BF16) projects $h_{\text{target}}$ and fuses it with the draft block embeddings (`[root_tok, 128799, 128799, 128799, 128799]`) via `eh_proj` (`[5120, 10240]` FP8).
2. **3-Stage `DSparkBlock` (`mtp.0..2`) + Low-Rank Markov Head (`dspark_markov_rank = 256`):**
   The 5-token draft block runs through `mtp.0`, `mtp.1`, and `mtp.2` (`compress_ratio = 0`, bidirectional intra-block + 128-window causal SWA attention using KV projected from $h_{\text{target}}$ via `mtp.{0,1,2}.attn.wkv_b` `[512, 15360]`, and top-3 of 128 `MXFP4` routed MoE + 1 shared expert). After `mtp.2.shared_head.norm`, sequential draft tokens $\hat{t}_1, \dots, \hat{t}_4$ are selected greedily by combining the shared LM head logits (`129,280` vocab) with the low-rank Markov transition head (`mtp.2.markov_head.embed` `[129280, 256]` + `mtp.2.markov_head.head` `[129280, 256]` conditioned on the previously drafted token $\hat{t}_{k-1}$).
3. **Lossless Causal Verification with Live `Engram` (`2,048 / 2,048` Exact Token Identity):**
   Given anchor token $t_0$ and 4 draft tokens $[\hat{t}_1, \hat{t}_2, \hat{t}_3, \hat{t}_4]$, [`verify_speculative_step`](megakernel/decode_megakernel.py) gathers the 5 causal 4-gram `Engram` windows (`wins [5, 4]`) via `fast_gather_both_chips` (`185.5 µs`) and runs all 5 candidate tokens through the 40-layer Pallas megakernel in a single pass (`spec_k = 5`), producing target greedy predictions $[t_1^*, t_2^*, t_3^*, t_4^*, t_5^*]$. If the first mismatch occurs at $j \in \{1..4\}$ ($\hat{t}_j \neq t_j^*$), the step emits $[t_1^*, \dots, t_j^*]$ (`n_keep = 1 + n_acc`) and `_commit_spec_state` rolls back the rejected SWA and compressor tail slots in `O(1)` device time. Across all 8 golden prompts (`2,048 / 2,048` tokens = **`100.0%` lossless token match**), `DSpark` achieves **`3.15 tok/step` mean acceptance (`4.92 tok/step` peak)** and **`7.27 ms` median / `4.36 ms` best effective TPOT (`6.16×` median / `10.28×` peak speedup vs. `vLLM`)**.
