# DeepSeek-V4-Flash & DeepSeek-V4.1-Flash on Google Cloud TPU v6e (Trillium)

Production deployment manifests, custom Pallas/SparseCore TPU kernels, full three-workload concurrency sweeps (`1k/1k`, `8k/1k`, `1k/8k` across `C=1..512`), GPQA Diamond evaluations, and kernel optimization deep dives for both **DeepSeek-V4-Flash (`284B` total / `13B` active, `256` experts)** and **DeepSeek-V4.1-Flash (`552B` total / `16B` active, `384` experts)** on a single **16-chip TPU v6e slice (`4×4` optical torus)**.

---

## 1. Executive Summary: Two Models on One 16-Chip TPU v6e Slice

| Metric / Feature | [`DeepSeek-V4-Flash` (`284B`)](./models/DeepSeekV4-Flash-v6e16/README.md) | [`DeepSeek-V4.1-Flash` (`552B`)](./models/DeepSeekV4.1-Flash-v6e16/README.md) |
|---|---|---|
| **Architecture** | `284B` total / `13B` active, `256` routed experts (`top_k=8`), CSA + HCA | `552B` total / `16B` active, `384` routed experts (`top_k=6`), Ratio-2 CSA2 + SWA + 4-stream `mHC` + Host-Offloaded Engram (`1` & `14`) |
| **Quantization on TPU v6e** | **INT8 MoE** (native v6e INT8 MXU) + **FP8** Dense & KV Cache | **MXFP4 MoE** (`4.25 bits/w` in HBM, 3-op bitwise IEEE-754 `bf16` dequant in `gmm_v2`) + **FP8** Dense & KV Cache |
| **Resident HBM Weight Footprint** | **197.1 GiB** across 16 chips (`12.3 GiB/chip`) | **378.78 GiB** across 16 chips (`23.7 GiB/chip`) + **94.42 GiB/rank** Engram pinned in host DRAM |
| **`1k/1k` Balanced Peak Output (`C=512`)** | **8,463.7 out tok/s** (`16,906.7` total tok/s; `529.0 tok/s/chip`) | **5,137.3 out tok/s** (`2K` ctx) / **4,310.5 out tok/s** (`16K` ctx; **8,140.0 tok/s** steady-state decode) |
| **`8k/1k` Prefill-Heavy Peak (`C=512`)** | **1,758.5 out tok/s** (`17,585.3` total tok/s) | **1,234.6 out tok/s** (`11,109.1` total tok/s; **14,704.2 tok/s** peak engine prefill) |
| **`1k/8k` Reasoning Peak (`9,216` tok/req)** | **8,067.9 out tok/s** (`@ C=512`, `40.32 ms` TPOT) | **3,980.0 out tok/s** (`@ C=256`, `62.9 ms` TPOT; **6,294.9 tok/s** steady-state decode `@ C=512`) |
| **Low-Latency Single-Stream (`C=1`)** | `35.3 out tok/s` (`27.7 ms` TPOT) | **425.5 out tok/s/req** (`2.35 ms/token` via [`megakernel-recipe/`](./megakernel-recipe/README.md) + `DSpark` `1+7`) |
| **GPQA Diamond Accuracy (`n=198`, `16K` ctx)** | Greedy parity `16/16` byte-identical against unpatched control | **83.2% (`164/197`) unconditional Pass@1** / **94.8% (`164/173`) on completed chains**, **0% (`0/197`) loops** |
| **Request Reliability (`C=1..512`)** | **100%** (`3,079 / 3,079` across 30 points) | **100%** (`3,079 / 3,079` across 30 points at `16K` ctx + `1,028 / 1,028` at `2K` ctx) |
| **Recipes & Kernel Work** | [`models/DeepSeekV4-Flash-v6e16/`](./models/DeepSeekV4-Flash-v6e16/README.md) | [`models/DeepSeekV4.1-Flash-v6e16/`](./models/DeepSeekV4.1-Flash-v6e16/README.md) · [`Kernel Optimizations Deep Dive`](./models/DeepSeekV4.1-Flash-v6e16/KERNEL-OPTIMIZATIONS.md) · [`Megakernel Recipe`](./megakernel-recipe/README.md) |

---

## 2. Combined 3-Workload Performance Curves (`V4.1-Flash` vs `V4-Flash`, `C=1..512`)

![All 3 Workloads Combined Overlay](./models/DeepSeekV4.1-Flash-v6e16/results/charts/v41-vs-v4-all-workloads.png)

![All 3 Workloads 3x3 Grid](./models/DeepSeekV4.1-Flash-v6e16/results/charts/v41-vs-v4-3x3-grid.png)

---

## 3. Kernel Optimizations

### 3.1 DeepSeek-V4.1-Flash (`552B`): Before vs. After Kernel Optimizations (`61.71 ms -> 37.33 ms/step`)

For a visual, blog-style deep dive into the three TPU v6e hardware bottlenecks and how each kernel optimization works under the hood, read **[`models/DeepSeekV4.1-Flash-v6e16/KERNEL-OPTIMIZATIONS.md`](./models/DeepSeekV4.1-Flash-v6e16/KERNEL-OPTIMIZATIONS.md)**. For our standalone 51-layer persistent Pallas Megakernel + `DSpark` (`1+7`) speculative decoding engine (`425.5 tok/s/req` at `C=1`), read **[`megakernel-recipe/README.md`](./megakernel-recipe/README.md)**.

![Decode Step Breakdown — V4.1-Flash vs V4-Flash](./models/DeepSeekV4.1-Flash-v6e16/results/charts/v41-xprof-decode-breakdown.png)

| Metric / Concurrency (`1k in / 1k out`, `16K` Ctx) | Before Kernel Optimizations | After Kernel Optimizations (Final) | Improvement |
|---|---:|---:|---:|
| **Hardware Decode Step Wall (`B=64`)** | `61.71 ms / step` | **`37.33 ms / step`** | **`-39.5%` (`1.65×` faster)** |
| **Hardware Prefill Step Wall (`4,096` tok)** | `270.65 ms / step` | **`245.53 ms / step`** | **`-9.3%` (`-25.12 ms`)** |
| **Single-Layer Routed MoE (`gmm_v2` `w13+w2`)** | `3,361.9 µs` | **`1,945.5 µs`** | **`-42.1%` (`1.73×` faster)** |
| **KV-Cache Reshape/Copy Overhead (`51` layers)** | `2.5385 ms / step` | **`0.0003 ms / step`** | **`-99.99%` (zero-copy `%bitcast`)** |
| **`C = 1` Output Throughput (`tok/s`)** | `17.9 tok/s` (`54.7 ms` TPOT) | **`18.3 tok/s`** (`53.8 ms` TPOT)* | **`+2.2%`** (*`425.5 tok/s` via [Megakernel](./megakernel-recipe/README.md)*) |
| **`C = 16` Output Throughput (`tok/s`)** | `354.1 tok/s` (`44.1 ms` TPOT) | **`374.2 tok/s`** (`41.7 ms` TPOT) | **`+5.7%`** |
| **`C = 32` Output Throughput (`tok/s`)** | `655.0 tok/s` (`46.7 ms` TPOT) | **`741.8 tok/s`** (`41.2 ms` TPOT) | **`+13.3%`** |
| **`C = 64` Output Throughput (`tok/s`)** | `858.3 tok/s` (`68.6 ms` TPOT) | **`1,524.5 tok/s`** (`38.5 ms` TPOT) | **`+77.6%` (`-43.9%` TPOT)** |
| **`C = 128` Output Throughput (`tok/s`)** | `2,017.6 tok/s` (`57.1 ms` TPOT) | **`2,520.7 tok/s`** (`45.2 ms` TPOT) | **`+24.9%` (`-20.8%` TPOT)** |
| **`C = 256` Output Throughput (`tok/s`)** | `3,164.1 tok/s` (`69.2 ms` TPOT) | **`3,485.0 tok/s`** (`62.8 ms` TPOT) | **`+10.1%` (`-9.2%` TPOT)** |
| **`C = 512` Output Throughput (`tok/s`)** | `4,310.5 tok/s` (`94.8 ms` TPOT) | **`4,310.5 tok/s`** (`94.8 ms` TPOT) | Prefill-chunk bound (`8,140 tok/s` decode) |

### 3.2 Summary of DeepSeek-V4.1-Flash TPU Backend & Kernel Engineering (`27` Patches)

Full patch queue (`25` `tpu-inference` + `2` `vLLM`) and reproduction script: **[`models/DeepSeekV4.1-Flash-v6e16/patches/README.md`](./models/DeepSeekV4.1-Flash-v6e16/patches/README.md)**.

| Kernel / Subsystem | Patch Link | What Changed & Measured Hardware Impact |
|---|---|---|
| **1. Removal of Unweighted Query `RMSNorm` (`qnorm`) in V4.1 Attention** | [`0018`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0018-fix-dsv41-remove-unweighted-per-head-RMSNorm-qnorm-f.patch) | **Primary full-context (`16K`) accuracy fix.** Replaced `rope_kernel.qnorm_rope` (inherited from V4.0) with `rope_kernel.rope` in `deepseek_v41_attention.py:387`. Eliminates unit-RMS query distortion against `attn_sink`, taking GPQA Diamond to **83.2% unconditional / 94.8% completed Pass@1** with **0/197 loops**. |
| **2. Split Byte-Plane Compressed RoPE & Exclusive Causal Top-K** | [`0007`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0007-Write-the-compressed-RoPE-record-in-the-byte-plane-l.patch), [`0013`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0013-Fix-the-indexer-causal-bound-for-compressed-KV-state.patch)–[`0017`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0017-Fix-mla_swa-page-aligned-_start_offset-and-deduplica.patch) | Aligns `deepseek_v41_compressor.py` RoPE storage with `csa_gather`'s two-byte-plane decode, fixes `streamindex_topk.py` causal mask to `k_span < (q_pos + 1) // ratio`, and retains `float32` softmax accumulators across the `SWA`/`SparseMLA` boundary. |
| **3. Native MXFP4 VMEM Dequant & Compact `[E, blocks, N]` Scale (`-47.46 GiB`)** | [`0001`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0001-Quantization-Claim-every-DeepSeek-V4-family-model-ty.patch), [`0005`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0005-Give-every-V4.1-sliding-window-layer-its-own-KV-cach.patch) | Keeps `384` routed experts in native 4-bit MXFP4 (`e2m1` + compact `u8` `e8m0` scale without second-minor broadcast padding, saving **`47.46 GiB` HBM** across 16 chips) and pins the `203.1 GB` Engram tables (`layers 1 & 14`) in host DRAM. |
| **4. SparseCore `nd_reduce_scatter` Offload & Shared-Expert Overlap** | [`0019`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0019-perf-moe-decouple-shared_experts.down_proj-from-post.patch) | Offloads post-MoE `reduce-scatter` to SparseCore (`SC Overlay`) and decouples `shared_experts.down_proj` via `jax.lax.optimization_barrier`, cutting decode step wall by **`-5.76 ms/step`** and prefill step wall by **`-25.12 ms/step`**. |
| **5. 4D/3D KV-Cache `%bitcast` Views & `mode="drop"` Scatter** | [`0020`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0020-perf-dsv41-preserve-4D-KV-cache-minor-tile-dimension.patch) | Preserves `(4, 128)` (`nope_cache`) and `(4, 256)` (`indexer` `cache`) minor tile dimensions via 3D views, eliminating **`99.99%` of KV-cache HBM relayout copies (`2.5385 -> 0.0003 ms/step`)**. |
| **6. `gmm_v2` `(tile_m=128, bucket_base=128)` + Bitwise IEEE-754 Dequant** | [`0021`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0021-perf-megablox-bitwise-IEEE-754-decode_e2m1-e8m0-and-.patch)–[`0025`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0025-perf-megablox-support-3D-compact_scale-E-num_blocks-.patch) | Replaces the 12-op `decode_e2m1`/`decode_e8m0` VPU sequence in `megablox/gmm_v2.py` with a 3-op bitwise IEEE-754 assembly and sets `(tile_m=128, bucket_base=128)`, cutting routed MoE kernel time by **`-49.2%`** and total decode step wall to **`37.33 ms/step`**. |

### 3.3 DeepSeek-V4-Flash (`284B`) Kernel Optimizations
Full recipe, ConfigMap patch, and sweep report: **[`models/DeepSeekV4-Flash-v6e16/README.md`](./models/DeepSeekV4-Flash-v6e16/README.md)** and **[`benchmark_sweep_report.md`](./models/DeepSeekV4-Flash-v6e16/results/benchmark_sweep_report.md)**.

| Kernel / Subsystem | Source Target | What Changed & Measured Hardware Impact |
|---|---|---|
| **1. Below-v7 Expert INT8 Enablement** | `tpu_inference/layers/vllm/quantization/mxfp4.py` | Requantizes routed experts to `int8` (`block_size=512`) at load time so `gmm_v2` runs on the v6e INT8 MXU (`1,836 TOPS/chip`, achieving `96.6%` of peak HBM bandwidth at `11.93 ms/step`). |
| **2. KV-Group Weakref Memoization of Compressor Metadata** | `tpu_inference/kernels/experimental/deepseek_v4/compressor/compressor_v1.py` | Hoists and memoizes compressed-attention scatter/boundary metadata once per KV-cache group (`3` builds per forward pass instead of `~60` across layers), cutting `compressor_v1.py` from `6.47 ms/step` to **`0.31 ms/step` (`-14.7%` total decode step time)**. |
| **3. SparseCore Offload for MoE Decode Combine** | `tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py` | Executes post-expert token combine (`ragged_gather_reduce_v2`) on TPU v6e's dual SparseCores concurrently with TensorCore ops, collapsing MoE combine time from `8.34 ms/step` to **`1.45 ms/step` (`-82.6%`)** and lifting `C=512` throughput to **`8,463.7 tok/s` (`+19.8%`)**. |

---

## 4. Benchmark Tables (`DeepSeek-V4-Flash` vs `DeepSeek-V4.1-Flash`, `1k/1k`)

| Concurrency ($C$) | V4-Flash Output (`tok/s`) | V4-Flash TPOT | V4.1-Flash (`2K` ctx) Output (`tok/s`) | V4.1 (`2K`) TPOT | V4.1-Flash (`16K` ctx) Output (`tok/s`) | V4.1 (`16K`) TPOT |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 35.3 | 27.7 ms | 18.9 | 49.1 ms | 18.3 | 53.8 ms |
| **2** | 70.7 | 27.8 ms | 39.5 | 49.6 ms | 36.4 | 54.1 ms |
| **4** | 140.5 | 28.0 ms | 78.7 | 49.8 ms | 72.1 | 54.6 ms |
| **8** | 277.5 | 28.4 ms | 154.4 | 50.8 ms | 151.2 | 52.0 ms |
| **16** | 598.6 | 26.3 ms | 444.5 | 34.9 ms | 374.2 | 41.7 ms |
| **32** | 1,125.4 | 26.9 ms | 821.6 | 37.0 ms | 741.8 | 41.2 ms |
| **64** | 2,089.5 | 28.8 ms | 1,601.4 | 36.5 ms | **1,524.5** | **38.5 ms** |
| **128** | 4,113.0 | 28.6 ms | 2,778.7 | 40.0 ms | **2,520.7** | **45.2 ms** |
| **256** | 6,671.9 | 33.7 ms | 4,096.9 | 51.4 ms | **3,485.0** | **62.8 ms** |
| **512** | **8,463.7** | **50.2 ms** | **5,137.3** | **77.3 ms** | **4,310.5** | **94.8 ms** |

---

## 5. Hardware Reference & Cost Efficiency on TPU v6e-16

| Property | TPU v6e-16 Value | Notes |
| :--- | :--- | :--- |
| **HBM Capacity** | `32 GB/chip` (`31.24 GiB` addressable) | `512 GB` (`499.84 GiB`) total across 16 chips |
| **Memory Bandwidth** | `1,638 GB/s/chip` | `26.2 TB/s` aggregate across 16 chips |
| **Compute Peak (INT8 / BF16)** | `1,836 TOPS` / `918 TFLOPS` per chip | `29.37 POPS` INT8 / `14.68 PFLOPS` BF16 aggregate |
| **Interconnect (ICI)** | `800 Gbps` bidirectional per chip | Direct optical 2D `4×4` torus (`4 × ct6e-standard-4t`) |

| Model & Pricing Tier (`16 × TPU v6e`) | Hourly Slice Cost | Peak Output `tok/s` | Cost per `1M` Output Tokens | Output Tokens per `$1.00` |
| :--- | :---: | :---: | :---: | :---: |
| **DeepSeek-V4-Flash (`284B`, 3-Yr CUD)** | `$19.52 / hr` | `8,463.7` | **`$0.64`** | **`1,561,000`** |
| **DeepSeek-V4-Flash (`284B`, Spot)** | `$21.60 / hr` | `8,463.7` | **`$0.71`** | **`1,411,000`** |
| **DeepSeek-V4.1-Flash (`552B`, 3-Yr CUD)** | `$19.52 / hr` | `5,137.3` (`8,140` decode) | **`$1.06`** (`$0.67` decode) | **`947,000`** (`1,500,000` decode) |
| **DeepSeek-V4.1-Flash (`552B`, Spot)** | `$21.60 / hr` | `5,137.3` (`8,140` decode) | **`$1.17`** (`$0.74` decode) | **`856,000`** (`1,356,000` decode) |

---

## 6. Repository Directory Structure

- **[`models/DeepSeekV4.1-Flash-v6e16/`](./models/DeepSeekV4.1-Flash-v6e16/README.md)** — Unified serving manifest (`dsv41-flash-v6e16-serving.yaml`), [`KERNEL-OPTIMIZATIONS.md`](./models/DeepSeekV4.1-Flash-v6e16/KERNEL-OPTIMIZATIONS.md) visual deep dive, [`patches/`](./models/DeepSeekV4.1-Flash-v6e16/patches/README.md) (`25` `tpu-inference` + `2` `vLLM` patches + `apply.sh`), 3-shape concurrency sweeps, and 198-question GPQA Diamond results for `DeepSeek-V4.1-Flash` (`552B`).
- **[`megakernel-recipe/`](./megakernel-recipe/README.md)** — Standalone 51-layer persistent Pallas Megakernel + `DSpark` (`1+7`) speculative decoding recipe (`425.5 accepted tok/s/req`, `2.35 ms/token` at `C=1`), covering Motivation, Approach, Results, and Architectural Learnings.
- **[`models/DeepSeekV4-Flash-v6e16/`](./models/DeepSeekV4-Flash-v6e16/README.md)** — Serving manifest (`dsv4-flash-v6e16-serving.yaml` with ConfigMap patches), 3-shape concurrency sweep (`1k/1k`, `8k/1k`, `1k/8k`), raw JSONs, and charts for `DeepSeek-V4-Flash` (`284B`).
- **[`recipes/`](./recipes/)** — Quick-start single-file Kubernetes template and async streaming benchmark client.
