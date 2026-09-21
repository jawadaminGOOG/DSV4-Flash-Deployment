# DeepSeek-V4-Flash & DeepSeek-V4.1-Flash on Google Cloud TPU v6e (Trillium)

Production deployment manifests, custom Pallas/SparseCore TPU kernels, full three-workload concurrency sweeps (`1k/1k`, `8k/1k`, `1k/8k` across `C=1..512`), GPQA Diamond evaluations, and XProf kernel analyses for both **DeepSeek-V4-Flash (`284B` total / `13B` active, `256` experts)** and **DeepSeek-V4.1-Flash (`552B` total / `16B` active, `384` experts)** on a single **16-chip TPU v6e slice (`4×4` optical torus)**.

---

## 1. Executive Summary: Two Models on One 16-Chip TPU v6e Slice

| Metric / Feature | [`DeepSeek-V4-Flash` (`284B`)](./models/DeepSeekV4-Flash-v6e16/README.md) | [`DeepSeek-V4.1-Flash` (`552B`)](./models/DeepSeekV4.1-Flash-v6e16/README.md) |
|---|---|---|
| **Architecture** | `284B` total / `13B` active, `256` routed experts (`top_k=8`), CSA + HCA | `552B` total / `16B` active, `384` routed experts (`top_k=6`), Ratio-2 CSA2 + SWA + 4-stream `mHC` + Host-Offloaded Engram (`1` & `14`) |
| **Quantization on TPU v6e** | **INT8 MoE** (native v6e INT8 MXU) + **FP8** Dense & KV Cache | **MXFP4 MoE** (`4.25 bits/w` in HBM, VMEM `bf16` dequant in `gmm_v2`) + **FP8** Dense & KV Cache |
| **Resident HBM Weight Footprint** | **197.1 GiB** across 16 chips (`12.3 GiB/chip`) | **378.78 GiB** across 16 chips (`23.7 GiB/chip`) + **94.42 GiB/rank** Engram pinned in host DRAM |
| **`1k/1k` Balanced Peak Output (`C=512`)** | **8,463.7 out tok/s** (`16,906.7` total tok/s; `529.0 tok/s/chip`) | **5,137.3 out tok/s** (`2K` ctx) / **4,310.5 out tok/s** (`16K` ctx; **8,140.0 tok/s** steady-state decode) |
| **`8k/1k` Prefill-Heavy Peak (`C=512`)** | **1,758.5 out tok/s** (`17,585.3` total tok/s) | **1,234.6 out tok/s** (`11,109.1` total tok/s; **14,704.2 tok/s** peak engine prefill) |
| **`1k/8k` Reasoning Peak (`9,216` tok/req)** | **8,067.9 out tok/s** (`@ C=512`, `40.32 ms` TPOT) | **3,980.0 out tok/s** (`@ C=256`, `62.9 ms` TPOT; **6,294.9 tok/s** steady-state decode `@ C=512`) |
| **GPQA Diamond Accuracy (`n=198`, `16K` ctx)** | Greedy parity `16/16` byte-identical against unpatched control | **83.2% (`164/197`) unconditional Pass@1** / **94.8% (`164/173`) on completed chains**, **0% (`0/197`) loops** |
| **Request Reliability (`C=1..512`)** | **100%** (`3,079 / 3,079` across 30 points) | **100%** (`3,079 / 3,079` across 30 points at `16K` ctx + `1,028 / 1,028` at `2K` ctx) |
| **Recipes & Kernel Work** | [`models/DeepSeekV4-Flash-v6e16/`](./models/DeepSeekV4-Flash-v6e16/README.md) | [`models/DeepSeekV4.1-Flash-v6e16/`](./models/DeepSeekV4.1-Flash-v6e16/README.md) · [`patches/`](./models/DeepSeekV4.1-Flash-v6e16/patches/README.md) · [`XProf Report`](./models/DeepSeekV4.1-Flash-v6e16/results/xprof_kernel_report.md) |

---

## 2. Combined 3-Workload Performance Curves (`V4.1-Flash` vs `V4-Flash`, `C=1..512`)

![All 3 Workloads Combined Overlay](./models/DeepSeekV4.1-Flash-v6e16/results/charts/v41-vs-v4-all-workloads.png)

![All 3 Workloads 3x3 Grid](./models/DeepSeekV4.1-Flash-v6e16/results/charts/v41-vs-v4-3x3-grid.png)

---

## 3. Kernel-Specific Engineering & Optimization Linkage by Model

### 3.1 DeepSeek-V4-Flash (`284B`) Kernel Optimizations
Full recipe, ConfigMap patch, and sweep report: **[`models/DeepSeekV4-Flash-v6e16/README.md`](./models/DeepSeekV4-Flash-v6e16/README.md)** and **[`benchmark_sweep_report.md`](./models/DeepSeekV4-Flash-v6e16/results/benchmark_sweep_report.md)**.

| Kernel / Subsystem | Source Target | What Changed & Measured Hardware Impact |
|---|---|---|
| **1. Below-v7 Expert INT8 Enablement** | `tpu_inference/layers/vllm/quantization/mxfp4.py` | Stock upstream `tpu-inference` cannot compile FP4 `gmm_v2` on TPU v6e (`Unsupported type 'vector<8x128x8xf4E2M1FN>'`). Requantizes routed experts to `int8` (`block_size=512`) at load time so `gmm_v2` runs on the v6e INT8 MXU (`1,836 TOPS/chip`, achieving `96.6%` of peak HBM bandwidth at `11.93 ms/step`). |
| **2. KV-Group Weakref Memoization of Compressor Metadata** | `tpu_inference/kernels/experimental/deepseek_v4/compressor/compressor_v1.py` | Hoists and memoizes compressed-attention scatter/boundary metadata once per KV-cache group (`3` builds per forward pass instead of `~60` across layers) using a `weakref` context key. Cuts `compressor_v1.py` from `6.47 ms/step` to **`0.31 ms/step` (`-14.7%` total decode step time)**. |
| **3. SparseCore Offload for MoE Decode Combine** | `tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py` | Lowers the TensorCore/SparseCore dispatch crossover threshold so the post-expert token combine (`ragged_gather_reduce_v2`) executes on TPU v6e's dual SparseCores concurrently with TensorCore ops. Collapses MoE combine time on the critical path from `8.34 ms/step` to **`1.45 ms/step` (`-82.6%`)**, lifting `C=512` serving throughput to **`8,463.7 tok/s` (`+19.8%` vs unpatched `7,065.0 tok/s`)**. |

### 3.2 DeepSeek-V4.1-Flash (`552B`) TPU Backend & Kernel Engineering (`20` Patches)
Full recipe, 20-patch queue (`0001..0018` `tpu-inference` + `2` `vLLM`), and XProf waterfall: **[`models/DeepSeekV4.1-Flash-v6e16/README.md`](./models/DeepSeekV4.1-Flash-v6e16/README.md)**, **[`patches/README.md`](./models/DeepSeekV4.1-Flash-v6e16/patches/README.md)**, and **[`xprof_kernel_report.md`](./models/DeepSeekV4.1-Flash-v6e16/results/xprof_kernel_report.md)**.

| Kernel / Subsystem | Patch Link | What Changed & Measured Hardware Impact |
|---|---|---|
| **1. Removal of Unweighted Query `RMSNorm` (`qnorm`) in V4.1 Attention** | [`0018-fix-dsv41-remove-unweighted-per-head-RMSNorm...patch`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0018-fix-dsv41-remove-unweighted-per-head-RMSNorm-qnorm-f.patch) | **Primary full-context (`16K`) accuracy fix.** Replaced `rope_kernel.qnorm_rope` (inherited from V4.0) with `rope_kernel.rope` in `deepseek_v41_attention.py:387` matching `deepseek-inference/model.py:772`. Eliminates unit-RMS query distortion against `attn_sink`, taking GPQA Diamond to **83.2% unconditional / 94.8% completed Pass@1** with **0/197 loops**. |
| **2. Split Byte-Plane Compressed RoPE Record** | [`0007-Write-the-compressed-RoPE-record...patch`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0007-Write-the-compressed-RoPE-record-in-the-byte-plane-l.patch), [`0008`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0008-Test-that-the-RoPE-record-decodes-the-way-the-gather.patch) | Aligns `deepseek_v41_compressor.py` RoPE storage with `csa_gather`'s two-byte-plane decode (`high` byte at `i`, `low` byte at `64 + i` $\rightarrow$ `(high << 8) \| low`). |
| **3. Exclusive Compressed Causal Bound & Chronological `streamindex_topk`** | [`0013`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0013-Fix-the-indexer-causal-bound-for-compressed-KV-state.patch), [`0014`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0014-Implement-short-context-indexer-bypass-matching-refe.patch), [`0015`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0015-Import-lax-and-sort-streamindex_topk-outputs-chronol.patch) | Fixes `streamindex_topk.py` Pallas causal mask to `k_span < (q_pos + 1) // ratio`, adds short-context bypass (`< 1,024` tokens), and sorts top-512 selected slots chronologically. |
| **4. `float32` SWA/SparseMLA Merge & Page-Aligned `_start_offset`** | [`0016`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0016-core_attention-retain-float32-accumulator-precision-.patch), [`0017`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0017-Fix-mla_swa-page-aligned-_start_offset-and-deduplica.patch) | Retains `f32` accumulators across the `SWA` (`1.44 ms/step` decode) and `SparseMLA` merge boundary and deduplicates KV-cache DMA updates in `mla_swa.py`. |
| **5. Native MXFP4 VMEM Dequant & Compact `[E, blocks, N]` Scale (`-47.46 GiB`)** | [`0001`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0001-Quantization-Claim-every-DeepSeek-V4-family-model-ty.patch), [`0005`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0005-Give-every-V4.1-sliding-window-layer-its-own-KV-cach.patch) | Keeps `384` routed experts in native 4-bit MXFP4 (`e2m1` + compact `u8` `e8m0` scale without second-minor broadcast padding, saving **`47.46 GiB` HBM** across 16 chips) and unpacks to `bf16` in VMEM inside `megablox/gmm_v2.py`. |
| **6. Host-Pinned `94.42 GiB/rank` Engram Lookup & Per-Host RunAI Lock** | [`0005`](./models/DeepSeekV4.1-Flash-v6e16/patches/tpu-inference/0005-Give-every-V4.1-sliding-window-layer-its-own-KV-cach.patch), [`vllm/0001`](./models/DeepSeekV4.1-Flash-v6e16/patches/vllm/0001-engram-add-a-non-CUDA-reference-path.patch), [`vllm/0002`](./models/DeepSeekV4.1-Flash-v6e16/patches/vllm/0002-weight_utils-serialise-the-runai-streamer-per-host.patch) | Offloads the `203.1 GB` Engram tables (`layers 1 & 14`) to pinned host DRAM (`< 0.6 ms` host gather) and serializes per-host GCS weight loading via `/dev/shm/vllm_runai_streamer_host.lock`. |

### 3.3 XProf Decode Breakdown & Next Hill-Climbing Targets (`DeepSeek-V4.1-Flash`)

Full report: **[`models/DeepSeekV4.1-Flash-v6e16/results/xprof_kernel_report.md`](./models/DeepSeekV4.1-Flash-v6e16/results/xprof_kernel_report.md)**.

![Decode Step Breakdown — V4.1-Flash vs V4-Flash](./models/DeepSeekV4.1-Flash-v6e16/results/charts/v41-xprof-decode-breakdown.png)

| Priority | Target Subsystem | XProf Measured Cost (`C=256` Decode) | Mechanism & Optimization Path | Recoverable Band |
|---:|---|---:|---|---:|
| **#1** | `fused_moe_gmm.py:330` + `deepseek_v4.py:91` | **`16.05 ms/step`** (`24.7%` decode; `114.58 ms` = `42.4%` prefill) | `%all-reduce.52` (`bf16[256,5120]`, `2.62 MB`) takes `356.5 µs/layer` vs `25.9 µs` for `%all-gather.122` (`~306 µs/layer` barrier wait), and `%fusion.647` couples `shared_experts.down_proj` to `%all-reduce.52`. Enable SparseCore `ENABLE_RS_KERNEL=1` + decouple `%fusion.647`. | **`2.0 – 8.0 ms/step`** |
| **#2** | `deepseek_v41_compressor.py:490` & `csa_gather.py:290` | **`4.40 ms/step`** (`6.8%` decode; `11.18 ms` prefill) | `.reshape(-1, 512)` on `u8[267,512,4,128]` (`69.98 MB`) changes minor tiling `T(4,128)(4,1) <-> T(8,128)(4,1)`, emitting physical `70 MB` HBM copies (`reshape.22232` & `reshape.23632`). Index 4D `cache` directly in Pallas. | **`2.5 – 3.1 ms/step`** |
| **#3** | `megablox/gmm_v2.py` (`tm_256` / `e2m1_to_bf16`) | **`20.62 ms/step`** (`31.8%` decode vs `10.73 ms` HBM floor) | `m=1536` across `g=24` groups (`~64` rows/group) tiled at `tm=256` pads up to `6,144` rows/layer (`18.5 ms` MXU floor). Autotune decode `tm ∈ {16, 32}` and/or register-LUT `e2m1_to_bf16`. | **`2.0 – 9.0 ms/step`** |
| **#4** | `deepseek_v41_indexer.py:291` & `streamindex_topk.py` | **`5.07 ms/step`** (`7.8%` decode; `14.74 ms` prefill) | Remove `jax.lax.cond` (`1.99 ms/step`) and bound Pallas `streamindex_topk` grid dynamically by `ceil_div(max_kv_len, block_k)` instead of sorting `s32[16, 18432]`. | **`1.0 – 3.0 ms/step`** |
| **#5** | `vllm/.../kernels/mhc/torch.py` | **`1.07 ms/step`** (`7,398` launches/step = `43.6%` of all launches; `8.83 ms` prefill) | Fuse 4-stream `pre_mix` + `RMSNorm` + `post_mix` into a single Pallas VMEM kernel in `T(8,128)`. | **`0.7 – 1.6 ms/step`** |

---

## 4. Benchmark Tables (`DeepSeek-V4-Flash` vs `DeepSeek-V4.1-Flash`, `1k/1k`)

| Concurrency ($C$) | V4-Flash Output (`tok/s`) | V4-Flash TPOT | V4.1-Flash (`2K` ctx) Output (`tok/s`) | V4.1 (`2K`) TPOT | V4.1-Flash (`16K` ctx) Output (`tok/s`) | V4.1 (`16K`) TPOT |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 35.3 | 27.7 ms | 18.9 | 49.1 ms | 17.9 | 54.7 ms |
| **2** | 70.7 | 27.8 ms | 39.5 | 49.6 ms | 35.4 | 55.5 ms |
| **4** | 140.5 | 28.0 ms | 78.7 | 49.8 ms | 69.5 | 56.5 ms |
| **8** | 277.5 | 28.4 ms | 154.4 | 50.8 ms | 144.4 | 54.4 ms |
| **16** | 598.6 | 26.3 ms | 444.5 | 34.9 ms | 354.1 | 44.1 ms |
| **32** | 1,125.4 | 26.9 ms | 821.6 | 37.0 ms | 655.0 | 46.7 ms |
| **64** | 2,089.5 | 28.8 ms | 1,601.4 | 36.5 ms | 858.3 | 68.6 ms |
| **128** | 4,113.0 | 28.6 ms | 2,778.7 | 40.0 ms | 2,017.6 | 57.1 ms |
| **256** | 6,671.9 | 33.7 ms | 4,096.9 | 51.4 ms | 3,164.1 | 69.2 ms |
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

- **[`models/DeepSeekV4-Flash-v6e16/`](./models/DeepSeekV4-Flash-v6e16/README.md)** — Serving manifest (`dsv4-flash-v6e16-serving.yaml` with ConfigMap patches), 3-shape concurrency sweep (`1k/1k`, `8k/1k`, `1k/8k`), raw JSONs, and charts for `DeepSeek-V4-Flash` (`284B`).
- **[`models/DeepSeekV4.1-Flash-v6e16/`](./models/DeepSeekV4.1-Flash-v6e16/README.md)** — Serving manifest (`dsv41-flash-v6e16-serving.yaml`), [`patches/`](./models/DeepSeekV4.1-Flash-v6e16/patches/README.md) (`18` `tpu-inference` + `2` `vLLM` patches + `apply.sh`), 3-shape concurrency sweeps, 198-question GPQA Diamond results, and [`results/xprof_kernel_report.md`](./models/DeepSeekV4.1-Flash-v6e16/results/xprof_kernel_report.md) for `DeepSeek-V4.1-Flash` (`552B`).
- **[`recipes/`](./recipes/)** — Quick-start single-file Kubernetes template and async streaming benchmark client.
