# DeepSeek-V4.1-Flash on TPU v6e-16

Production deployment recipes, custom Pallas/SparseCore TPU kernels, 198-question GPQA Diamond accuracy evaluations (`94.8%` completed-chain Pass@1), and full three-workload concurrency sweeps (`1k/1k`, `8k/1k`, `1k/8k` across `C=1..512`) for **`deepseek-ai/DeepSeek-V4.1-Flash`** (`552B` total / `16B` active parameters, `384` routed experts) on a single 16-chip **TPU v6e** slice (`4 × ct6e-standard-4t` in a `4×4` optical torus on GKE).

---

## 1. Three Self-Contained Serving & Kernel Optimization Recipes

All deployments, patches, technical notes, benchmark results, and reproduction scripts are organized into **three independent, self-contained recipe directories** under `models/DeepSeekV4.1-Flash-v6e16/`, tailored to three hardware operating regimes:

| Recipe Directory | Target Regime & Architecture | Decode Step Wall (`C=64`) | Headline Throughput & Latency | Self-Contained Contents |
|---|---|---|---|---|
| **[`./kernel-optimizations-recipe/`](./kernel-optimizations-recipe/README.md)** | **Full-Context Batched vLLM/XLA (`C=1..512`)**<br>SparseCore `nd_reduce_scatter` offload, 3D KV-cache `%bitcast` views (`-99.99%` relayout), 3-op bitwise IEEE-754 MXFP4 unpacking (`tile_m=128`) | **`37.33 ms/step`**<br>(`-39.5%` vs. `61.71 ms` baseline) | **`1,524.5 tok/s`** (`@ C=64`)<br>**`2,520.7 tok/s`** (`@ C=128`)<br>**`5,137.3 tok/s`** (`@ C=512` peak) | [`README.md`](./kernel-optimizations-recipe/README.md) · [`TECH-NOTE.md`](./kernel-optimizations-recipe/TECH-NOTE.md) · [`dsv41-flash-v6e16-serving.yaml`](./kernel-optimizations-recipe/dsv41-flash-v6e16-serving.yaml) · [`patches/`](./kernel-optimizations-recipe/patches/README.md) (`0001..0025`, `4b8edd8e`) · [`results/`](./kernel-optimizations-recipe/results/) · [`scripts/`](./kernel-optimizations-recipe/scripts/) |
| **[`./megakernel-recipe/`](./megakernel-recipe/README.md)** | **Ultra-Low-Latency Persistent Pallas Megakernel + `DSpark` (`C=1..24`)**<br>51-layer persistent VMEM Pallas kernel (`12.99 / 16.00 MiB` VMEM) + `DSpark` (`1+7`) speculative decoding (`0` XLA barrier launches/step) | **`3.21 ms` base**<br>**`2.35 ms/tok` effective** | **`425.5 tok/s/req`** (`@ C=1`, `15.9×`)<br>**`1,427.0 tok/s`** (`@ C=16`, `3.4×`) | [`README.md`](./megakernel-recipe/README.md) · [`TECH-NOTE.md`](./megakernel-recipe/TECH-NOTE.md) · [`dsv41-flash-v6e16-megakernel-serving.yaml`](./megakernel-recipe/dsv41-flash-v6e16-megakernel-serving.yaml) · [`patches/`](./megakernel-recipe/patches/README.md) (`0001..0026`) · [`megakernel/`](./megakernel-recipe/megakernel/) · [`results/`](./megakernel-recipe/results/) · [`scripts/`](./megakernel-recipe/scripts/) |
| **[`./fused-kernels-recipe/`](./fused-kernels-recipe/README.md)** | **High-Concurrency Fused `W13+SiLU+W2` MoE + Hybrid `EP=8 × TP=2` (`C=64..256`)**<br>Pairwise ICI `ppermute` (`48` half-width experts/pair, `-37.0%` straggler wait) + single-kernel `gmm_v2_fused_w13_w2` (`23.35 / 30.72 MiB` VMEM) | **`32.72 ms/step`**<br>(`-47.0%` vs. `61.71 ms` baseline) | **`1,740.4 tok/s`** (`@ C=64`, `+14.2%`)<br>**`2,984.5 tok/s`** (`@ C=128`, `+18.4%`)<br>**`4,758.5 tok/s`** (`@ C=190` 1-wave max, `297.4 tok/s/chip`) | [`README.md`](./fused-kernels-recipe/README.md) · [`TECH-NOTE.md`](./fused-kernels-recipe/TECH-NOTE.md) · [`dsv41-flash-v6e16-fused-kernels-serving.yaml`](./fused-kernels-recipe/dsv41-flash-v6e16-fused-kernels-serving.yaml) · [`patches/`](./fused-kernels-recipe/patches/README.md) (`0001..0026`, `623b2904`) · [`results/`](./fused-kernels-recipe/results/) · [`scripts/`](./fused-kernels-recipe/scripts/) |

---

## 2. Highlights Across the Three Recipes

- **Full-Context (`16,384`-token) Correctness & GPQA Diamond Verified (`198` Questions, `T=1.0`):**
  - **83.2% (`164/197`) unconditional Pass@1** and **94.8% (`164/173`) Pass@1 on completed chains** at `max_tokens=14336`.
  - **87.8% (`173/197`) natural `<|EOT|>` termination** and **0.0% (`0/197`) degenerate repetition loops**.
  - **100% (`55/55`) pass rate** across known-answer smoke (`8/8`), fine boundary (`18/18`), length control (`23/23`), and long-context needle-in-a-haystack (`14/14` up to `2,056` tokens).
- **100% Request Reliability Across All Concurrency Sweep Points (`C=1..512`):**
  - **Ultra-Low Latency (`C=1..24`, [`megakernel-recipe/`](./megakernel-recipe/README.md)):** Achieves **`425.5 accepted output tok/s/req` (`2.35 ms/token` effective TPOT)** at `C=1` (`15.9×` faster than standard XLA dispatch) and **`1,427.0 output tok/s`** at `C=16`.
  - **High-Concurrency Single-Wave Saturation (`C=64..190`, [`fused-kernels-recipe/`](./fused-kernels-recipe/README.md)):** Achieves **`1,740.4 output tok/s` (`34.5 ms` TPOT)** at `C=64` (`+102.8%` over unoptimized baseline), **`2,984.5 output tok/s` (`38.5 ms` TPOT)** at `C=128`, and **`4,758.5 output tok/s` (`297.4 tok/s/chip`)** at the `C=190` single-wave KV ceiling.
  - **Multi-Wave & Prefill-Heavy Throughput (`C=256..512`, [`kernel-optimizations-recipe/`](./kernel-optimizations-recipe/README.md)):** Peaks at **`5,137.3 output tok/s` (`2K` ctx) / `4,310.5 output tok/s` (`16K` ctx; `8,140.0 tok/s` steady-state decode)** on `1k/1k`, **`1,234.6 output tok/s` (`14,704.2 tok/s` engine prefill)** on `8k/1k`, and **`3,980.0 output tok/s` (`6,294.9 tok/s` steady-state decode)** on `1k/8k`.

---

## 3. Combined 3-Workload Comparison (`V4.1-Flash` vs `V4-Flash`, `C=1..512`)

![All 3 Workloads Combined Overlay](kernel-optimizations-recipe/results/charts/v41-vs-v4-all-workloads.png)

![All 3 Workloads 3x3 Grid](kernel-optimizations-recipe/results/charts/v41-vs-v4-3x3-grid.png)

| Workload Pattern | V4.1 Peak Output `tok/s` (`16K` ctx) | V4.1 Peak Total `tok/s` (`16K` ctx) | V4.1 Peak Engine Rate | V4-Flash Peak Output `tok/s` (`2K`/`9K` ctx) | V4.1 / V4 Ratio |
|---|---:|---:|---:|---:|---:|
| **`1k/1k` (Balanced, `2K` ctx)** | **5,137.3** (`@ C=512`) | **10,262.1** | **8,140.0 decode tok/s** | 8,463.7 (`@ C=512`) | **0.61×** e2e (`0.96×` decode) |
| **`1k/1k` (Balanced, `16K` ctx)** | **4,758.5** (`@ C=190` [Fused MoE](fused-kernels-recipe/README.md)) / **4,310.5** (`@ C=512`) | **9,517.0** | — | 8,463.7 (`@ C=512`) | **0.56×** |
| **`8k/1k` (Prefill-Heavy, `16K` ctx)** | **1,234.6** (`@ C=512`) | **11,109.1** | **14,704.2 prefill tok/s** | 1,758.5 (`@ C=512`) | **0.70×** |
| **`1k/8k` (Reasoning, `16K` ctx)** | **3,980.0** (`@ C=256`) | **4,476.3** | **6,294.9 decode tok/s** | 8,067.9 (`@ C=512`) | **0.49×** e2e (`0.78×` decode) |

---

## 4. End-to-End Progression Across Optimization Recipes (`1k in / 1k out`, `16,384` Context)

![Decode Step Breakdown — V4.1-Flash vs V4-Flash](kernel-optimizations-recipe/results/charts/v41-xprof-decode-breakdown.png)

![Fused W13+SiLU+W2 + Hybrid EP=8 x TP=2 Analysis](fused-kernels-recipe/results/charts/fused-moe-ep8-tp2-analysis.png)

| Metric / Concurrency (`C`) | Unoptimized Baseline (`d85ce9c1`) | [`kernel-optimizations-recipe/`](./kernel-optimizations-recipe/README.md) (`4b8edd8e`) | **[`fused-kernels-recipe/`](./fused-kernels-recipe/README.md) (`623b2904`)** | Total Improvement |
|---|---:|---:|---:|---:|
| **Hardware Decode Step Wall (`B=64`)** | `61.71 ms / step` | `37.33 ms / step` | **`32.72 ms / step`** | **`-47.0%` (`1.89×` faster)** |
| **Hardware Prefill Step Wall (`4,096` tok)** | `270.65 ms / step` | `245.53 ms / step` | **`245.53 ms / step`** | **`-9.3%` (`-25.12 ms`)** |
| **ICI Collective Straggler Wait (`40` MoE layers)** | `16.08 ms / step` | `11.07 ms / step` | **`6.97 ms / step`** | **`-56.7%` (`-37.0%` vs. EP=16)** |
| **KV-Cache Reshape/Copy Overhead (`51` layers)** | `2.5385 ms / step` | `0.0003 ms / step` | **`0.0003 ms / step`** | **`-99.99%` (zero-copy `%bitcast`)** |
| **`C = 1` Output Throughput (`tok/s`)** | `17.9 tok/s` (`54.7 ms` TPOT) | `18.3 tok/s` (`53.8 ms` TPOT) | **`425.5 tok/s`** (`2.35 ms` TPOT)* | **`23.8×`** (*via [`megakernel-recipe/`](./megakernel-recipe/README.md)*) |
| **`C = 64` Output Throughput (`tok/s`)** | `858.3 tok/s` (`68.6 ms` TPOT) | `1,524.5 tok/s` (`38.5 ms` TPOT) | **`1,740.4 tok/s`** (`34.5 ms` TPOT) | **`+102.8%` (`+14.2%` vs. 2-Pass)** |
| **`C = 128` Output Throughput (`tok/s`)** | `2,017.6 tok/s` (`57.1 ms` TPOT) | `2,520.7 tok/s` (`45.2 ms` TPOT) | **`2,984.5 tok/s`** (`38.5 ms` TPOT) | **`+47.9%` (`+18.4%` vs. 2-Pass)** |
| **`C = 190` Single-Wave Saturation (`tok/s`)** | `3,164.1 tok/s` | `3,992.6 tok/s` | **`4,758.5 tok/s`** (`297.4 tok/s/chip`) | **`+50.4%` (`+19.2%` vs. 2-Pass)** |
| **`C = 512` Output Throughput (`tok/s`)** | `4,310.5 tok/s` (`94.8 ms` TPOT) | `4,310.5 tok/s` (`94.8 ms` TPOT) | **`4,310.5 tok/s`** (`94.8 ms` TPOT) | Prefill-chunk bound (`8,140 tok/s` decode) |
