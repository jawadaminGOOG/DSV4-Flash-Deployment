# DeepSeek-V4.1-Flash on TPU v6e-16: XProf Kernel & Collective Optimizations Recipe (`61.71 ms -> 37.33 ms/step`)

A self-contained full-context (`16,384`-token) vLLM/XLA serving recipe, 27-patch stack (`25` `tpu-inference` + `2` `vLLM` patches, commit `4b8edd8e`), 198-question GPQA Diamond accuracy evaluation (`94.8%` completed-chain Pass@1), three-workload concurrency sweep (`1k/1k`, `8k/1k`, `1k/8k` across `C=1..512`), and XProf hardware profiling technical note ([`TECH-NOTE.md`](TECH-NOTE.md)) for **DeepSeek-V4.1-Flash (`552B` total / `16B` active, `384` routed experts)** on a single **16-chip TPU v6e slice (`4×4` optical torus)**.

---

## 1. Key Optimizations & Measured Impact

1. **SparseCore `nd_reduce_scatter` Offload & Shared-Expert Stream Decoupling ([`Patch 0019`](patches/tpu-inference/0019-perf-moe-decouple-shared_experts.down_proj-from-post.patch)):**
   Offloads the post-MoE 2D torus `reduce-scatter` to TPU v6e's SparseCore DMA engines (`SC Overlay`) and decouples `shared_experts.down_proj` via `jax.lax.optimization_barrier`, cutting decode step wall by **`-5.76 ms/step` (`61.71 -> 55.95 ms/step`)** and prefill step wall by **`-25.12 ms/step` (`270.65 -> 245.53 ms/step`)**.
2. **Minor-Dimension-Preserving 3D KV-Cache `%bitcast` Views & `mode="drop"` Scatter ([`Patch 0020`](patches/tpu-inference/0020-perf-dsv41-preserve-4D-KV-cache-minor-tile-dimension.patch)):**
   Preserves the `(4, 128)` (`nope_cache`) and `(4, 256)` (`indexer` `cache`) TPU minor tile dimensions via 3D views (`(-1, 4, 128)` / `(-1, 4, 256)`) and replaces dummy-row `concatenate` + `[:-1]` pad copies with hardware `mode="drop"` scatters, eliminating **`99.99%` of KV-cache HBM relayout copies (`2.5385 -> 0.0003 ms/step`)**.
3. **`gmm_v2` `(tile_m=128, bucket_base=128)` & 3-Instruction Bitwise IEEE-754 MXFP4 Unpacking ([`Patches 0021..0025`](patches/README.md)):**
   Replaces the 12-instruction `decode_e2m1`/`decode_e8m0` VPU sequence in `tpu_inference/kernels/megablox/gmm_v2.py` with a 3-instruction bitwise IEEE-754 mantissa/exponent assembly (`max_abs_diff = 0.0`) and sets decode tiling to `(tile_m=128, bucket_base=128)`, cutting single-layer MoE time by **`-42.1%` (`3,361.9 -> 1,945.5 µs`)** and lowering median decode step wall to **`37.33 ms/step` (`-39.5%`)**.

---

## 2. Directory Contents

| Path | Description |
|---|---|
| **[`README.md`](README.md)** | Overview and quick-start guide for the XProf Kernel & Collective Optimizations recipe. |
| **[`TECH-NOTE.md`](TECH-NOTE.md)** | Hardware profiling & kernel optimization deep dive with XProf waterfall breakdowns and Mermaid diagrams. |
| **[`dsv41-flash-v6e16-serving.yaml`](dsv41-flash-v6e16-serving.yaml)** | Ready-to-apply GKE Job & Service manifest (`TP=16`, `EP=16`, `attn_dp=16`, SparseCore collective offload). |
| **[`patches/`](patches/README.md)** | Complete 27-patch stack (`25` `tpu-inference` patches `0001..0025` + `2` `vLLM` patches + [`apply.sh`](patches/apply.sh), commit `4b8edd8e`). |
| **[`results/`](results/)** | Full 10-level concurrency sweep raw JSONs ([`1k1k.json`](results/raw/1k1k.json), [`8k1k.json`](results/raw/8k1k.json), [`1k8k.json`](results/raw/1k8k.json)), 198-question GPQA Diamond evaluation ([`gpqa_diamond_v41.json`](results/accuracy/gpqa_diamond_v41.json)), and charts in [`results/charts/`](results/charts/). |
| **[`scripts/`](scripts/)** | GPQA Diamond evaluator ([`eval_gpqa_diamond.py`](scripts/eval_gpqa_diamond.py)), isolated `gmm_v2` TPU v6e hardware microbenchmark ([`microbench_gmm_v2_decode.py`](scripts/microbench_gmm_v2_decode.py)), and chart generator ([`generate_charts.py`](scripts/generate_charts.py)). |

---

## 3. Reproducing the Patched Image & Deployment

```bash
./patches/apply.sh /tmp/dsv41-xprof-build
kubectl apply -f dsv41-flash-v6e16-serving.yaml
```
