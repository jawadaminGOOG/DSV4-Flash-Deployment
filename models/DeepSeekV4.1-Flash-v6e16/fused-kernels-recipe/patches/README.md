# Patches for the Single-Kernel Fused `W13+SiLU+W2` MoE & Hybrid `EP=8 × TP=2` Recipe (`0001..0026`)

This directory contains the complete, self-contained **28-patch stack** (**26 against [`tpu-inference`](https://github.com/vllm-project/tpu-inference)** `0001..0026` and **2 against [`vLLM`](https://github.com/vllm-project/vllm)** `0001..0002`) that enables the Single-Kernel Fused `W13+SiLU+W2` Pallas Megakernel (`gmm_v2_fused_w13_w2`) + Hybrid `EP=8 × TP=2` Routed Expert Sharding recipe (`commit 623b2904`).

## Base Commits & Patched Head

| Repository | Base Commit | Patched Head | Patch Count | Apply Command |
|---|---|---|---|---|
| `vllm-project/tpu-inference` | `f22b5068d9326e7899cd2fc8afd4de79f36d20f4` | `623b2904` | `26` (`0001..0026`) | `git am` |
| `vllm-project/vllm` | `9b959b86577c082c0b2bf9e2c22263255a36ad83` | Patched | `2` (`0001..0002`) | `git apply` |

```bash
./apply.sh /path/to/workdir
```

---

## Patch Breakdown

- **Foundations, Quantization & Full-Context (`16K`) Correctness (`0001`–`0018`):** Full DeepSeek-V4.1-Flash TPU model backbone, CSA2 compressor/indexer, native MXFP4 compact scales (`[E, num_blocks, N]`), host-pinned `203.1 GB` Engram tables, and `qnorm` removal (`rope_kernel.rope`).
- **XProf Kernel & Collective Optimizations (`0019`–`0025`):** SparseCore `nd_reduce_scatter` offload + shared-expert stream decoupling (`0019`), 4D/3D KV-cache `%bitcast` views + `mode="drop"` scatter (`0020`), and `gmm_v2` `(tile_m=128, bucket_base=128)` with 3-op bitwise IEEE-754 E2M1/E8M0 VMEM dequantization (`0021..0025`).
- **Hybrid `EP=8 × TP=2` + Single-Kernel Fused `W13+SiLU+W2` Pallas Megakernel ([`0026-perf-moe-hybrid-ep8-tp2-and-fused-w13-w2-pallas-megakernel.patch`](tpu-inference/0026-perf-moe-hybrid-ep8-tp2-and-fused-w13-w2-pallas-megakernel.patch)):**
  - **`tpu_inference/layers/common/process_weights/moe_weights.py`:** Adds `_hybrid_ep_tp_swap_local` and `_apply_hybrid_ep_tp_shard_map` (`TPU_HYBRID_EP8_TP2_MOE=1`), pairing adjacent TPU ranks (`(0,1), (2,3), ..., (14,15)`) via pairwise ICI `ppermute` so each 2-chip TP group holds `48` half-width experts (`w13` width `2304`, `w2` contracting dim `1152 = 9 × 128` with zero HBM padding), cutting cross-rank routing straggler wait by **`-37.0%` (`11.066 -> 6.967 ms/step`)**.
  - **`tpu_inference/kernels/megablox/gmm_v2.py`:** Adds `gmm_v2_fused_w13_w2` (`TPU_MEGABLOX_FUSED_W13_W2=1`), fusing `w13` (`[128, 5120] @ [5120, 2304]`), gated `silu_and_mul_with_clamp` (`-> [128, 1152]` `bf16` in VMEM registers), and `w2` (`[128, 1152] @ [1152, 5120]`) into a single Pallas kernel per MoE layer (`23.35 / 30.72 MiB` VMEM), cutting kernel dispatches from `80` to `40` per step and eliminating intermediate HBM round-trips.
  - **`tpu_inference/layers/common/fused_moe_gmm.py`:** Wires `gmm_v2_fused_w13_w2` and Hybrid `EP=8 × TP=2` into the batched `fused_moe_func` execution path.
