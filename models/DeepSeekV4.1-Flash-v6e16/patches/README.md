# Patches That Enable DeepSeek-V4.1-Flash on TPU v6e

The stock serving image cannot run `deepseek-ai/DeepSeek-V4.1-Flash`. Twenty patches enable full-context (`16,384`-token) serving, accuracy, and throughput on TPU v6e-16: **eighteen against [tpu-inference](https://github.com/vllm-project/tpu-inference)** (`0001..0018`) and **two against [vLLM](https://github.com/vllm-project/vllm)** (`0001..0002`).

Every patch states its base commit and is verified to apply cleanly. Running `apply.sh` against fresh clones of `tpu-inference` (`f22b5068`) and `vllm` (`9b959b86`) produces a tree identical (`git diff` empty) to commit `d85ce9c1` on which the 3-workload concurrency sweeps, the 198-question GPQA Diamond evaluation, and the XProf profiles were measured.

## Base Commits

| Repository | Base Commit | Date | Patched Head | Apply Command |
|---|---|---|---|---|
| `vllm-project/tpu-inference` | `f22b5068d9326e7899cd2fc8afd4de79f36d20f4` | 2026-09-09 | `d85ce9c1` (`18` patches) | `git am` |
| `vllm-project/vllm` | `9b959b86577c082c0b2bf9e2c22263255a36ad83` | 2026-09-10 | `2` patches | `git apply` |

## Apply Them

```bash
./apply.sh /path/to/workdir
```

Or manually:

```bash
git clone https://github.com/vllm-project/tpu-inference.git
git -C tpu-inference checkout f22b5068d9326e7899cd2fc8afd4de79f36d20f4
git -C tpu-inference am /path/to/patches/tpu-inference/*.patch

git clone https://github.com/vllm-project/vllm.git
git -C vllm checkout 9b959b86577c082c0b2bf9e2c22263255a36ad83
for p in /path/to/patches/vllm/*.patch; do git -C vllm apply "$p"; done
```

---

## What the 18 `tpu-inference` Patches Do

### Foundations, Quantization, Backbone & Memory Layout (`0001`–`0008`)

- **[`0001-Quantization-Claim-every-DeepSeek-V4-family-model-ty.patch`](tpu-inference/0001-Quantization-Claim-every-DeepSeek-V4-family-model-ty.patch)** — Widens `VllmDeepseekV4Fp8Config.override_quantization_method` so `deepseek_v41` and `deepseek_v41_text` route through `deepseek_v4_fp8` instead of falling through to unquantized dense linear layers. Includes a 10-case unit test.
- **[`0002-Add-the-DeepSeek-V4.1-backbone-for-TPU.patch`](tpu-inference/0002-Add-the-DeepSeek-V4.1-backbone-for-TPU.patch)** — Adds `tpu_inference/models/vllm/experimental/deepseek_v41.py`: 40-layer decoder stack (`hidden_size=5120`, `384` routed experts, `head_dim=512`, `rms_norm_eps=1e-20`, 4-stream `mHC` hyper-connections).
- **[`0003-Add-the-DeepSeek-V4.1-TPU-attention-compressor-and-i.patch`](tpu-inference/0003-Add-the-DeepSeek-V4.1-TPU-attention-compressor-and-i.patch)** — Adds `deepseek_v41_attention.py` (CSA2 sparse MLA + SWA), `deepseek_v41_compressor.py` (KV pooling across 4 compressor groups), `deepseek_v41_indexer.py` (top-512 index selection), and `kv_cache_manager.py` support for 55 KV/state arrays across 51 logical layers.
- **[`0004-Mask-padding-rows-out-of-the-DeepSeek-V4.1-cache-wri.patch`](tpu-inference/0004-Mask-padding-rows-out-of-the-DeepSeek-V4.1-cache-wri.patch)** — Masks padded batch rows out of KV-cache scatter writes so short batches cannot overwrite slot 0. Includes a 295-line compressor unit test.
- **[`0005-Give-every-V4.1-sliding-window-layer-its-own-KV-cach.patch`](tpu-inference/0005-Give-every-V4.1-sliding-window-layer-its-own-KV-cach.patch)** — Gives each of the 40 sliding-window attention layers its own KV array; adds native MXFP4 weight-only `gmm_v2` VMEM dequantization (`e2m1` + `e8m0` scale in compact `[E, num_blocks, N]` layout, saving `47.46 GiB` of HBM across the 16-chip slice); and adds `engram_host_lookup.py` + `_drop_host_resident` so the `94.42 GiB/rank` Engram embedding tables remain pinned in host DRAM.
- **[`0006-Trace-the-compressed-cache-bytes-and-the-sliding-win.patch`](tpu-inference/0006-Trace-the-compressed-cache-bytes-and-the-sliding-win.patch)** — Optional env-gated diagnostic tracing for compressed KV cache bytes and SWA handoff.
- **[`0007-Write-the-compressed-RoPE-record-in-the-byte-plane-l.patch`](tpu-inference/0007-Write-the-compressed-RoPE-record-in-the-byte-plane-l.patch)** — Writes the compressed RoPE record as two split byte planes (`high` byte at offset `i`, `low` byte at offset `64 + i`) matching the layout `csa_gather` decodes (`(high << 8) | low`).
- **[`0008-Test-that-the-RoPE-record-decodes-the-way-the-gather.patch`](tpu-inference/0008-Test-that-the-RoPE-record-decodes-the-way-the-gather.patch)** — Round-trip regression test verifying the compressor byte-plane write against `csa_gather` decoding.

### Full-Context (`16K`) Correctness, Causal Bounds & Precision Fixes (`0009`–`0018`)

- **[`0009`](tpu-inference/0009-Trace-the-RMS-and-the-zero-fraction-beside-the-layer.patch)–[`0012`](tpu-inference/0012-Make-the-mHC-and-unfilled-parameter-reports-readable.patch)** — Layer-wise RMS/zero-fraction diagnostics and load-time verification of `mHC` stream-mixing weights and parameter initialization.
- **[`0013-Fix-the-indexer-causal-bound-for-compressed-KV-state.patch`](tpu-inference/0013-Fix-the-indexer-causal-bound-for-compressed-KV-state.patch)** — Fixes the `streamindex_topk.py` Pallas causal mask bound from inclusive `k_span <= q_pos // compression_ratio` to the exact exclusive bound `k_span < (q_pos + 1) // compression_ratio` matching the DeepSeek reference (`model.py:561-565`), preventing odd query positions (`q_pos = 64`) from attending to partially-formed compressed KV slots.
- **[`0014-Implement-short-context-indexer-bypass-matching-refe.patch`](tpu-inference/0014-Implement-short-context-indexer-bypass-matching-refe.patch)** — Aligns the short-context indexer bypass threshold (`max_seq_len < compressed_topk * compress_ratio`, i.e., `1,024` tokens for ratio-2 layers) with the reference vLLM implementation.
- **[`0015-Import-lax-and-sort-streamindex_topk-outputs-chronol.patch`](tpu-inference/0015-Import-lax-and-sort-streamindex_topk-outputs-chronol.patch)** — Sorts `streamindex_topk` selected KV indices in chronological order before `csa_gather`.
- **[`0016-core_attention-retain-float32-accumulator-precision-.patch`](tpu-inference/0016-core_attention-retain-float32-accumulator-precision-.patch)** — Retains `float32` softmax/output accumulator precision across the SWA and SparseMLA merge boundary instead of truncating intermediate attention outputs to `bfloat16` before log-sum-exp rescaling.
- **[`0017-Fix-mla_swa-page-aligned-_start_offset-and-deduplica.patch`](tpu-inference/0017-Fix-mla_swa-page-aligned-_start_offset-and-deduplica.patch)** — Fixes `mla_swa.py` page-aligned `_start_offset` indexing and deduplicates redundant KV-cache DMA writes.
- **[`0018-fix-dsv41-remove-unweighted-per-head-RMSNorm-qnorm-f.patch`](tpu-inference/0018-fix-dsv41-remove-unweighted-per-head-RMSNorm-qnorm-f.patch)** — **Primary full-context accuracy fix.** DeepSeek-V4.0 applied an unweighted per-head `RMSNorm` (`qnorm`) to queries after `wq_b`, whereas DeepSeek-V4.1 removed `qnorm` (`apply_q_norm = False` in `deepseek-inference/model.py:772` and `vllm/deepseek_v4_1/attention.py:881`). Calling `rope_kernel.qnorm_rope` forced every query head to unit RMS (`||q_h||_2 = sqrt(512)`), distorting `(q · k) / sqrt(512)` relative to the learned `attn_sink` denominator (`exp(attn_sink - m_curr)`) as context grew past 62 tokens. Switching `deepseek_v41_attention.py:387` to `rope_kernel.rope` restores 100% long-context recall (`55/55` diagnostic/needle checks) and `94.8%` completed-chain GPQA Diamond Pass@1 (`0/197` repetition loops).

---

## What the 2 `vLLM` Patches Do

- **[`0001-engram-add-a-non-CUDA-reference-path.patch`](vllm/0001-engram-add-a-non-CUDA-reference-path.patch)** — Adds platform-agnostic `_lookup_reference`, `_fused_engram_post_wkv_reference`, and `_hash_ids_reference` in `engram.py` so CPU-offloaded Engram tables run on TPU without CUDA UVA calls.
- **[`0002-weight_utils-serialise-the-runai-streamer-per-host.patch`](vllm/0002-weight_utils-serialise-the-runai-streamer-per-host.patch)** — Adds a per-node `/dev/shm/vllm_runai_streamer_host.lock` (`flock`) in `weight_utils.py` so multi-rank TPU workers on the same host do not exhaust host DRAM during concurrent GCS weight streaming.
