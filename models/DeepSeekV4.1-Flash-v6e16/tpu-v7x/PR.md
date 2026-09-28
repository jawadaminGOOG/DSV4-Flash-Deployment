# DeepSeek-V4.1-Flash Pallas Megakernel & DSpark on GKE TPU v7x (`2x2x4` & `2x2x1`)

## Summary

Adds a complete, forensically peer-reviewed Pallas megakernel (`tpu_custom_call == 1`) and DSpark speculative decoding implementation (`deepseek_v41/`) and GKE TPU v7x deployment recipe (`models/DeepSeekV4.1-Flash-v6e16/tpu-v7x/`) for the real **DeepSeek-V4.1-Flash** checkpoint (`40` MoE layers + `3` DSpark draft layers, `510.3 GB` across `48` safetensors shards) on **TPU v7x (`TPU7x`)** across both **`2x2x4` (`32` TensorCores / `16` chips)** and **`2x2x1` (`8` TensorCores / `4` chips)**.

### Key Components
- **Single-`pl.pallas_call` (`tpu_custom_call == 1`) 40-Layer In-VMEM Decode Megakernel (`deepseek_v41/decode_megakernel.py`):**
  - Fuses all 40 backbone layers (`@pl.loop(0, 40)`, `35` MLA-CSA `r=2` + `5` full MLA `r=1`), multi-head hyper-connections (`mHC`, `n_streams=4`, 20-step Sinkhorn-Knopp), 32-head Hadamard Indexer (`topk=512`), `mla_fp4` (`e2m1` + `e8m0`) KV cache, Engram 2/3-gram prime-XOR hash lookup (layers 1 & 12), FP8 shared expert, MXFP4 routed experts (`192` total, top-6 sigmoid routing), final RMSNorm, and sharded LM head inside a **single `pl.pallas_call` (`tpu_custom_call == 1`)**.
  - **Dynamic Active-Expert HBM→VMEM Streaming (`EP32×TP1` / `EP8×TP1`):** Each TensorCore owns `6` (`tp32`) or `24` (`tp8`) whole routed experts in HBM and streams only the locally active experts (`<= 6` for `B=1`) into VMEM per layer via `pltpu.sync_copy`.
  - **VMEM Pool Aliasing (`pool_alias.py`):** Compresses per-core VMEM from `671.55 MiB` unaliased to **`23.31 MiB` total VMEM** on `2x2x4` (`< 48 MiB` soft budget).
  - **In-Kernel Barrier & Butterfly All-Reduce (`collectives32.py`, `deepseek_v41/collectives.py`):** Zero-host-sync remote VMEM DMAs across 32 TensorCores (`2x2x4`) or 8 TensorCores (`2x2x1`).
- **DSpark Speculative Decoding & Vectorized `q = 6` Block Verification (`deepseek_v41/dspark.py`):**
  - 3-layer DSpark drafter (`layers 40..42`, `max_draft_tokens = 5`) paired with a vectorized single-`pallas_call` target verification kernel (`_megakernel_block_in_pallas`, `q = 6`).
- **Pre-Sharder, Fast Parallel Loader & Resumable GKE Spot Runner (`deepseek_v41/preshard.py`, `load.py`, `runner.py`, `server.py`):**
  - Pre-shards the 510.3 GB checkpoint once into per-rank directories so all 32 TensorCores load in **`64.86 s`** on `2x2x4` (`35.21 s` on `2x2x1`, unlinking `/dev/shm` staging files per rank immediately after HBM placement), with per-stage GCS checkpointing for seamless spot preemption recovery.

---

## Measured Results (`TPU7x` In-VMEM Megakernel vs Public 8× B200 & XLA Control)

| Metric | **TPU v7x In-VMEM Megakernel (`tpu_custom_call == 1`)** | Public 8× B200 (vLLM / SGLang) | TPU v7x XLA Control | Evidence Artifact |
|---|---|---|---|---|
| **Cold Startup (GCS → HBM)** | **`64.86 s`** (`2x2x4`), **`35.21 s`** (`2x2x1`) (`<= 180 s` target) | — | — | `../../models/DeepSeekV4.1-Flash-v6e16/tpu-v7x/results/v7x16/cold_startup.json` |
| **Total Aliased VMEM / Core** | **`23.31 MiB`** (`2x2x4`, `671.55 MiB` unaliased); **`38.20 MiB`** (`2x2x1`) | — | — | `../../models/DeepSeekV4.1-Flash-v6e16/tpu-v7x/results/v7x16/cold_startup.json` |
| **Reference Parity (`sinkhorn=20`)** | **`96.88%` top-1**, **`100.0%` top-5**, `0.9962` cos (`2x2x1`: `95.31%` top-1, `100.0%` top-5) | — | — | `../../models/DeepSeekV4.1-Flash-v6e16/tpu-v7x/results/v7x16/parity_ref.json` |
| **Negative Control (`sinkhorn=0`)** | **`0.00%` top-1** (`< 50%` threshold) | — | — | `../../models/DeepSeekV4.1-Flash-v6e16/tpu-v7x/results/v7x16/parity_neg_control.json` |
| **Decode `B=1` Median (`tpu_custom_call == 1`)** | **`3.577 ms` / `279.55 tok/s`** (`p90=3.617 ms`; `2x2x1`: `4.179 ms` / `239.29 tok/s`) | `7.87 ms` / `127.0 tok/s` (**`2.20×` faster** on `2x2x4`, **`1.88×`** on `2x2x1`) | `213.16 ms` / `4.69 tok/s` (**`59.59×` faster**) | `../../models/DeepSeekV4.1-Flash-v6e16/tpu-v7x/results/v7x16/perf_b1.json` |
| **Decode `B=2` Median (`tpu_custom_call == 1`)** | **`3.789 ms` / `527.86 tok/s`** (`p90=3.822 ms`; `2x2x1`: `4.908 ms` / `407.53 tok/s`) | `8.81 ms` / `227.0 tok/s` (**`2.33×` faster** on `2x2x4`, **`1.80×`** on `2x2x1`) | `286.90 ms` / `6.97 tok/s` (**`75.72×` faster**) | `../../models/DeepSeekV4.1-Flash-v6e16/tpu-v7x/results/v7x16/perf_b2.json` |
| **Decode `B=4` Median (`tpu_custom_call == 1`)** | **`4.018 ms` / `995.60 tok/s`** (`p90=4.065 ms`; `2x2x1`: `5.640 ms` / `709.21 tok/s`) | `10.72 ms` / `373.0 tok/s` (**`2.67×` faster** on `2x2x4`, **`1.90×`** on `2x2x1`) | `287.98 ms` / `13.89 tok/s` (**`71.68×` faster**) | `../../models/DeepSeekV4.1-Flash-v6e16/tpu-v7x/results/v7x16/perf_b4.json` |
| **Decode `B=8` Median (`tpu_custom_call == 1`)** | **`4.691 ms` / `1,705.33 tok/s`** (`p90=4.738 ms`; `2x2x1`: `7.829 ms` / `1,021.88 tok/s`) | `12.58 ms` / `636.0 tok/s` (**`2.68×` faster** on `2x2x4`, **`1.61×`** on `2x2x1`) | `301.49 ms` / `26.53 tok/s` (**`64.27×` faster**) | `../../models/DeepSeekV4.1-Flash-v6e16/tpu-v7x/results/v7x16/perf_b8.json` |
| **DSpark Verify (`q=6` vs `q=1`)** | **`4/4` exact token match (`100.0%`)**, **`1.575×` speedup** (`2.73–3.99` tok/step → **`440.3 – 1,115.4 tok/s`** effective) | `363.0 – 452.0 tok/s` (MTP/EAGLE3) | — | `../../models/DeepSeekV4.1-Flash-v6e16/tpu-v7x/results/v7x16/dspark_verify.json` |
| **GSM8K-100 (DSpark OFF)** | **`100 / 100` (`100.0%` EM)** (`2x2x1`: **`99 / 100 = 99.0%` EM**) | — | — | `../../models/DeepSeekV4.1-Flash-v6e16/tpu-v7x/results/v7x16/gsm8k_dspark_off.json` |
| **GSM8K-100 (DSpark ON)** | **`99 / 100` (`99.0%` EM)** (`3.99` emitted tok/step; `2x2x1`: `99.0%`, `4.00` tok/step) | — | — | `../../models/DeepSeekV4.1-Flash-v6e16/tpu-v7x/results/v7x16/gsm8k_dspark_on.json` |
| **GPQA-Diamond (DSpark ON)** | **`36 / 50` (`72.0%` EM)** (`3.60` emitted tok/step, `24,167` generated tokens) | — | — | `../../models/DeepSeekV4.1-Flash-v6e16/tpu-v7x/results/v7x16/gpqa_diamond.json` |
