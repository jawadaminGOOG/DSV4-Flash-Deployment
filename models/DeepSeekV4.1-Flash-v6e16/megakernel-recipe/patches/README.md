# Patches for the 51-Layer Persistent Pallas Megakernel + `DSpark` (`1+7`) Speculative Decoding Recipe

This directory contains the self-contained **28-patch stack** (**26 against [`tpu-inference`](https://github.com/vllm-project/tpu-inference)** `0001..0026` and **2 against [`vLLM`](https://github.com/vllm-project/vllm)** `0001..0002`) that enables the DeepSeek-V4.1-Flash TPU v6e-16 backbone (`0001..0025`) plus the **51-Layer Persistent Pallas Decode Megakernel + `DSpark` (`1+7`) Speculative Decoding Engine** (`0026`).

## Apply Command

```bash
./apply.sh /path/to/workdir
```

## Patch Summary

- **`tpu-inference/0001..0025` + `vllm/0001..0002`:** Full DeepSeek-V4.1-Flash TPU v6e-16 model backbone, quantization, `16K` accuracy fixes, and XProf kernel optimizations.
- **[`tpu-inference/0026-feat-megakernel-persistent-51-layer-pallas-decode-and-dspark.patch`](tpu-inference/0026-feat-megakernel-persistent-51-layer-pallas-decode-and-dspark.patch):** Adds the standalone `megakernel/` package (`decode_megakernel.py`, `collectives16.py`, `dspark.py`, `engram_prologue.py`, `pool_alias.py`) implementing the 51-layer persistent VMEM Pallas Megakernel (`12.99 / 16.00 MiB` VMEM) and `DSpark` (`1+7`) speculative decoding verifier (`425.5 accepted output tok/s/req`, `2.35 ms/token` at `C=1`).
