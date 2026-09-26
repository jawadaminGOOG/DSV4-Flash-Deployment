# Patches for the 40-Layer Pallas Decode Megakernel + `DSpark` (`mtp.0..2`) Speculative Decoding Recipe

This directory contains the self-contained **28-patch stack** (**26 against [`tpu-inference`](https://github.com/vllm-project/tpu-inference)** `0001..0026` and **2 against [`vLLM`](https://github.com/vllm-project/vllm)** `0001..0002`) that enables the DeepSeek-V4.1-Flash TPU v6e-16 backbone (`0001..0025`) plus the **40-Layer Pallas Decode Megakernel + `DSpark` (`mtp.0..2`) Speculative Decoding Engine** (`0026`).

## Apply Command

```bash
./apply.sh /path/to/workdir
```

## Patch Summary

- **`tpu-inference/0001..0025` + `vllm/0001..0002`:** Full DeepSeek-V4.1-Flash TPU v6e-16 model backbone, quantization, `16K` accuracy fixes, and XProf kernel optimizations.
- **[`tpu-inference/0026-feat-megakernel-40-layer-pallas-decode-and-dspark.patch`](tpu-inference/0026-feat-megakernel-40-layer-pallas-decode-and-dspark.patch):** Adds the standalone `megakernel/` package (`config.py`, `load.py`, `engine_jax.py`, `decode_megakernel.py`, `dspark.py`, `collectives16.py`, `pool_alias.py`) implementing the 40-layer single-`pallas_call` decode megakernel (`8.49 ms` TPOT at `C=1`, `5.28×` faster than `vLLM`) and `DSpark` (`mtp.0..2`) lossless `1+4` speculative decoding engine (`5.57 ms` median / `4.50 ms` best effective TPOT at `C=1`, `3.48 tok/step` mean acceptance).
