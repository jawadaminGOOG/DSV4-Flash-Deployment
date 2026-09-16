# Patches that make DeepSeek-V4.1-Flash run on TPU

The stock serving image cannot run this model. Ten patches make it run: eight against
[tpu-inference](https://github.com/vllm-project/tpu-inference) and two against
[vLLM](https://github.com/vllm-project/vllm). Apply them, build one image, and the recipe in the
directory above serves the model.

Each patch states its own base commit, and every patch in this directory is verified to apply to
that base.

## Base commits

| Repository | Base commit | Date | Apply with |
|---|---|---|---|
| `vllm-project/tpu-inference` | `f22b5068d9326e7899cd2fc8afd4de79f36d20f4` | 2026-09-09 | `git am` |
| `vllm-project/vllm` | `9b959b86577c082c0b2bf9e2c22263255a36ad83` | 2026-09-10 | `git apply` |

The vLLM base is the commit that first added the V4.1 model definitions
(`[Model] DeepSeek-V4.1-Flash Model Definitions`).

CAUTION: vLLM renamed the package from `deepseek_v4_1` to `deepseek_v41` on 2026-09-14 in
`[Refactor] Normalize DeepSeek V4.1 model package`. The vLLM patches here name the old path. On a
newer checkout, apply them with `git apply --directory` or rename the path in the patch header
first.

## Apply them

```bash
./apply.sh /path/to/workdir
```

The script clones both repositories at the base commits, applies every patch in order, and stops
on the first failure. Read it before you run it; it is 30 lines.

To do it by hand:

```bash
git clone https://github.com/vllm-project/tpu-inference.git
git -C tpu-inference checkout f22b5068d9326e7899cd2fc8afd4de79f36d20f4
git -C tpu-inference am /path/to/patches/tpu-inference/*.patch

git clone https://github.com/vllm-project/vllm.git
git -C vllm checkout 9b959b86577c082c0b2bf9e2c22263255a36ad83
for p in /path/to/patches/vllm/*.patch; do git -C vllm apply "$p"; done
```

## What the tpu-inference patches do

`tpu-inference/`, 23 files, 4,181 insertions, 35 deletions.

### 0001 — Claim every DeepSeek V4 family model type for `deepseek_v4_fp8`

The quantisation config matched the model type `deepseek_v4` exactly, so `deepseek_v4_1` fell
through to the dense path and the checkpoint failed to load. The patch matches the whole family.
It ships a unit test.

### 0002 — Add the DeepSeek V4.1 backbone for TPU

The decoder: 40 layers, `hidden_size` 5120, 384 routed experts, `head_dim` 512, `rms_norm_eps`
1e-20. `mhc_torch.py` holds the torch-side helper the wrapper calls.

### 0003 — Add the DeepSeek V4.1 TPU attention, compressor and indexer

The three new kernels, and the cache-manager change that gives them their arrays.

- `deepseek_v41_attention.py` — the CSA2 attention path.
- `deepseek_v41_compressor.py` — writes the compressed key-value record.
- `deepseek_v41_indexer.py` — the sparse index that picks 512 of the cached positions.

`kv_cache_manager.py` gains a same-name `isinstance` fix. V4 and V4.1 each define a class called
`DeepseekV4IndexerCache`, so a check against the V4 class alone misses every V4.1 layer.

### 0004 — Mask padding rows out of the DeepSeek V4.1 cache writes

A padded batch row wrote into the cache at slot 0 and corrupted a real sequence. The patch masks
those rows. It ships a 295-line unit test for the compressor.

### 0005 — Give every V4.1 sliding-window layer its own KV cache array

This patch is larger than its title. It carries the whole memory and quantisation story, and it is
the one to read first if you care about performance:

| File | Change |
|---|---|
| `deepseek_v41_attention.py` | one KV array for each of the 40 sliding-window layers, instead of one shared array |
| `mxfp4.py` | the native MXFP4 path, and the compact `[E, num_blocks, N]` scale layout that saves 47.46 GiB |
| `megablox/gmm_v2.py` | the grouped matrix multiply reads MXFP4 blocks natively |
| `fused_moe_gmm.py`, `moe_weights.py` | route the mixture-of-experts weights through that path |
| `cleanup_sharding.py` | `_drop_host_resident` stops the sweep that pulled host tensors onto the device |
| `engram_host_lookup.py` | the Engram tables stay in host DRAM and the lookup reaches them |
| `vllm_model_wrapper.py` | the wrapper exposes the torch module the TPU path needs |

### 0006 — Trace the compressed cache bytes and the sliding-window handoff

Diagnostic logging behind a flag. Keep it: it is what found patch 0007.

### 0007 — Write the compressed RoPE record in the byte-plane layout the gather reads

**The patch that makes the model correct.** 13 lines.

`csa_gather` reads the high byte of channel `i` at offset `i`, and the low byte at offset
`64 + i`, then rebuilds the value as `(high << 8) | low`. The record is two byte planes. The
compressor wrote it as interleaved little-endian bf16. Every rotary value came back wrong, and
every build before this patch produced repeated-token output that looked like a sampler fault.

### 0008 — Test that the RoPE record decodes the way the gather decodes it

The regression test for 0007. It decodes the written record the same way the kernel does and
asserts the round trip. Without it, the next person to touch the compressor reintroduces the bug.

## What the vLLM patches do

`vllm/`, 2 files.

### 0001 — `engram.py`: add a non-CUDA reference path

Engram lookup was CUDA-only and failed with `Engram CPU offload requires UVA support`. The patch
adds `_lookup_reference`, `_fused_engram_post_wkv_reference` and `_hash_ids_reference`, and a
platform branch that selects them off CUDA.

### 0002 — `weight_utils.py`: serialise the RunAI streamer per host

Sixteen ranks on one node each started a RunAI streamer and each reserved its own host buffer, so
the node ran out of memory during load. The patch adds `_runai_host_lock`, a `flock` on
`/dev/shm/vllm_runai_streamer_host.lock`. `flock` is a kernel object, so the lock covers one node,
which is the scope that matters.

## How the recipe uses them

The recipe does not patch anything at container start. Build one image from the patched trees and
point the manifest at it:

```bash
# From the workdir that apply.sh created.
docker build -t <YOUR_REGISTRY>/tpu-inference:v41 -f tpu-inference/docker/Dockerfile .
docker push <YOUR_REGISTRY>/tpu-inference:v41
```

Then set that tag in `image:` in `../dsv41-flash-v6e16-serving.yaml`.

The manifest keeps one guard from the development setup: it imports the model registry before the
server loads weights, and it exits if `DeepseekV41ForCausalLM` is missing. Keep the guard. Without
it a wrong image fails about 20 minutes later, during weight load, with an unrelated error.

## What these patches do not do

- They do not remove the 16-way replication of the non-expert weights, which is 111.76 GiB of the
  378.78 GiB resident total. That needs a sharding change and it is the largest remaining lever.
- They do not raise `--max-num-batched-tokens` above 256. See the memory ceilings in the README
  one directory up.
- Patch 0006 is diagnostic logging, not a fix. Remove it if you do not want the log volume.
