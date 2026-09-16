# DeepSeek-V4.1-Flash on TPU v6e-16

A vLLM serving recipe and a balanced-workload concurrency sweep for
`deepseek-ai/DeepSeek-V4.1-Flash` on one 16-chip **TPU v6e** slice: 4 × `ct6e-standard-4t` in a
`4x4` optical torus, on GKE. Measured 2026-09-12.

DeepSeek-V4.1-Flash had no TPU path before this work. The backbone, the CSA2 attention, the
compressor and the indexer are new, and the compressed rotary record needed a byte-layout fix
before the model produced correct tokens at all. The kernel section below gives the detail.

## Highlights

- **5,137.30 output tok/s** at concurrency 512 on the balanced `1k/1k` workload — **321.1 tok/s
  per chip**. The curve is still climbing at 512 streams.
- **Decode is near parity with DeepSeek-V4-Flash on the same slice.** The server reports
  **8,140 tok/s** of steady-state decode at concurrency 512, against 8,463.70 tok/s measured end
  to end for V4-Flash. The end-to-end gap is prefill, and the prefill batch size is 8× smaller on
  this arm for a memory reason given below.
- **Zero failed requests.** 1,028 requests across ten concurrency levels, from 1 to 512.
- **Greedy decoding is deterministic.** 40 prompts, three passes at concurrency 1, 40/40
  byte-identical. 100 synthetic key-value-slot prompts at concurrency 16, 100/100 identical.
- **The weights need 378.78 GiB of the 499.84 GiB slice.** 111.76 GiB of that is 16-way
  replication of every non-expert weight. That number sets every other limit in this recipe.

## Configuration

| Item | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4.1-Flash` (552B total, 16B active, 384 routed experts) |
| Hardware | 1 × TPU v6e-16, `4x4` topology, 4 × `ct6e-standard-4t` on GKE |
| Serving stack | vLLM on the `tpu-inference` JAX backend, multi-host over Ray |
| Parallelism | TP=16, expert parallel, 16-way DP attention |
| MoE quantisation | MXFP4 weights, native four-bit kernel path |
| Dense quantisation | INT8 |
| KV cache | `fp8` |
| Context length | 2,048 |
| Prefill batch | `--max-num-batched-tokens 256` |
| Sequence ceiling | 256 |
| Memory cap | `--gpu-memory-utilization 0.80` |

The full manifest is [`dsv41-flash-v6e16-serving.yaml`](dsv41-flash-v6e16-serving.yaml). It stages
the model code through a `PYTHONPATH` overlay, built from a ConfigMap before the server starts. The
serving command it runs is:

```bash
vllm serve "${MODEL_URI}" \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --load-format=runai_streamer \
  --trust-remote-code \
  --kv-cache-dtype=fp8 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --additional_config '{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' \
  --gpu-memory-utilization 0.80 \
  --no-enable-prefix-caching \
  --max-model-len 2048 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 256 \
  --host 0.0.0.0 --port 8000
```

## The kernel and backend work this recipe needs

The model runs on a new TPU backend, about 4,200 lines across 23 files. No file of the
DeepSeek-V4 path changes, so the V4 recipe in
[`models/DeepSeekV4-Flash-v6e16/`](../DeepSeekV4-Flash-v6e16) still runs unmodified.

### 1. The compressed rotary record is two byte planes, not interleaved bf16

**This is the fix that made the model correct.** Every build before it produced repeated-token
output, the signature of a `NaN` in the forward pass.

`csa_gather` reads the **high** byte of channel `i` at offset `i`, and the **low** byte of channel
`i` at offset `64 + i`, then rebuilds the value as `(high << 8) | low`. The record is **two byte
planes**. The writer emitted interleaved little-endian `bf16`, which the gather then decoded as a
different number in every channel. Nothing downstream of the rotary record can be right until the
two agree.

A regression test now decodes a written record exactly the way the gather decodes it, so a later
refactor cannot reintroduce the mismatch silently.

### 2. The new attention path

| Component | What it does |
|---|---|
| Backbone | The V4.1 decoder stack for TPU: 40 layers, `hidden_size` 5120, `head_dim` 512, `rms_norm_eps` 1e-20 |
| CSA2 attention | Compressed sparse attention with `index_topk` 512, over the compressed rotary record above |
| Compressor | Per-layer key-value pooling, with the gate present only where the compression ratio exceeds 1 |
| Indexer | The top-k selection that feeds CSA2 |
| Engram host lookup | Host-resident lookup for the two Engram layers, ids 1 and 14 |

### 3. Key-value cache layout

V4.1 mixes four cache kinds in one model: multi-head latent attention, compressed sparse
attention with a companion rotary array, sliding-window attention with a 128-token window, and
three state arrays. The runner now builds **55 arrays for 51 layers**, and gives every
sliding-window layer its own array rather than sharing one. Padding rows are masked out of the
cache writes, so a short batch cannot corrupt a neighbouring slot.

### 4. Quantisation

- `deepseek_v4_fp8` claims every DeepSeek-V4-family model type, so V4.1 reaches the same quantised
  path without a separate config class.
- The MXFP4 expert scale is `uint8`. A broadcast axis of 1 in the second-minor position makes the
  slice reserve four bytes for each byte, so 15.82 GiB of scale would hold 63.28 GiB. The stack
  keeps the scale as `[E, num_blocks, N]` and the kernel adds the broadcast axis back in VMEM.
  This single layout change is worth **47.46 GiB** across the slice.

## Three memory ceilings worth knowing before you retune

The weights take 378.78 GiB of the 499.84 GiB slice. Everything below follows from that.

- **`--gpu-memory-utilization` cannot go above 0.80 on this build, and it has a hard floor just
  under 0.76.** Each 0.01 step is only 0.31 GiB per chip. The cap counts the weights and the
  key-value pool, and it does not count the scratch the backbone compile needs:

  | Value | Result |
  |---|---|
  | 0.93 | hard out-of-memory |
  | 0.85 | compile asked 608.77 MiB more than a chip holds |
  | 0.82 | program load asked 1.49 GiB contiguous, 1.36 GiB reservable |
  | 0.81 | started, then failed on the first request at 1.40 GiB contiguous |
  | **0.80** | **serves** |

- **`--max-num-batched-tokens` cannot go above 256.** With data-parallel attention the runner
  builds its token buckets up to `max-num-batched-tokens × dp_size`, so the value is multiplied by
  16. At 1024 the top bucket is 16,384 tokens and the backbone compile asks for 393 MiB more than a
  chip holds. At 512 the runtime could not find 1.49 GiB of contiguous HBM for the step program.
  **This is the single largest performance limit in the recipe**, and it is why the end-to-end
  comparison below is not a like-for-like one.

- **111.76 GiB of the resident weights is 16-way replication of every non-expert weight.** Removing
  it is the largest available lever and it needs a sharding change, not a flag.

### One node-disk ceiling as well

A cold compile of this model takes about 87 minutes, so the persistent JAX compile cache is worth
keeping. The cache grows to about 42 GiB for each model and it never prunes itself.

CAUTION: do not run a second model from the same node pool with a second `hostPath` cache. Two
caches on one node exhaust the ephemeral storage. The kubelet then evicts every serving pod with
`The node was low on resource: ephemeral-storage`, and the Job ends with `BackoffLimitExceeded`.
The eviction looks like a model failure in the pod log, and it is not one.

Point the cache at a bucket to remove the ceiling. Delete the `jaxcache` volume and its mount, then
set `JAX_COMPILATION_CACHE_DIR` to `gs://<YOUR_BUCKET>/jaxcache`. The image already holds `gcsfs`,
`etils` and `tensorstore`, and the service account already reads the bucket. All four nodes then
share one cache, so only one node pays for each compile.

## Benchmark results (concurrency sweep 1 → 512)

Balanced `1k/1k`, measured with a streaming client running inside the serving pod, at
`temperature 0` with `ignore_eos`. Every level from 1 to 512 is measured, not interpolated.

![V4.1 against V4, 1k/1k](results/charts/v41-vs-v4-1k1k.png)

| Concurrency | Output tok/s | tok/s per chip | TTFT p50 | TTFT p90 | TPOT | Success |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.93 | 1.18 | 1,130 ms | 12,000 ms | 49.12 ms | 4/4 |
| 2 | 39.49 | 2.47 | 1,125 ms | 1,126 ms | 49.62 ms | 4/4 |
| 4 | 78.70 | 4.92 | 1,093 ms | 1,093 ms | 49.79 ms | 4/4 |
| 8 | 154.42 | 9.65 | 1,039 ms | 1,040 ms | 50.84 ms | 8/8 |
| 16 | 444.52 | 27.78 | 1,112 ms | 1,112 ms | 34.94 ms | 16/16 |
| 32 | 821.58 | 51.35 | 2,468 ms | 2,469 ms | 36.95 ms | 32/32 |
| 64 | 1,601.35 | 100.08 | 3,764 ms | 5,298 ms | 36.50 ms | 64/64 |
| 128 | 2,778.66 | 173.67 | 6,312 ms | 9,758 ms | 40.03 ms | 128/128 |
| 256 | 4,096.94 | 256.06 | 10,691 ms | 17,317 ms | 51.39 ms | 256/256 |
| 512 | **5,137.30** | **321.08** | 19,811 ms | 34,300 ms | 77.31 ms | 512/512 |

**The 12,000 ms TTFT at concurrency 1 is the sampler compile on the first request.** The same
request shape at concurrency 2 returns its first token in 1,126 ms.

### Against DeepSeek-V4-Flash on the same slice

| Concurrency | V4.1 tok/s | V4 tok/s | V4.1 / V4 | V4.1 TPOT | V4 TPOT |
|---:|---:|---:|---:|---:|---:|
| 1 | 18.9 | 35.3 | 0.54 | 49.1 ms | 27.7 ms |
| 2 | 39.5 | 70.7 | 0.56 | 49.6 ms | 27.8 ms |
| 4 | 78.7 | 140.5 | 0.56 | 49.8 ms | 28.0 ms |
| 8 | 154.4 | 277.5 | 0.56 | 50.8 ms | 28.4 ms |
| 16 | 444.5 | 598.6 | 0.74 | 34.9 ms | 26.3 ms |
| 32 | 821.6 | 1,125.4 | 0.73 | 37.0 ms | 26.9 ms |
| 64 | 1,601.4 | 2,089.5 | 0.77 | 36.5 ms | 28.8 ms |
| 128 | 2,778.7 | 4,113.0 | 0.68 | 40.0 ms | 28.6 ms |
| 256 | 4,096.9 | 6,671.9 | 0.61 | 51.4 ms | 33.7 ms |
| 512 | **5,137.3** | 8,463.7 | 0.61 | 77.3 ms | 50.2 ms |

**CAUTION: read this table with its caveat, or it will mislead you.** The two arms are not matched
on prefill batching. V4.1 ran `--max-num-batched-tokens 256`; V4-Flash ran 2,048. That is an 8×
difference and it works against V4.1 at every point. It is not a tuning choice: 256 is the largest
value that runs inside the V4.1 memory budget, and the table above the sweep shows what each larger
value did.

**Decode is near parity; the gap is prefill.** At concurrency 512 the V4.1 server's own meter
reports **8,140 tok/s of steady-state decode**, against 8,463.70 tok/s measured end to end at the
client for V4-Flash. The client figure includes prefill, and the time-to-first-token column is
where the difference lives: the V4.1 p90 is about double the V4 p90 at every level above 16.

V4.1 also carries more work per token. It activates 16B parameters during decode against
DeepSeek-V4-Flash's 13B, and its resident weights are 1.92× larger.

The V4-Flash figures come from
[`models/DeepSeekV4-Flash-v6e16/`](../DeepSeekV4-Flash-v6e16), measured on this same slice with
this same client.

## Correctness

| Check | Result |
|---|---|
| Greedy determinism, concurrency 1, 40 prompts, three passes | **40/40 byte-identical** |
| Greedy determinism, concurrency 16, synthetic slot prompts | **100/100 identical** |
| Greedy determinism, concurrency 16, prose prompts | 85/100 identical |
| Empty completions | 0 |
| Request errors | 0 |

The concurrency-1 result is the noise floor, and it shows the kernels themselves are deterministic.

The synthetic prompts test key-value addressing directly. A model can only continue a pseudo-random
word cycle if attention reads the right cache rows, and a wrong slot breaks the cycle visibly.
100/100 at concurrency 16 is the clearest available confirmation that the cache path is correct.

The 15 prose differences at concurrency 16 come from continuous batching. vLLM composes a different
batch on each pass, which changes the reduction order inside a kernel, and a 128-token prose
continuation amplifies a last-bit difference into a different word. This is why
`greedy_determinism.py` separates a gating comparison, two runs at the same concurrency, from an
informational comparison at a different concurrency.

**Use the chat endpoint, not the raw completion endpoint.** This is a reasoning model and it needs
its own chat template. `/v1/completions` gives base-model continuation, which loops at temperature 0
even when the model is healthy.

## What this recipe does not show

1. **Only the `1k/1k` shape is measured.** The `8k/1k` and `1k/8k` shapes need a 9,216-token
   context, which this arm does not serve.
2. **No published benchmark score.** GPQA Diamond needs a long chain of thought, and a 2,048-token
   context truncates every question in the set.
3. **No cross-backend agreement test.** This project has no NVIDIA GPU, so the strongest
   correctness check — the same prompts on the TPU port and on the merged NVIDIA path, compared
   token for token — could not run.

## Files

| File | Purpose |
|---|---|
| [`dsv41-flash-v6e16-serving.yaml`](dsv41-flash-v6e16-serving.yaml) | Multi-host serving Job, headless Service, and the model-code overlay as a ConfigMap |
| [`scripts/benchmark_sweep.py`](scripts/benchmark_sweep.py) | The sweep client: three shapes, ten concurrency levels, one JSON row per point |
| [`scripts/greedy_determinism.py`](scripts/greedy_determinism.py) | Greedy determinism over 200 prompts, half prose and half synthetic slot prompts |
| [`scripts/smoke.py`](scripts/smoke.py) | Four known-answer questions on the chat endpoint, run first after any restart |
| [`scripts/gpqa_diamond.py`](scripts/gpqa_diamond.py) | GPQA Diamond Pass@1, with a separate truncation count |
| [`scripts/plot_sweep.gp`](scripts/plot_sweep.gp) | Builds the chart from the table in its own header |
| [`results/raw/1k1k.json`](results/raw/1k1k.json) | Every measured point, unprocessed |

## Reproduce

```bash
# 1. Deploy. Substitute your registry, bucket and service account first.
kubectl apply -f dsv41-flash-v6e16-serving.yaml
kubectl logs -f job/dsv41-v6e16 | grep "Application startup complete"

# 2. Check correctness before measuring anything.
POD=$(kubectl get pods -l app=dsv41-v6e16 -o name | head -1 | cut -d/ -f2)
kubectl cp scripts/smoke.py "${POD}:/tmp/smoke.py"
kubectl exec "${POD}" -- python3 /tmp/smoke.py

# 3. Run the balanced sweep from inside the serving pod.
kubectl cp scripts/benchmark_sweep.py "${POD}:/tmp/benchmark_sweep.py"
kubectl exec "${POD}" -- python3 /tmp/benchmark_sweep.py 1k1k /tmp/1k1k.json
kubectl cp "${POD}:/tmp/1k1k.json" results/raw/1k1k.json

# 4. Rebuild the chart.
gnuplot -e "outfile='results/charts/v41-vs-v4-1k1k.png'" scripts/plot_sweep.gp
```

A cold JAX compile takes about 87 minutes. The cache is a `hostPath` at
`/var/tmp/dsv41-jax-cache`, so it survives pod replacement but not node replacement.

**CAUTION: watch the node disk.** The compile cache grows to about 42 GiB for each model. Two
models on one node pass the ephemeral-storage threshold, and GKE then evicts every serving pod with
`The node was low on resource: ephemeral-storage`. Prune the cache, or point
`JAX_COMPILATION_CACHE_DIR` at a bucket.

## Related

- [`models/DeepSeekV4-Flash-v6e16/`](../DeepSeekV4-Flash-v6e16) — the DeepSeek-V4-Flash recipe and
  the full three-shape sweep this comparison uses as its baseline.
- The repository [`README.md`](../../README.md) covers the hardware reference, the cost model, and
  the serving architecture in more depth.
