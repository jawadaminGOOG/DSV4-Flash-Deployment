# DeepSeek-V4-Flash on TPU v6e-16

A vLLM serving recipe and a full concurrency sweep for `deepseek-ai/DeepSeek-V4-Flash` on one
16-chip **TPU v6e** slice: 4 × `ct6e-standard-4t` in a `4x4` optical torus, on GKE. Measured
2026-09-02.

## Highlights

- **8,463.70 output tok/s** at concurrency 512 on the balanced `1k/1k` workload — **529.0 tok/s per
  chip**. The curve is still climbing at 512 streams.
- **8,067.90 output tok/s** on the `1k/8k` reasoning workload, at **40.32 ms** per output token.
  Long generations are the strongest shape on this slice, not the weakest.
- Time per output token drifts from **27.69 ms** at a single stream, to **28.60 ms** at 128 streams,
  to **50.19 ms** at 512 — about **19.9 tok/s per user** with the slice fully loaded.
- **Zero failed requests.** 3,079 requests across 30 measured points, at every concurrency level
  from 1 to 512, on all three workloads.
- **One workload is slow and it is a known configuration limit, not a model limit.** The `8k/1k`
  prefill-heavy shape reaches only 1,758.53 tok/s, because `--max-num-batched-tokens` is pinned at
  2048 and an 8,192-token prompt therefore takes four chunked prefill steps.

## Configuration

| Item | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Flash` (284B total, 13B active, 256 experts) |
| Hardware | 1 × TPU v6e-16, `4x4` topology, 4 × `ct6e-standard-4t` on GKE |
| Serving stack | vLLM on the `tpu-inference` JAX backend, multi-host over Ray |
| Parallelism | TP=16, expert parallel, 16-way DP attention |
| MoE quantisation | INT8 weights, block size 512 |
| KV cache | `fp8` |
| Collective overlap | `--xla_tpu_all_gather_collective_matmul_mode=post_spmd` and the matching reduce-scatter mode |
| Context length | 2,048 for the balanced shape, 9,216 for the two 8k shapes |
| Sequence ceiling | 512 at 2,048 context, 256 at 9,216 context. See the SMEM limit below |

The full manifest is [`dsv4-flash-v6e16-serving.yaml`](dsv4-flash-v6e16-serving.yaml). It carries the
source patch as a ConfigMap and applies it before the server starts. The serving command it runs is:

```bash
vllm serve "${MODEL_URI}" \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash \
  --load-format=runai_streamer \
  --trust-remote-code \
  --kv-cache-dtype=fp8 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --additional_config '{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 512 \
  --host 0.0.0.0 --port 8000
```

### The patch this recipe applies

The ConfigMap holds one patch against `tpu_inference`, with two independent changes:

1. **The compressed-attention metadata is derived once per forward pass, not once per layer.** The
   derivation is memoised on the KV-cache group, through a weak reference so a finished trace's
   context is released. Builds per pass fall from about 60 to 3.
2. **The MoE combine runs on the SparseCore instead of the TensorCore.** The combine falls from
   8.340 to 1.449 ms per step, and the whole decode step falls from 39.660 to 34.502 ms.

Both are measured on this slice with the profile in the same session. Greedy output stays
byte-identical to the unpatched build.

### Three memory ceilings worth knowing before you retune

- `--max-num-seqs` sizes the sampling pre-compilation buffer, `(max_num_seqs × 16, 129280)` in
  fp32. At 512 that is a **3.95 GiB** transient allocation on one chip.
- `--max-num-batched-tokens` sizes the compile ladder. **2048 is the largest value that compiles on
  this slice.** This is what limits the prefill-heavy shape.
- The **block table is prefetched into a 1 MiB SMEM** at 4 bytes per 16-token block, and it shares
  that 1 MiB with about 0.11 KiB of scheduler arrays per sequence. `max_num_seqs 512` at 9,216
  context needs 1,179,648 bytes for the table alone and fails to start. Measured on this slice:

  | `--max-num-seqs` at 9,216 context | block table | scheduler arrays | total of 1.00M | result |
  |---:|---:|---:|---:|---|
  | 256 | 576.0K | 27.1K | 603.1K | serves |
  | 384 | 864.0K | 42.3K | 906.3K | serves |
  | 432 | 972.0K | 56.5K | 1028.6K | fails by 4.6K |

  The practical ceiling at 9,216 context is near `--max-num-seqs 416`. This recipe uses 256, which
  is the value the whole sweep was measured on.

## Benchmark results (concurrency sweep 1 → 512)

Three workload patterns, measured with a streaming client running inside the serving pod, at
`temperature 0` with `ignore_eos`. Every level from 1 to 512 is measured, not interpolated.

| Pattern | Peak output tok/s | @ conc | Req/s | TTFT mean @ 512 | TPOT @ 512 |
|---|---:|---:|---:|---:|---:|
| `1k/1k` (balanced) | **8,463.70** | 512 | 8.27 | 10.03 s | 50.19 ms |
| `8k/1k` (prefill-heavy) | 1,758.53 | 512 | 1.72 | 132.31 s | 52.56 ms |
| `1k/8k` (reasoning) | 8,067.90 | 512 | 0.98 | 10.42 s | 40.32 ms |

![Output throughput vs concurrency](results/charts/throughput_vs_concurrency.png)

Full per-concurrency tables, the TTFT and TPOT charts, and the caveats that belong beside each row
are in [`results/benchmark_sweep_report.md`](results/benchmark_sweep_report.md).

**Two serving arms, not one.** The SMEM ceiling above forbids `max_num_seqs 512` at a 9,216-token
context, so the `1k/1k` rows come from the 2048/512 arm and the two 8k rows come from the 9216/256
arm. Do not read the three rows as one system state.

**The reasoning shape is limited by the KV pool, not by the sequence cap.** A `1k/8k` request holds
only its 1,024-token prompt at admission, so the scheduler admits far more than 256 of them, and
each one then grows. The pool holds about 313 sequences at the full 9,216-token length. Raising the
cap to 384 measured 7,992.80 tok/s at C=512, inside the reproducibility band of the 8,067.90 above.

## Files

| File | Purpose |
|---|---|
| [`dsv4-flash-v6e16-serving.yaml`](dsv4-flash-v6e16-serving.yaml) | Multi-host serving Job, headless Service, and the source patch as a ConfigMap |
| [`scripts/benchmark_sweep.py`](scripts/benchmark_sweep.py) | The sweep client: three shapes, ten concurrency levels, one JSON row per point |
| [`scripts/report.py`](scripts/report.py) | Builds the report and the three charts from the raw JSON |
| [`results/benchmark_sweep_report.md`](results/benchmark_sweep_report.md) | The full write-up: configuration, caveats, three per-concurrency tables |
| [`results/charts/`](results/charts) | Throughput, TTFT and TPOT against concurrency |
| [`results/raw/`](results/raw) | Every measured point, unprocessed |

## Reproduce

```bash
# 1. Deploy the serving arm. Substitute your registry, bucket and service account first.
kubectl apply -f dsv4-flash-v6e16-serving.yaml
kubectl logs -f job/dsv4-v6e16 | grep "Application startup complete"

# 2. Run the balanced sweep from inside the serving pod.
POD=$(kubectl get pods -l app=dsv4-v6e16 -o name | head -1 | cut -d/ -f2)
kubectl cp scripts/benchmark_sweep.py "${POD}:/tmp/benchmark_sweep.py"
kubectl exec "${POD}" -- python3 /tmp/benchmark_sweep.py 1k1k /tmp/1k1k.json
kubectl cp "${POD}:/tmp/1k1k.json" results/raw/1k1k.json

# 3. For the two 8k shapes, redeploy with the long-context arm, then repeat step 2
#    with the shape name 8k1k or 1k8k.
sed -e 's/--max-model-len 2048/--max-model-len 9216/' \
    -e 's/--max-num-seqs 512/--max-num-seqs 256/' \
    dsv4-flash-v6e16-serving.yaml > /tmp/longctx.yaml
kubectl delete job dsv4-v6e16 --wait=true && kubectl apply -f /tmp/longctx.yaml

# 4. Rebuild the report and the charts.
python3 scripts/report.py results/raw results
```

The long-context arm compiles for about 41 minutes on a cold JAX cache. The 2,048-token arm takes
about 10 minutes.

## Related

- The repository [`README.md`](../../README.md) covers the hardware reference, the cost model, and
  the serving architecture in more depth.
- [`recipes/`](../../recipes) holds the original single-shape recipe and its benchmark client.
