# DeepSeek-V4-Flash on Google Cloud TPU v6e (Trillium)

High-throughput deployment and benchmark recipes for **DeepSeek-V4-Flash** (284B total / 13B active parameters, 256 MoE experts) on **Google Cloud TPU v6e**.

---

## 1. Executive Summary

DeepSeek-V4-Flash uses a 256-expert Mixture-of-Experts (MoE) architecture with Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA).

This repository configures **16-way Data-Parallel (DP) Attention**, **Collective Matmul V2 overlap**, **INT8 MoE quantization**, and **FP8 KV caching**. A single **16-chip TPU v6e slice (4x4 optical torus)** achieves:

* **8,463.7 output tokens/sec** at concurrency $C=512$ ($529.0\text{ tok/s per chip}$) on the balanced 1k/1k workload.
* **8,067.9 output tokens/sec** on the 1k/8k reasoning workload, at **40.32 ms** per output token.
* **100% request success rate** — 3,079 requests across 30 measured points, from $C=1$ to $C=512$.

Those figures come from the patched serving arm. The full three-shape sweep, its charts, its raw
data and the manifest that produced it are in
[`models/DeepSeekV4-Flash-v6e16/`](./models/DeepSeekV4-Flash-v6e16).

---

## 2. Benchmark Results

### 2.1 Concurrency Scaling Sweep (TPU v6e-16)

Workload parameters:
* **Input Sequence Length (ISL)**: 1,024 prompt tokens
* **Output Sequence Length (OSL)**: 1,024 generated tokens
* **Serving Mode**: Streaming Server-Sent Events (SSE), `temperature=0.0`, `ignore_eos=True`, burst arrival (`rate=inf`)
* **Hardware**: 16x TPU v6e chips (4 nodes $\times$ 4 chips, 4x4 2D Torus optical interconnect)

| Concurrency ($C$) | Total Requests | Output Throughput | Per-Chip Output | Aggregate Throughput | TTFT P50 | TTFT P90 | TPOT (Mean) | Stream Speed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **16** | 16 / 16 | **598.6 tok/s** | 37.4 tok/s | **1,195.8 tok/s** | 500.2 ms | 500.5 ms | **26.26 ms** | 38.1 tok/s |
| **32** | 32 / 32 | **1,125.4 tok/s** | 70.3 tok/s | **2,248.0 tok/s** | 1,643.9 ms | 1,644.6 ms | **26.85 ms** | 37.2 tok/s |
| **64** | 64 / 64 | **2,089.5 tok/s** | 130.6 tok/s | **4,173.9 tok/s** | 1,710.0 ms | 2,612.7 ms | **28.83 ms** | 34.7 tok/s |
| **128** | 128 / 128 | **4,113.0 tok/s** | 257.1 tok/s | **8,215.9 tok/s** | 2,983.6 ms | 3,869.2 ms | **28.60 ms** | 35.0 tok/s |
| **256** | 256 / 256 | **6,671.9 tok/s** | 417.0 tok/s | **13,327.5 tok/s** | 4,861.9 ms | 7,511.0 ms | **33.74 ms** | 29.6 tok/s |
| **512** | 512 / 512 | **8,463.7 tok/s** | **529.0 tok/s** | **16,906.7 tok/s** | **9,650.2 ms** | **18,745.5 ms** | **50.19 ms** | **19.9 tok/s** |

Measured on the patched serving arm described in
[`models/DeepSeekV4-Flash-v6e16/`](./models/DeepSeekV4-Flash-v6e16). The same client on the
unpatched build reached **7,065.0 tok/s** at $C=512$ with a **103.15 ms** TPOT. Two source changes
account for the difference: the compressed-attention metadata is now derived once per forward pass
instead of once per layer, and the MoE combine runs on the SparseCore instead of the TensorCore.

### 2.1.1 The other two workload shapes

The same sweep covers a prefill-heavy and a reasoning shape, at every concurrency level from 1 to
512:

| Pattern | Peak output tok/s | @ conc | Req/s | TTFT mean @ 512 | TPOT @ 512 |
|---|---:|---:|---:|---:|---:|
| `1k/1k` (balanced) | **8,463.70** | 512 | 8.27 | 10.03 s | 50.19 ms |
| `8k/1k` (prefill-heavy) | 1,758.53 | 512 | 1.72 | 132.31 s | 52.56 ms |
| `1k/8k` (reasoning) | 8,067.90 | 512 | 0.98 | 10.42 s | 40.32 ms |

The prefill-heavy shape is limited by `--max-num-batched-tokens 2048`, which is the largest value
that compiles on this slice, so an 8,192-token prompt takes four chunked prefill steps. Full tables,
charts and the caveats that belong beside each row are in
[`models/DeepSeekV4-Flash-v6e16/results/benchmark_sweep_report.md`](./models/DeepSeekV4-Flash-v6e16/results/benchmark_sweep_report.md).

---

### 2.2 Direct Comparison vs. NVIDIA GH200

| Metric | Reference Baseline (4x NVIDIA GH200) | Google Cloud TPU v6e (16 Chips) |
| :--- | :--- | :--- |
| **Hardware** | 4x NVIDIA GH200 (480GB HBM3e, 900 GB/s NVLink) | 16x Google TPU v6e (32GB HBM2e, 4x4 Optical Torus) |
| **Quantization** | FP8 MoE + FP8 KV Cache | INT8 MoE + FP8 KV Cache |
| **Workload** | $\text{ISL}=1024, \text{OSL}=1024, \text{ignore\_eos}=\text{True}$ | $\text{ISL}=1024, \text{OSL}=1024, \text{ignore\_eos}=\text{True}$ |
| **Peak Output Throughput** | **17,634.0 tok/s** | **8,463.7 tok/s** |
| **Aggregate Throughput** | ~35,268.0 tok/s | **16,906.7 tok/s** |
| **Time Per Output Token (TPOT)**| ~28.0 ms (estimated) | **50.19 ms** |
| **Request Success Rate** | 100% | **100% (512 / 512 completed)** |

---

## 3. Cost & Economic Efficiency

Comparison of output token throughput per dollar across Google Cloud pricing:

| Hardware Platform | Hourly Cost | Peak Output tok/s | Cost per 1M Output Tokens | Output Tokens per $1.00 |
| :--- | :---: | :---: | :---: | :---: |
| **TPU v6e-16 (3-Year Commitment)** | **$19.52 / hr** | **8,463.7** | **$0.64** | **1,561,000** |
| **TPU v6e-16 (Spot / Flex-Start)** | **$21.60 / hr** | **8,463.7** | **$0.71** | **1,411,000** |
| **TPU v6e-16 (On-Demand List)** | **$43.20 / hr** | **8,463.7** | **$1.42** | **705,000** |
| **4x NVIDIA H200 (GCP On-Demand)** | $42.40 / hr | ~14,000.0 (est) | **$0.84** | **1,189,000** |

### Key Takeaways
* **Competitive Economics**: On spot or 3-year commitment tiers ($19.52 – $21.60/hr for 16 chips), TPU v6e serves DeepSeek-V4-Flash at **$0.64 – $0.71 per 1M output tokens** (1.41M – 1.56M tokens/$).
* **Integrated Interconnect**: The native 2D optical torus (ICI) interconnect removes external InfiniBand network switch costs.

---

## 4. Serving Architecture & Optimizations

### 4.1 16-way Data-Parallel (DP) Attention
Standard Tensor Parallelism requires all-reduce communication during every decode step.
With `enable_dp_attention: true`:
* Attention layers shard request sequences across the 16 TPU chips independently (`mesh(attn_dp=16)`).
* Each chip computes attention locally without cross-chip communication during decode steps.

### 4.2 Collective Matmul V2 Overlapping
MoE token routing requires all-gather and reduce-scatter collective communications.
With the following flag:
```bash
LIBTPU_INIT_ARGS="--xla_tpu_all_gather_collective_matmul_mode=post_spmd --xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd"
```
The XLA compiler overlaps inter-chip communication directly with Matrix Multiply Unit (MXU) computation.

### 4.3 Memory Allocation
* **INT8 MoE Weights**: The 256 routed experts occupy **23.65 GB per chip** on 16 chips.
* **FP8 KV Cache**: FP8 format reduces KV cache memory consumption by 50% compared to BF16.
* **Memory Limits**: The configuration `--max-model-len 2048`, `--max-num-seqs 512`, and `--gpu-memory-utilization 0.85` reserves **26.68 GB** of static state, leaving roughly **4.56 GB of free HBM headroom** per chip.
* **Headroom caveat**: That margin is thin, and a single startup allocation consumes most of it transiently. Sampling pre-compilation builds a dummy logits tensor of shape `(max_num_seqs × dp_size, vocab_size)` in float32 — at `--max-num-seqs 512`, 16-way DP attention and a 129,280-token vocabulary, that is `(8192, 129280)`, or **3.95 GiB** — as a single contiguous buffer on one chip, and only then shards it (to 253 MiB per chip). The buffer is released before serving, so it is a peak rather than a resident cost, but it leaves only ~600 MB of slack at the moment it exists.
* Separately, under the C=512 tier the TPU runtime does hit `RESOURCE_EXHAUSTED` on individual program loads and recovers via `ExecutePrepareWithOomRetries` (defragment and retry). No request failed as a result, but the retries stall the step and contribute to the gap between steady-state decode rate and end-to-end throughput. Lower `--gpu-memory-utilization` if you need a larger margin.
* **`max_num_seqs × max_model_len` must stay at or below 4,194,304.** The attention block table is prefetched into **SMEM**, which is **1 MiB per core**, at 4 bytes per 16-token block. At `--max-num-seqs 512` and `--max-model-len 9216` the engine asks for `512 × (9216 / 16) × 4 = 1,179,648` bytes and fails to start with `RESOURCE_EXHAUSTED ... space=smem`. Raise the context and you must lower the sequence ceiling to match. This limit is independent of HBM and it is the one that binds as context grows.
* **`--max-num-seqs 512` is a ceiling, not a preference.** That buffer scales linearly with the flag, so `--max-num-seqs 1024` asks for 7.89 GiB and the engine fails to start with `RuntimeBufferAllocationFailure`. Separately, the KV cache holds `81,427 tokens` per chip — 1,302,832 across the slice, or **636 concurrent sequences at the full 2,048-token context** — so values much above ~640 could not be filled at this context length in any case.

---

## 5. Hardware Reference: TPU v6e (Trillium)

| Property | Value | Notes |
| :--- | :--- | :--- |
| **HBM Capacity** | 32 GB per chip (31.24 GB addressable) | 512 GB total for 16 chips |
| **Memory Bandwidth** | 1,638 GB/s per chip | 26.2 TB/s aggregate for 16 chips |
| **Compute Peak (INT8)** | 1,836 TOPS per chip | 29.37 POPS aggregate for 16 chips |
| **Compute Peak (BF16)** | 918 TFLOPS per chip | 14.68 PFLOPS aggregate for 16 chips |
| **Interconnect (ICI)** | 800 Gbps bidirectional per chip | Direct optical links forming 2D torus |
| **Topology** | 4x4 Torus (4 nodes $\times$ 4 chips) | Multi-host Ray cluster over VPC |

---

## 6. Instructions for Deployment and Benchmarking

Find all deployment manifests and benchmark scripts in the [`recipes/`](./recipes) directory:
* [`recipes/dsv4-flash-v6e16.yaml`](./recipes/dsv4-flash-v6e16.yaml): Kubernetes headless service and multi-host Job.
* [`recipes/benchmark.py`](./recipes/benchmark.py): Async streaming benchmark script.

### 6.1 Prerequisites
1. Create a Google Kubernetes Engine (GKE) cluster with a 16-chip TPU v6e node pool:
   ```bash
   gcloud container node-pools create dsv4-v6e16-pool \
     --cluster=<YOUR_CLUSTER_NAME> \
     --zone=<YOUR_ZONE> \
     --node-locations=<YOUR_ZONE> \
     --machine-type=ct6e-standard-4t \
     --num-nodes=4 \
     --tpu-topology=4x4
   ```
2. Store DeepSeek-V4-Flash model weights in a Google Cloud Storage bucket (for example, `gs://<YOUR_GCS_BUCKET>/deepseek-v4-flash`).
3. Prepare a container image that contains JAX/XLA, vLLM, and TPU dependencies.
4. Grant the pods read access to the bucket through Workload Identity. The manifest creates the
   Kubernetes service account `dsv4-sa`; bind it to a Google service account that can read the
   checkpoint:
   ```bash
   gcloud storage buckets add-iam-policy-binding gs://<YOUR_GCS_BUCKET> \
     --member="serviceAccount:<YOUR_GOOGLE_SERVICE_ACCOUNT>" \
     --role=roles/storage.objectViewer

   gcloud iam service-accounts add-iam-policy-binding <YOUR_GOOGLE_SERVICE_ACCOUNT> \
     --role=roles/iam.workloadIdentityUser \
     --member="serviceAccount:<YOUR_PROJECT>.svc.id.goog[default/dsv4-sa]"
   ```
   Then replace `<YOUR_GOOGLE_SERVICE_ACCOUNT>` in the `ServiceAccount` annotation in
   [`recipes/dsv4-flash-v6e16.yaml`](./recipes/dsv4-flash-v6e16.yaml).

---

### 6.2 Deploy the Serving Cluster

1. Update the image and storage bucket in [`recipes/dsv4-flash-v6e16.yaml`](./recipes/dsv4-flash-v6e16.yaml):
   * Replace `<YOUR_CONTAINER_REGISTRY>` with your container registry URI.
   * Replace `<YOUR_GCS_BUCKET>` with your GCS bucket path.

2. Apply the Kubernetes manifest:
   ```bash
   kubectl apply -f recipes/dsv4-flash-v6e16.yaml
   ```

3. Monitor pod status:
   ```bash
   kubectl get pods -l app=dsv4-v6e16 -o wide
   kubectl logs dsv4-v6e16-0 --tail=50 -f
   ```

   > **Expect a long first start.** On a cold node pool the engine spends roughly **46 minutes** in
   > `init engine (profile, create kv cache, warmup model)`, almost all of it XLA compilation,
   > before `Application startup complete`. Size any liveness or readiness probe accordingly.
   > `JAX_COMPILATION_CACHE_DIR` is set to `/jaxcache` so subsequent starts on the same node reuse
   > the compiled artifacts: with that cache warm the same step takes about **6.5 minutes**
   > (`init engine ... took 390.13 s (compilation: 372.57 s)`). The cache lives on a `hostPath`, so
   > it survives pod restarts but not node replacement.

4. Verify server readiness:
   ```bash
   kubectl exec dsv4-v6e16-0 -- curl -s http://127.0.0.1:8000/v1/models
   ```

---

### 6.3 Run the Benchmark Suite

1. Forward port 8000 to your local machine:
   ```bash
   kubectl port-forward dsv4-v6e16-0 8000:8000
   ```

2. Run the concurrency sweep script:
   ```bash
   python3 recipes/benchmark.py \
     --host 127.0.0.1 \
     --port 8000 \
     --model deepseek-ai/DeepSeek-V4-Flash \
     --isl 1024 \
     --osl 1024 \
     --concurrencies 16 32 64 128 256 512 \
     --num-chips 16 \
     --output-json results.json
   ```

---

## 7. Directory Structure

```
├── README.md                              # Project documentation and benchmark report
├── recipes/
│   ├── dsv4-flash-v6e16.yaml              # Kubernetes Job and Service for TPU v6e-16
│   ├── benchmark.py                       # Async streaming benchmark suite
│   └── publish_to_ubench.py               # Publishes results.json to the UBench BigQuery backend
└── models/
    └── DeepSeekV4-Flash-v6e16/
        ├── README.md                      # Serving recipe and the full concurrency sweep
        ├── dsv4-flash-v6e16-serving.yaml  # Patched serving Job, Service and patch ConfigMap
        ├── scripts/                       # Sweep client and report generator
        └── results/                       # Report, PNG charts and every raw measurement
```

---

## 8. License

This repository is available under the Apache 2.0 License.
