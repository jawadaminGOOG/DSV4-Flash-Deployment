# DeepSeek-V4-Flash on Google Cloud TPU v6e (Trillium)

High-throughput deployment and benchmark recipes for **DeepSeek-V4-Flash** (284B total / 13B active parameters, 256 MoE experts) on **Google Cloud TPU v6e**.

---

## 1. Executive Summary

DeepSeek-V4-Flash uses a 256-expert Mixture-of-Experts (MoE) architecture with Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA).

This repository configures **16-way Data-Parallel (DP) Attention**, **Collective Matmul V2 overlap**, **INT8 MoE quantization**, and **FP8 KV caching**. A single **16-chip TPU v6e slice (4x4 optical torus)** achieves:

* **7,065.0 output tokens/sec** at concurrency $C=512$ ($441.6\text{ tok/s per chip}$).
* **14,130.1 aggregate tokens/sec** (prompt + generation).
* **100% request success rate** across all concurrency tiers with exact token validation.

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
| **16** | 32 / 32 | **507.0 tok/s** | 31.7 tok/s | **1,014.1 tok/s** | 1,429.1 ms | 1,536.4 ms | **57.12 ms** | 17.5 tok/s |
| **32** | 32 / 32 | **957.6 tok/s** | 59.8 tok/s | **1,915.1 tok/s** | 2,465.5 ms | 2,608.3 ms | **52.56 ms** | 19.0 tok/s |
| **64** | 64 / 64 | **1,811.9 tok/s** | 113.2 tok/s | **3,623.8 tok/s** | 3,089.2 ms | 3,252.0 ms | **55.25 ms** | 18.1 tok/s |
| **128** | 128 / 128 | **3,355.0 tok/s** | 209.7 tok/s | **6,710.1 tok/s** | 4,124.5 ms | 5,259.5 ms | **62.95 ms** | 15.9 tok/s |
| **256** | 256 / 256 | **5,504.4 tok/s** | 344.0 tok/s | **11,008.9 tok/s** | 7,023.0 ms | 9,813.7 ms | **70.00 ms** | 14.3 tok/s |
| **512** | 512 / 512 | **7,065.0 tok/s** | **441.6 tok/s** | **14,130.1 tok/s** | **11,885.6 ms** | **21,681.1 ms** | **103.15 ms** | **9.7 tok/s** |

---

### 2.2 Direct Comparison vs. NVIDIA GH200

| Metric | Reference Baseline (4x NVIDIA GH200) | Google Cloud TPU v6e (16 Chips) |
| :--- | :--- | :--- |
| **Hardware** | 4x NVIDIA GH200 (480GB HBM3e, 900 GB/s NVLink) | 16x Google TPU v6e (32GB HBM2e, 4x4 Optical Torus) |
| **Quantization** | FP8 MoE + FP8 KV Cache | INT8 MoE + FP8 KV Cache |
| **Workload** | $\text{ISL}=1024, \text{OSL}=1024, \text{ignore\_eos}=\text{True}$ | $\text{ISL}=1024, \text{OSL}=1024, \text{ignore\_eos}=\text{True}$ |
| **Peak Output Throughput** | **17,634.0 tok/s** | **7,065.0 tok/s** |
| **Aggregate Throughput** | ~35,268.0 tok/s | **14,130.1 tok/s** |
| **Time Per Output Token (TPOT)**| ~28.0 ms (estimated) | **103.15 ms** |
| **Request Success Rate** | 100% | **100% (512 / 512 completed)** |

---

## 3. Cost & Economic Efficiency

Comparison of output token throughput per dollar across Google Cloud pricing:

| Hardware Platform | Hourly Cost | Peak Output tok/s | Cost per 1M Output Tokens | Output Tokens per $1.00 |
| :--- | :---: | :---: | :---: | :---: |
| **TPU v6e-16 (3-Year Commitment)** | **$19.52 / hr** | **7,065.0** | **$0.77** | **1,302,000** |
| **TPU v6e-16 (Spot / Flex-Start)** | **$21.60 / hr** | **7,065.0** | **$0.85** | **1,177,000** |
| **TPU v6e-16 (On-Demand List)** | **$43.20 / hr** | **7,065.0** | **$1.70** | **589,000** |
| **4x NVIDIA H200 (GCP On-Demand)** | $42.40 / hr | ~14,000.0 (est) | **$0.84** | **1,189,000** |

### Key Takeaways
* **Competitive Economics**: On spot or 3-year commitment tiers ($19.52 – $21.60/hr for 16 chips), TPU v6e serves DeepSeek-V4-Flash at **$0.77 – $0.85 per 1M output tokens** (>1.17M – 1.30M tokens/$).
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
* **Headroom caveat**: That margin is thin, and most of it is spent on one allocation. At startup, sampling pre-compilation builds a dummy logits tensor of shape `(max_num_seqs × dp_size, vocab_size)` in float32 — at `--max-num-seqs 512`, 16-way DP and a 129,280-token vocabulary, that is `(8192, 129280)`, or **3.95 GiB**, materialized as a single contiguous buffer before it is sharded. Roughly 600 MB of headroom remains after it. Under the C=512 tier the TPU runtime consequently hits `RESOURCE_EXHAUSTED` on individual program loads and recovers via `ExecutePrepareWithOomRetries` (defragment and retry). No request failed as a result, but the retries stall the step and contribute to the gap between steady-state decode rate and end-to-end throughput.
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
├── README.md                          # Project documentation and benchmark report
└── recipes/
    ├── dsv4-flash-v6e16.yaml          # Kubernetes Job and Service for TPU v6e-16
    ├── benchmark.py                   # Async streaming benchmark suite
    └── publish_to_ubench.py           # Publishes results.json to the UBench BigQuery backend
```

---

## 8. License

This repository is available under the Apache 2.0 License.
