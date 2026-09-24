#!/usr/bin/env bash
# End-to-end verification, XProf capture, and concurrency benchmark suite for
# DeepSeek-V4.1-Flash with Fused W13+SiLU+W2 Pallas Megakernel + Hybrid EP=8 x TP=2
# on TPU v6e-16.
set -euo pipefail

SERVICE_URL="${SERVICE_URL:-http://localhost:8000}"
OUT_DIR="${OUT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../results/raw" && pwd)}"
mkdir -p "${OUT_DIR}"

echo "[1/4] Waiting for vLLM OpenAI server at ${SERVICE_URL}/health ..."
for i in $(seq 1 120); do
  if curl -sf "${SERVICE_URL}/health" >/dev/null 2>&1; then
    echo "Server ready."
    break
  fi
  sleep 5
done

echo "[2/4] Running 16/16 greedy determinism & known-answer verification ..."
python3 "$(dirname "${BASH_SOURCE[0]}")/../../kernel-optimizations-recipe/scripts/eval_gpqa_diamond.py" \
  --base-url "${SERVICE_URL}/v1" \
  --model "deepseek-ai/DeepSeek-V4.1-Flash" \
  --limit 24 \
  --max-tokens 4096 \
  --output "${OUT_DIR}/gpqa_subset.json"

echo "[3/4] Running 1k/1k concurrency sweep across C={64, 128, 185, 190, 256} ..."
for c in 64 128 185 190 256; do
  python3 "$(dirname "${BASH_SOURCE[0]}")/../../../../recipes/benchmark_client.py" \
    --base-url "${SERVICE_URL}/v1" \
    --model "deepseek-ai/DeepSeek-V4.1-Flash" \
    --input-len 1024 \
    --output-len 1024 \
    --concurrency "${c}" \
    --num-prompts "$((c * 2))" \
    --output "${OUT_DIR}/1k1k-c${c}.json"
done

echo "[4/4] Generating 4-panel analysis chart ..."
python3 "$(dirname "${BASH_SOURCE[0]}")/plot_fused_moe_analysis.py"
echo "Benchmark suite complete. Results written to ${OUT_DIR}."
