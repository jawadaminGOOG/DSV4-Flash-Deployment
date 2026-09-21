#!/usr/bin/env bash
# Clone tpu-inference and vLLM at the base commits, then apply every patch.
#
# Usage: ./apply.sh <workdir>
#
# The script stops on the first failure. It writes only inside <workdir>.
set -euo pipefail

WORKDIR="${1:?usage: apply.sh <workdir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TI_REPO="https://github.com/vllm-project/tpu-inference.git"
TI_BASE="f22b5068d9326e7899cd2fc8afd4de79f36d20f4"
VLLM_REPO="https://github.com/vllm-project/vllm.git"
VLLM_BASE="9b959b86577c082c0b2bf9e2c22263255a36ad83"

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# `git am` needs an identity. Set one for this clone only, so the script never
# touches the caller's global git config.
[ -d tpu-inference ] || git clone "${TI_REPO}" tpu-inference
git -C tpu-inference checkout --detach "${TI_BASE}"
git -C tpu-inference config user.name "patch applier"
git -C tpu-inference config user.email "patch-applier@invalid"
git -C tpu-inference am "${HERE}"/tpu-inference/*.patch
echo "tpu-inference: $(git -C tpu-inference log --oneline -1)"

[ -d vllm ] || git clone "${VLLM_REPO}" vllm
git -C vllm checkout --detach "${VLLM_BASE}"
for p in "${HERE}"/vllm/*.patch; do
  git -C vllm apply "${p}"
  echo "vllm: applied $(basename "${p}")"
done

echo
echo "Both trees are patched under ${WORKDIR}."
echo "Next: build one image from them and set it in dsv41-flash-v6e16-serving.yaml."
