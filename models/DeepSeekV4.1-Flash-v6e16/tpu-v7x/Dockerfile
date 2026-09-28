# Runtime image for the DeepSeek-V4.1 v7x megakernel: the pinned JAX 0.11.1 stack,
# CPU PyTorch (for golden reference runs on c4-highmem-192), gcsfs/etils (for GCS cache),
# and lm-eval + HTTP server dependencies.
# Code can also be overlaid at start from a tarball (CODE_TGZ=gs://...), so a code change does
# not need a rebuild.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch" \
        "jax[tpu]==0.11.1" "jaxlib==0.11.1" "libtpu==0.0.46.1" \
        "numpy>=2" "safetensors>=0.6" "tiktoken>=0.14.0" "tokenizers>=0.21" \
        "huggingface-hub>=0.34" "google-cloud-storage>=2.18" "ml_dtypes" \
        "gcsfs" "etils[epath]" "fastapi" "uvicorn" "aiohttp" "pytest" \
        "datasets>=3.0.0" "lm-eval[api]>=0.4.8" \
    && python -c "import jax, jaxlib, torch, lm_eval; print('jax', jax.__version__, 'torch', torch.__version__, 'lm_eval', lm_eval.__version__)"

WORKDIR /app
COPY . /app
ENV PYTHONPATH=/app
ENTRYPOINT ["/bin/bash", "/app/deepseek_v41/tpu-v7x/scripts/entrypoint.sh"]
