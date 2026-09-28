#!/usr/bin/env bash
# Container entrypoint: optionally overlay code from GCS, sync persistent JAX compilation cache
# from/to GCS, export the multi-host JAX settings derived from the JobSet, then run the command.
set -euo pipefail

if [[ -n "${CODE_TGZ:-}" ]]; then
  python - "$CODE_TGZ" <<'EOF'
import io, sys, tarfile
from google.cloud import storage
uri = sys.argv[1]
bucket, _, name = uri[5:].partition("/")
data = storage.Client().bucket(bucket).blob(name).download_as_bytes()
tarfile.open(fileobj=io.BytesIO(data)).extractall("/app")
print(f"overlaid code from {uri} ({len(data)} bytes)", flush=True)
EOF
fi

JAX_CACHE_GCS="${JAX_CACHE_GCS:-gs://dsv41-v7x-jawadamin-us-central1-0926/jax-cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/jax-cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

sync_cache_down() {
  python - "$JAX_CACHE_GCS" "$JAX_COMPILATION_CACHE_DIR" <<'EOF' || true
import os, sys
from google.cloud import storage
uri, local_dir = sys.argv[1], sys.argv[2]
if not uri.startswith("gs://"):
    sys.exit(0)
bucket_name, _, prefix = uri[5:].partition("/")
prefix = prefix.rstrip("/") + "/"
b = storage.Client().bucket(bucket_name)
n = 0
for blob in b.list_blobs(prefix=prefix):
    rel = blob.name[len(prefix):]
    if not rel or rel.endswith("/"):
        continue
    dst = os.path.join(local_dir, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    blob.download_to_filename(dst)
    n += 1
if n:
    print(f"synced {n} cached XLA compilation entries from {uri}", flush=True)
EOF
}

sync_cache_up() {
  if [[ "${JOB_COMPLETION_INDEX:-0}" != "0" ]]; then
    return 0
  fi
  python - "$JAX_CACHE_GCS" "$JAX_COMPILATION_CACHE_DIR" <<'EOF' || true
import os, sys
from google.cloud import storage
uri, local_dir = sys.argv[1], sys.argv[2]
if not uri.startswith("gs://") or not os.path.isdir(local_dir):
    sys.exit(0)
bucket_name, _, prefix = uri[5:].partition("/")
prefix = prefix.rstrip("/") + "/"
b = storage.Client().bucket(bucket_name)
n = 0
for root, _, files in os.walk(local_dir):
    for f in files:
        src = os.path.join(root, f)
        rel = os.path.relpath(src, local_dir)
        blob = b.blob(prefix + rel)
        if not blob.exists():
            blob.upload_from_filename(src)
            n += 1
if n:
    print(f"uploaded {n} new XLA compilation entries to {uri}", flush=True)
EOF
}

sync_cache_down
trap sync_cache_up EXIT

# JobSet gives every pod JOB_COMPLETION_INDEX; the process id of a multi-host slice is that index.
export TPU_WORKER_ID="${TPU_WORKER_ID:-${JOB_COMPLETION_INDEX:-0}}"
export CLOUD_TPU_TASK_ID="${CLOUD_TPU_TASK_ID:-$TPU_WORKER_ID}"
echo "entrypoint: host=$(hostname) worker=${TPU_WORKER_ID} processes=${NUM_PROCESSES:-1} coordinator=${COORDINATOR_ADDRESS:-none}" >&2
"$@"
