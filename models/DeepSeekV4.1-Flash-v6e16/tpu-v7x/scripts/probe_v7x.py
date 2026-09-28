"""Hardware probe for TPU v7x slices on GKE.

Records the facts the DeepSeek-V4.1 decode kernel depends on: the JAX device model and
ordering, per-device HBM, whether Pallas lowers a 64 MiB VMEM kernel and FP8 / FP4 dots,
in-kernel remote DMA latency and bandwidth to XOR peers, HBM bandwidth, and storage-to-HBM
load throughput. Every check is isolated, so one failure does not hide the others, and the
results are written as JSON (one file per process) to a local directory or a GCS prefix.

Run on CPU (interpret mode) to check the script itself::

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        python probe_v7x.py --interpret --out /tmp/probe
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import socket
import sys
import time
import traceback

import numpy as np


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write(out: str, name: str, payload: dict) -> None:
    data = json.dumps(payload, indent=2, default=str)
    if out.startswith("gs://"):
        from google.cloud import storage

        bucket, _, prefix = out[5:].partition("/")
        storage.Client().bucket(bucket).blob(f"{prefix.rstrip('/')}/{name}").upload_from_string(
            data, content_type="application/json"
        )
    else:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, name), "w") as f:
            f.write(data)


def _run(results: dict, name: str, fn, *args, **kwargs) -> None:
    start = time.perf_counter()
    try:
        results[name] = {"ok": True, **fn(*args, **kwargs)}
    except Exception as exc:  # noqa: BLE001 - a probe records every failure and moves on
        results[name] = {"ok": False, "error": repr(exc), "trace": traceback.format_exc()[-4000:]}
    results[name]["seconds"] = round(time.perf_counter() - start, 3)
    print(f"[probe] {name}: ok={results[name]['ok']} ({results[name]['seconds']} s)", flush=True)


def maybe_initialize_distributed() -> None:
    """Joins the multi-host JAX runtime when the pod belongs to a multi-host JobSet."""
    import jax

    num = int(os.environ.get("NUM_PROCESSES", "1"))
    if num <= 1:
        return
    rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("TPU_WORKER_ID", "0")))
    os.environ.setdefault("TPU_WORKER_ID", str(rank))
    os.environ.setdefault("CLOUD_TPU_TASK_ID", str(rank))
    jax.distributed.initialize(
        coordinator_address=os.environ["COORDINATOR_ADDRESS"],
        num_processes=num,
        process_id=rank,
        initialization_timeout=600,
    )


def probe_environment() -> dict:
    import jax
    import jaxlib

    try:
        import libtpu  # noqa: F401

        libtpu_version = getattr(libtpu, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        libtpu_version = None
    devices = []
    for d in sorted(jax.devices(), key=lambda d: (d.process_index, d.id)):
        devices.append(
            {
                "id": d.id,
                "process_index": d.process_index,
                "kind": d.device_kind,
                "coords": list(getattr(d, "coords", []) or []),
                "core_on_chip": getattr(d, "core_on_chip", None),
                "platform": d.platform,
            }
        )
    memory = {}
    for d in jax.local_devices():
        try:
            stats = d.memory_stats() or {}
            memory[str(d.id)] = {k: stats.get(k) for k in ("bytes_limit", "bytes_in_use", "largest_free_block_bytes")}
        except Exception as exc:  # noqa: BLE001
            memory[str(d.id)] = repr(exc)
    tpu_info = None
    try:
        from jax.experimental.pallas import tpu as pltpu

        info = pltpu.get_tpu_info()
        tpu_info = {k: getattr(info, k) for k in dir(info) if not k.startswith("_") and not callable(getattr(info, k))}
    except Exception as exc:  # noqa: BLE001
        tpu_info = repr(exc)
    env = {k: v for k, v in os.environ.items() if k.startswith(("TPU", "JAX", "XLA", "LIBTPU", "MEGASCALE", "JOB_", "COORDINATOR", "NUM_PROCESSES"))}
    return {
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "libtpu": libtpu_version,
        "backend": jax.default_backend(),
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "devices": devices,
        "memory": memory,
        "tpu_info": tpu_info,
        "env": env,
        "cpu_count": os.cpu_count(),
        "shm_bytes_free": _statvfs_free("/dev/shm"),
    }


def _statvfs_free(path: str):
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except OSError:
        return None


def probe_hbm_bandwidth(gib: float = 4.0) -> dict:
    import jax
    import jax.numpy as jnp

    device = jax.local_devices()[0]
    n = int(gib * 2**30 // 2)
    x = jax.device_put(jnp.ones((n,), jnp.bfloat16), device)
    f = jax.jit(lambda a: a * 2 + 1)
    f(x).block_until_ready()
    reps = 10
    start = time.perf_counter()
    for _ in range(reps):
        y = f(x)
    y.block_until_ready()
    seconds = (time.perf_counter() - start) / reps
    return {"bytes_moved_per_rep": 2 * n * 2, "seconds_per_rep": seconds, "gbps": 2 * n * 2 / seconds / 1e9}


def probe_pallas_features(interpret) -> dict:
    """Lowers the kernel features the decode kernel needs, one at a time."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    out: dict = {}
    device = jax.local_devices()[0]

    # 1. A 60 MiB VMEM scratch under a 64 MiB limit.
    def big_vmem_kernel(x_ref, o_ref, scratch):
        scratch[...] = jnp.zeros(scratch.shape, scratch.dtype)
        scratch[0:8, :] = x_ref[...]
        o_ref[...] = scratch[0:8, :] + 1

    rows = (60 * 2**20) // (4 * 1024)
    try:
        f = pl.pallas_call(
            big_vmem_kernel,
            out_shape=jax.ShapeDtypeStruct((8, 1024), jnp.float32),
            scratch_shapes=[pltpu.VMEM((rows, 1024), jnp.float32)],
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 2**20),
            interpret=interpret,
        )
        r = f(jax.device_put(jnp.ones((8, 1024), jnp.float32), device))
        out["vmem_60mib"] = bool(jnp.all(r == 2))
    except Exception as exc:  # noqa: BLE001
        out["vmem_60mib"] = repr(exc)[:2000]

    # 2. FP8 x FP8, BF16 x FP8, and MXFP4 -> FP8 upcast dots, checked against XLA.
    k, n, m = 512, 256, 16
    rng = np.random.default_rng(0)
    a32 = rng.standard_normal((m, k)).astype(np.float32)
    b32 = rng.standard_normal((k, n)).astype(np.float32)
    a8 = jnp.asarray(a32, jnp.float8_e4m3fn)
    b8 = jnp.asarray(b32, jnp.float8_e4m3fn)
    abf = jnp.asarray(a32, jnp.bfloat16)

    def dot_kernel(a_ref, b_ref, o_ref):
        o_ref[...] = jax.lax.dot_general(
            a_ref[...], b_ref[...], (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )

    for name, a, b in (("fp8_x_fp8", a8, b8), ("bf16_x_fp8", abf, b8)):
        try:
            f = pl.pallas_call(dot_kernel, out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32), interpret=interpret)
            got = np.asarray(f(jax.device_put(a, device), jax.device_put(b, device)))
            want = np.asarray(jnp.dot(a.astype(jnp.float32), b.astype(jnp.float32)))
            out[name] = {"max_abs_err": float(np.max(np.abs(got - want))), "max_abs_ref": float(np.max(np.abs(want)))}
        except Exception as exc:  # noqa: BLE001
            out[name] = repr(exc)[:2000]

    # MXFP4 codes packed eight per uint32 along the contraction, as the expert weights are.
    codes = rng.integers(0, 16, size=(k, n), dtype=np.uint8)
    packed = np.zeros((k // 8, n), np.uint32)
    for j in range(8):
        packed |= codes[j::8].astype(np.uint32) << (4 * j)
    e2m1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)

    def fp4_kernel(a_ref, p_ref, o_ref):
        values = pltpu.bitcast(p_ref[...], jnp.float4_e2m1fn).astype(jnp.float8_e4m3fn)
        o_ref[...] = jax.lax.dot_general(
            a_ref[...], values, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )

    try:
        f = pl.pallas_call(fp4_kernel, out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32), interpret=interpret)
        got = np.asarray(f(jax.device_put(abf, device), jax.device_put(jnp.asarray(packed), device)))
        # The in-kernel bitcast defines the nibble order; record both candidate orders.
        want_lsb = np.asarray(abf, np.float32) @ e2m1[codes]
        interleaved = np.zeros_like(codes)
        for j in range(8):
            interleaved[j * (k // 8) : (j + 1) * (k // 8)] = codes[j::8]
        want_rowmajor = np.asarray(abf, np.float32) @ e2m1[interleaved]
        out["mxfp4_to_fp8"] = {
            "err_vs_rows_interleaved_by_nibble": float(np.max(np.abs(got - want_lsb))),
            "err_vs_rows_blocked_by_nibble": float(np.max(np.abs(got - want_rowmajor))),
            "max_abs_ref": float(np.max(np.abs(want_lsb))),
        }
    except Exception as exc:  # noqa: BLE001
        out["mxfp4_to_fp8"] = repr(exc)[:2000]
    return out


def probe_remote_dma(interpret, payload_bytes=(4096, 1 << 20, 8 << 20), iters: int = 16) -> dict:
    """Times in-kernel HBM-to-HBM remote copies to each XOR peer distance on a 1-D mesh."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    devices = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
    size = len(devices)
    if size < 2:
        return {"skipped": "one device"}
    mesh = Mesh(np.array(devices), ("tp",))
    offsets = [1 << b for b in range(size.bit_length() - 1) if (1 << b) < size]
    results = {"devices": size, "offsets": offsets, "timings": {}}

    for nbytes in payload_bytes:
        rows = max(8, nbytes // (128 * 4))

        def kernel(x_ref, o_ref, send, recv):
            rank = jax.lax.axis_index("tp")
            barrier = pltpu.get_barrier_semaphore()
            for off in offsets:
                pl.semaphore_signal(barrier, 1, device_id=(rank ^ off,), device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(barrier, len(offsets))
            for _ in range(iters):
                for i, off in enumerate(offsets):
                    copy = pltpu.make_async_remote_copy(
                        x_ref, o_ref.at[i], send.at[i], recv.at[i],
                        device_id=(rank ^ off,), device_id_type=pl.DeviceIdType.MESH,
                    )
                    copy.start()
                    copy.wait()

        def local(x):
            return pl.pallas_call(
                kernel,
                out_shape=jax.ShapeDtypeStruct((len(offsets), rows, 128), jnp.float32),
                in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
                out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                scratch_shapes=[pltpu.SemaphoreType.DMA((len(offsets),)), pltpu.SemaphoreType.DMA((len(offsets),))],
                compiler_params=pltpu.CompilerParams(collective_id=7),
                interpret=interpret,
            )(x[0])[None]

        f = jax.jit(jax.shard_map(local, mesh=mesh, in_specs=P("tp"), out_specs=P("tp"), check_vma=False))
        x = jax.device_put(
            jnp.arange(size * rows * 128, dtype=jnp.float32).reshape(size, rows, 128),
            NamedSharding(mesh, P("tp")),
        )
        f(x).block_until_ready()
        reps = 3
        start = time.perf_counter()
        for _ in range(reps):
            y = f(x)
        y.block_until_ready()
        per_call = (time.perf_counter() - start) / reps
        per_copy = per_call / (iters * len(offsets))
        results["timings"][str(rows * 128 * 4)] = {
            "seconds_per_call": per_call,
            "seconds_per_copy_avg": per_copy,
            "gbps_per_copy": rows * 128 * 4 / per_copy / 1e9,
        }
    # One correctness check on the last payload: rank r's slot i must hold rank (r ^ offsets[i])'s input.
    from jax.experimental import multihost_utils

    got = np.asarray(multihost_utils.process_allgather(y, tiled=True))
    src = np.asarray(multihost_utils.process_allgather(x, tiled=True))
    ok = all(np.array_equal(got[r, i], src[r ^ off]) for r in range(size) for i, off in enumerate(offsets))
    results["data_correct"] = bool(ok)
    return results


def probe_storage(gcs_prefix: str, files: list[str], workers: int, stage_dir: str) -> dict:
    """Downloads checkpoint shards with parallel range reads, then loads them into HBM."""
    import jax
    import jax.numpy as jnp
    from google.cloud import storage

    bucket_name, _, prefix = gcs_prefix[5:].partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    chunk = 64 << 20
    os.makedirs(stage_dir, exist_ok=True)

    def fetch(job):
        name, start, end, path = job
        data = bucket.blob(f"{prefix.rstrip('/')}/{name}").download_as_bytes(start=start, end=end - 1, raw_download=True)
        with open(path, "r+b") as f:
            f.seek(start)
            f.write(data)
        return len(data)

    jobs, total = [], 0
    for name in files:
        blob = bucket.get_blob(f"{prefix.rstrip('/')}/{name}")
        path = os.path.join(stage_dir, name)
        with open(path, "wb") as f:
            f.truncate(blob.size)
        for s in range(0, blob.size, chunk):
            jobs.append((name, s, min(blob.size, s + chunk), path))
        total += blob.size
    start = time.perf_counter()
    with cf.ThreadPoolExecutor(workers) as pool:
        moved = sum(pool.map(fetch, jobs))
    download_s = time.perf_counter() - start

    # Host RAM -> HBM: spread the staged bytes over the local devices.
    local = jax.local_devices()
    arrays = [np.memmap(os.path.join(stage_dir, n), dtype=np.uint8, mode="r") for n in files]
    start = time.perf_counter()
    placed = []
    for i, arr in enumerate(arrays):
        usable = (arr.size // 4096) * 4096
        placed.append(jax.device_put(np.asarray(arr[:usable]).reshape(-1, 4096), local[i % len(local)]))
    for p in placed:
        p.block_until_ready()
    h2d_s = time.perf_counter() - start
    for n in files:
        os.remove(os.path.join(stage_dir, n))
    return {
        "files": files,
        "bytes": total,
        "moved": moved,
        "workers": workers,
        "download_seconds": download_s,
        "download_gbps": total / download_s / 1e9,
        "host_to_hbm_seconds": h2d_s,
        "host_to_hbm_gbps": total / h2d_s / 1e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="local dir or gs:// prefix for results")
    parser.add_argument("--interpret", action="store_true", help="run Pallas in TPU interpret mode (CPU)")
    parser.add_argument("--skip", default="", help="comma list of checks to skip")
    parser.add_argument("--gcs-prefix", default="gs://dsv41-v7x-jawadamin-us-central1-0926/hf/deepseek-v4.1-flash")
    parser.add_argument("--files-per-host", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=64)
    parser.add_argument("--stage-dir", default="/dev/shm/probe")
    args = parser.parse_args()

    maybe_initialize_distributed()
    import jax

    interpret = False
    if args.interpret:
        from jax.experimental.pallas import tpu as pltpu

        interpret = pltpu.InterpretParams()
    skip = set(filter(None, args.skip.split(",")))
    results: dict = {"started": _now()}
    _run(results, "environment", probe_environment)
    if "hbm" not in skip:
        _run(results, "hbm_bandwidth", probe_hbm_bandwidth, 0.25 if args.interpret else 4.0)
    if "pallas" not in skip:
        _run(results, "pallas_features", probe_pallas_features, interpret)
    if "remote" not in skip:
        _run(results, "remote_dma", probe_remote_dma, interpret, (4096, 1 << 20) if args.interpret else (4096, 1 << 20, 8 << 20))
    if "storage" not in skip:
        pid = jax.process_index()
        # Shards 3..42 are ~7.39 GB each; each process reads a disjoint set.
        first = 3 + pid * args.files_per_host
        files = [f"model-{i:05d}-of-00048.safetensors" for i in range(first, first + args.files_per_host)]
        _run(results, "storage", probe_storage, args.gcs_prefix, files, args.download_workers, args.stage_dir)
    results["finished"] = _now()
    _write(args.out, f"process{jax.process_index():02d}.json", results)
    print(json.dumps({k: (v.get("ok") if isinstance(v, dict) else v) for k, v in results.items()}), flush=True)


if __name__ == "__main__":
    main()
