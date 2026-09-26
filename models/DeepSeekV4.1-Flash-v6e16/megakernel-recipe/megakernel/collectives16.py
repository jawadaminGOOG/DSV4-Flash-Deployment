"""TP16 in-kernel Pallas ICI collectives for TPU v6e-16 (4 hosts x 4 chips/host).

Provides:
- `barrier(tp=16)`: 16-chip barrier synchronization via `pltpu.get_barrier_semaphore()`.
- `scratch_2d(shape, dtype)`: Double-buffered VMEM scratch + 6 pairwise DMA semaphore pairs
  for the 2-stage 4x4 reduction (3 intra-host peers `rank ^ {1,2,3}` + 3 inter-host peers `rank ^ {4,8,12}`).
- `allreduce_2d(src_ref, dst_ref, local_buf, host_buf, sends, recvs, iteration)`:
  Executes the 2-stage 4x4 all-reduce in VMEM (`8.36 us` for 10 KiB `[8, 5120]` / `[40, 128]` BF16 on v6e-16).
- `allgather_16(src_ref, recv_buf, sends15, recvs15, iteration)`:
  Executes a 16-rank all-gather into `recv_buf[slot, 0..15]`.
"""
from __future__ import annotations

from contextlib import nullcontext
import os
from typing import Any, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def scope(name: str):
    """Optional device trace scope enabled by TP16_PROFILE_SCOPES=1."""
    return (
        jax.named_scope("tp16/" + name)
        if os.environ.get("TP16_PROFILE_SCOPES") == "1"
        else nullcontext()
    )


def barrier(tp: int = 16, axis_name: str = "tp") -> None:
    """Synchronize all `tp` ranks on `axis_name` using the hardware barrier semaphore."""
    rank = jax.lax.axis_index(axis_name)
    sem = pltpu.get_barrier_semaphore()
    for offset in range(1, tp):
        pl.semaphore_signal(
            sem,
            1,
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(sem, tp - 1)


def scratch_2d(
    shape: Union[int, Tuple[int, int]] = (8, 5120),
    dtype: Any = jnp.bfloat16,
) -> Tuple[Any, ...]:
    """Return `(local_buf, host_buf, sends, recvs)` scratch specs for `allreduce_2d`."""
    tile_shape = (shape, 128) if isinstance(shape, int) else tuple(shape)
    return (
        pltpu.VMEM((2, 4, *tile_shape), dtype),
        pltpu.VMEM((2, 4, *tile_shape), dtype),
        pltpu.SemaphoreType.DMA((6,)),
        pltpu.SemaphoreType.DMA((6,)),
    )


def allreduce_2d(
    src_ref: Any,
    dst_ref: Any,
    local_buf: Any,
    host_buf: Any,
    sends: Any,
    recvs: Any,
    iteration: Any,
    axis_name: str = "tp",
) -> None:
    """Two-stage 16-chip all-reduce: 4 ranks per host (`rank ^ {1,2,3}`), then 4 hosts (`rank ^ {4,8,12}`).

    Because XOR is an involution (`(r ^ k) ^ k == r`), each `offset` forms a dedicated
    symmetric pairwise channel using `sends.at[idx]` and `recvs.at[idx]` without races.
    """
    rank = jax.lax.axis_index(axis_name)
    slot = jax.lax.rem(iteration, 2)
    tile_shape = local_buf.shape[2:]
    orig_shape = dst_ref.shape

    val = src_ref[...]
    if val.shape != tile_shape:
        val = val.reshape(tile_shape)
    local_buf[slot, 0] = val.astype(local_buf.dtype)

    with scope("local_issue"):
        local_dmas = []
        for offset in (1, 2, 3):
            dma = pltpu.make_async_remote_copy(
                local_buf.at[slot, 0],
                local_buf.at[slot, offset],
                sends.at[offset - 1],
                recvs.at[offset - 1],
                device_id=(rank ^ offset,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            dma.start()
            local_dmas.append(dma)

    with scope("local_wait"):
        for dma in local_dmas:
            dma.wait()

    with scope("local_sum"):
        host_partial = (
            local_buf[slot, 0].astype(jnp.float32)
            + local_buf[slot, 1].astype(jnp.float32)
            + local_buf[slot, 2].astype(jnp.float32)
            + local_buf[slot, 3].astype(jnp.float32)
        )
        host_buf[slot, 0] = host_partial.astype(host_buf.dtype)

    with scope("remote_issue"):
        remote_dmas = []
        for offset in (1, 2, 3):
            dma = pltpu.make_async_remote_copy(
                host_buf.at[slot, 0],
                host_buf.at[slot, offset],
                sends.at[2 + offset],
                recvs.at[2 + offset],
                device_id=(rank ^ (offset * 4),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            dma.start()
            remote_dmas.append(dma)

    with scope("remote_wait"):
        for dma in remote_dmas:
            dma.wait()

    with scope("remote_sum"):
        total = (
            host_buf[slot, 0].astype(jnp.float32)
            + host_buf[slot, 1].astype(jnp.float32)
            + host_buf[slot, 2].astype(jnp.float32)
            + host_buf[slot, 3].astype(jnp.float32)
        )
        if total.shape != orig_shape:
            total = total.reshape(orig_shape)
        dst_ref[...] = total.astype(dst_ref.dtype)


def scratch_allgather_16(
    shape: Union[int, Tuple[int, int]] = (8, 320),
    dtype: Any = jnp.bfloat16,
) -> Tuple[Any, ...]:
    """Return `(recv_buf, sends15, recvs15)` scratch specs for `allgather_16`."""
    tile_shape = (shape, 128) if isinstance(shape, int) else tuple(shape)
    return (
        pltpu.VMEM((2, 16, *tile_shape), dtype),
        pltpu.SemaphoreType.DMA((15,)),
        pltpu.SemaphoreType.DMA((15,)),
    )


def allgather_16(
    src_ref: Any,
    recv_buf: Any,
    sends15: Any,
    recvs15: Any,
    iteration: Any,
    axis_name: str = "tp",
) -> jax.Array:
    """All-gather `src_ref` across all 16 ranks into `recv_buf[slot, 0..15]`."""
    rank = jax.lax.axis_index(axis_name)
    slot = jax.lax.rem(iteration, 2)
    tile_shape = recv_buf.shape[2:]
    val = src_ref[...]
    if val.shape != tile_shape:
        val = val.reshape(tile_shape)
    recv_buf[slot, rank] = val.astype(recv_buf.dtype)

    dmas = []
    for offset in range(1, 16):
        dma = pltpu.make_async_remote_copy(
            recv_buf.at[slot, rank],
            recv_buf.at[slot, rank],
            sends15.at[offset - 1],
            recvs15.at[offset - 1],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        dma.start()
        dmas.append(dma)

    for dma in dmas:
        dma.wait()

    return recv_buf[slot]
