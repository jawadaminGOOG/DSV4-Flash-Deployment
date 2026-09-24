# Copyright 2026 DeepSeek-V4.1-Flash TPU Deployment Authors.
# Adapted from Inferact/tpu-megakernels/collectives32.py for 16-chip TPU v6e 4x4 torus.
# Apache-2.0 License.
"""In-VMEM 2D torus collectives for 16-chip TPU v6e (`4x4` mesh: `('ep_y', 'etp_x')`).

Implements low-latency `tpu.make_async_remote_copy` ring collectives directly
inside a Pallas kernel body with zero host/HBM synchronization:
1. `torus_row_all_reduce_4`: 3-step ring all-reduce along the 4-chip `etp_x` axis.
2. `torus_col_all_reduce_4`: 3-step ring all-reduce along the 4-chip `ep_y` axis.
3. `torus_2d_all_reduce_16`: 6-step 2D torus all-reduce across all 16 chips (`etp_x` then `ep_y`).
"""

from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _ring_axis_all_reduce_4(
    data_vmem_ref,
    scratch_ping_ref,
    scratch_pong_ref,
    send_sems_ref,
    recv_sems_ref,
    barrier_sem_ref,
    *,
    axis_name: str,
) -> None:
    """Performs a 4-way ring all-reduce in VMEM along a single 4-chip torus axis (`etp_x` or `ep_y`).

    First exchanges a zero-byte barrier handshake with the immediate right neighbor
    `(my_id + 1) % 4` so all 4 chips along `axis_name` enter the remote DMA ring in lockstep,
    then streams `data_vmem_ref` across 3 ring hops and accumulates in FP32/BF16 in VMEM.
    """
    my_id = lax.axis_index(axis_name)
    right_neighbor = lax.rem(my_id + jnp.int32(1), jnp.int32(4))
    left_neighbor = lax.rem(my_id + jnp.int32(3), jnp.int32(4))

    # Seed scratch_ping_ref with local contribution and accumulate into data_vmem_ref in FP32
    local_val = data_vmem_ref[...]
    scratch_ping_ref[...] = local_val
    acc = local_val.astype(jnp.float32)

    # Receiver (i+1) signals sender (i = left_neighbor) that scratch_pong_ref is ready for Hop 1
    pl.semaphore_signal(
        barrier_sem_ref.at[0],
        inc=1,
        device_id={axis_name: left_neighbor},
    )
    pl.semaphore_wait(barrier_sem_ref.at[0], 1)

    # Hop 1: send scratch_ping_ref -> right_neighbor's scratch_pong_ref
    copy_1 = pltpu.make_async_remote_copy(
        src_ref=scratch_ping_ref,
        dst_ref=scratch_pong_ref,
        send_sem=send_sems_ref.at[0],
        recv_sem=recv_sems_ref.at[0],
        device_id={axis_name: right_neighbor},
    )
    copy_1.start()
    copy_1.wait_send()
    copy_1.wait_recv()
    acc = acc + scratch_pong_ref[...].astype(jnp.float32)

    # Receiver (i+1) signals sender (i = left_neighbor) that scratch_ping_ref is ready for Hop 2
    pl.semaphore_signal(
        barrier_sem_ref.at[0],
        inc=1,
        device_id={axis_name: left_neighbor},
    )
    pl.semaphore_wait(barrier_sem_ref.at[0], 1)

    # Hop 2: send scratch_pong_ref -> right_neighbor's scratch_ping_ref
    copy_2 = pltpu.make_async_remote_copy(
        src_ref=scratch_pong_ref,
        dst_ref=scratch_ping_ref,
        send_sem=send_sems_ref.at[1],
        recv_sem=recv_sems_ref.at[1],
        device_id={axis_name: right_neighbor},
    )
    copy_2.start()
    copy_2.wait_send()
    copy_2.wait_recv()
    acc = acc + scratch_ping_ref[...].astype(jnp.float32)

    # Receiver (i+1) signals sender (i = left_neighbor) that scratch_pong_ref is ready for Hop 3
    pl.semaphore_signal(
        barrier_sem_ref.at[0],
        inc=1,
        device_id={axis_name: left_neighbor},
    )
    pl.semaphore_wait(barrier_sem_ref.at[0], 1)

    # Hop 3: send scratch_ping_ref -> right_neighbor's scratch_pong_ref
    copy_3 = pltpu.make_async_remote_copy(
        src_ref=scratch_ping_ref,
        dst_ref=scratch_pong_ref,
        send_sem=send_sems_ref.at[2],
        recv_sem=recv_sems_ref.at[2],
        device_id={axis_name: right_neighbor},
    )
    copy_3.start()
    copy_3.wait_send()
    copy_3.wait_recv()
    acc = acc + scratch_pong_ref[...].astype(jnp.float32)

    data_vmem_ref[...] = acc.astype(data_vmem_ref.dtype)


def torus_row_all_reduce_4(
    data_vmem_ref,
    scratch_ping_ref,
    scratch_pong_ref,
    send_sems_ref,
    recv_sems_ref,
    barrier_sem_ref,
    *,
    axis_x: str = "etp_x",
) -> None:
    """All-reduces `data_vmem_ref` across the 4 chips in the same `etp_x` torus row (`ETP=4`)."""
    _ring_axis_all_reduce_4(
        data_vmem_ref,
        scratch_ping_ref,
        scratch_pong_ref,
        send_sems_ref,
        recv_sems_ref,
        barrier_sem_ref,
        axis_name=axis_x,
    )


def torus_col_all_reduce_4(
    data_vmem_ref,
    scratch_ping_ref,
    scratch_pong_ref,
    send_sems_ref,
    recv_sems_ref,
    barrier_sem_ref,
    *,
    axis_y: str = "ep_y",
) -> None:
    """All-reduces `data_vmem_ref` across the 4 chips in the same `ep_y` torus column (`EP=4`)."""
    _ring_axis_all_reduce_4(
        data_vmem_ref,
        scratch_ping_ref,
        scratch_pong_ref,
        send_sems_ref,
        recv_sems_ref,
        barrier_sem_ref,
        axis_name=axis_y,
    )


def torus_2d_all_reduce_16(
    data_vmem_ref,
    scratch_ping_ref,
    scratch_pong_ref,
    row_send_sems_ref,
    row_recv_sems_ref,
    row_barrier_sem_ref,
    col_send_sems_ref,
    col_recv_sems_ref,
    col_barrier_sem_ref,
    *,
    axis_y: str = "ep_y",
    axis_x: str = "etp_x",
) -> None:
    """Performs a full 16-chip 2D torus all-reduce in 6 hops (`3` along `etp_x` + `3` along `ep_y`).

    Used after both `TP=16` Attention (`o_proj` partial sums) and `ETP=4 x EP=4` MoE (`w2` partial sums).
    """
    torus_row_all_reduce_4(
        data_vmem_ref,
        scratch_ping_ref,
        scratch_pong_ref,
        row_send_sems_ref,
        row_recv_sems_ref,
        row_barrier_sem_ref,
        axis_x=axis_x,
    )
    torus_col_all_reduce_4(
        data_vmem_ref,
        scratch_ping_ref,
        scratch_pong_ref,
        col_send_sems_ref,
        col_recv_sems_ref,
        col_barrier_sem_ref,
        axis_y=axis_y,
    )
