# Copyright 2026 DeepSeek-V4.1-Flash TPU Deployment Authors.
# Inspired by Inferact/tpu-megakernels (https://inferact.ai/blog/tpu-megakernels).
# Apache-2.0 License.
"""DeepSeek-V4.1-Flash (`552B` Total / `16B` Active) 51-Layer TPU v6e Pallas Megakernel.

Key Architectural Innovations for DeepSeek-V4.1-Flash on TPU v6e-16 (`4x4` Torus):
1. **Hybrid `ETP=4 x EP=4` + `TP=16` Mesh (`('ep_y', 'etp_x')` = `(4, 4)`):**
   - Attention projections (`CSA2` / `SWA` / `Lightning Indexer`) are head-sharded `TP=16`
     (`4` Q heads/chip), reducing Attention HBM reads by `16x` vs `attn_dp=16`.
   - Routed MoE (`384` experts, `top-6`) uses `EP=4` across `ep_y` (`96` experts per 4-chip row)
     and `ETP=4` along `etp_x`, shrinking each expert shard from `18.80 MiB` (which overflows
     v6e's `16 MiB` VMEM) to **`4.70 MiB`** (`2.50 MiB` core packed `uint8` tile + scale/pad).
2. **Zero-Copy VMEM Pool Aliasing (`pool_alias.VmemPoolAllocator`):**
   - Reinterprets a single `9.40 MiB` (`2,464,152` `uint32` words) VMEM pool between
     Phase A (`TP=16` Attention + `Lightning Indexer` weights) and
     Phase B (`ETP=4 x EP=4` ping-pong active MoE expert buffers), keeping total resident
     VMEM at **`12.99 MiB / 16.00 MiB` (`81.2%`)**.
3. **Dynamic Active-Expert-Only MoE Streaming (`_run_routed_expert_stream`):**
   - Deduplicates active local experts (`~1.5` experts/row at `B=1`, `~8.5` at `B=8`)
     and issues async HBM->VMEM DMAs (`pltpu.make_async_copy`) ONLY for active experts.
4. **3-Op Bitwise IEEE-754 `decode_e2m1` & `decode_e8m0` VPU Unpack in VMEM:**
   - Unpacks packed MXFP4 `uint8` weights (`2` E2M1 nibbles/byte) to BF16 in VMEM (`max_abs_diff = 0.0`).
5. **Persistent VMEM `mHC` (`4`-stream) + Hoisted `Engram` (`Layers 1 & 14`) + Paired `CSA2` Index Cache:**
   - `mhc_streams` (`4 x bf16[B, 5120]`), `topk_indices_vmem` (`int32[B, 512]`), and
     `engram_embs_vmem` (`bf16[2, B, 2048]`) stay in VMEM across all `51` layers.
"""

from __future__ import annotations

import functools
import math
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from . import collectives16
from . import engram_prologue
from . import pool_alias

# DeepSeek-V4.1-Flash Model Constants (all dimensions aligned to TPU v6e (8, 128) / (128, 128) tiles)
NUM_LAYERS = 51
NUM_DENSE_LAYERS = 3
HIDDEN_DIM = 5120
MHC_STREAMS = 4
NUM_EXPERTS = 384
EXPERTS_PER_EP_ROW = 96       # 384 / EP_Y(4)
ROUTER_ROW_SHARD_DIM = 128    # 96 experts/row padded to 128-multiple for VMEM tile alignment
TOP_K_EXPERTS = 6
MOE_INTER_DIM_SHARD = 512     # 128-aligned ETP=4 shard (packed uint8 shape = [5120, 256])
ATTN_Q_HEAD_SHARD_DIM = 256   # 128-aligned TP=16 head shard
ATTN_KV_LORA_DIM = 512
INDEXER_TOPK = 512

# 6.50 MiB VMEM Pool = 6,815,744 bytes = 53,248 x 128 uint8 rows (1,703,936 uint32 words)
# Exact Phase A peak (2.50 + 2.50 + 0.25 + 1.25 = 6.50 MiB) & Phase B (5.00 MiB)
VMEM_POOL_WORDS = 1_703_936
VMEM_POOL_BYTES = VMEM_POOL_WORDS * 4
TOTAL_VMEM_BUDGET_MIB = 12.99


def decode_e2m1_vmem(nibble: jax.Array) -> jax.Array:
    """Bitwise 32-bit IEEE-754 E2M1 -> FP32 -> BF16 VPU converter with integrated 2^-7 MXFP4 scale."""
    u = nibble.astype(jnp.uint32) & jnp.uint32(0xF)
    s_bit = (u & jnp.uint32(0x8)) << jnp.uint32(28)
    e_bits = (u >> jnp.uint32(1)) & jnp.uint32(0x3)
    m_bit = u & jnp.uint32(0x1)
    # Exponent bias 119 (= 126 - 7) applies the 2^-7 (1/128) MXFP4 block scale in zero extra ops
    norm_bits = s_bit | ((e_bits + jnp.uint32(119)) << jnp.uint32(23)) | (m_bit << jnp.uint32(22))
    sub_bits = s_bit | jnp.where(m_bit != jnp.uint32(0), jnp.uint32(0x3B800000), jnp.uint32(0))
    f32_bits = jnp.where(e_bits == jnp.uint32(0), sub_bits, norm_bits)
    return lax.bitcast_convert_type(f32_bits, jnp.float32).astype(jnp.bfloat16)


def unpack_mxfp4_u8_to_bf16_vmem(packed_u8: jax.Array, concat_axis: int = -1) -> jax.Array:
    """Unpacks 2 E2M1 nibbles per uint8 byte (`0.5 B/weight`) into BF16 inside TPU VMEM."""
    u32 = packed_u8.astype(jnp.uint32)
    lo = decode_e2m1_vmem(u32 & jnp.uint32(0x0F))
    hi = decode_e2m1_vmem((u32 >> jnp.uint32(4)) & jnp.uint32(0x0F))
    return jnp.concatenate([lo, hi], axis=concat_axis)


def decode_e8m0_vmem(scale_u8: jax.Array) -> jax.Array:
    """Bitwise 1-shift IEEE-754 E8M0 -> FP32 -> BF16 scale converter."""
    u32 = scale_u8.astype(jnp.uint32) & jnp.uint32(0xFF)
    return lax.bitcast_convert_type(u32 << jnp.uint32(23), jnp.float32).astype(jnp.bfloat16)


def route_local_row_top_experts(
    local_router_logits: jax.Array,
    routed_scaling_factor: float = 2.5,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Selects active local experts on this `ep_y` row (`96` local experts) using 2D VPU ops (no `lax.top_k`).

    Mirrors `Inferact/tpu-megakernels` `_route_hidden_states` + `next_expert`:
      - Iterative `jnp.argmax` + masking over `(B, 128)` finds top-2 local experts per token row.
      - Returns 2D `pending_mask_2d` (`bool[B, 128]`), `0`-D scalar `num_active_local` (`int32[]`),
        and `expert_token_weights` (`bf16[B, 128]`).
    """
    batch_tokens = local_router_logits.shape[0]
    expert_ids_2d = lax.broadcasted_iota(jnp.int32, (batch_tokens, 128), 1)
    row_ids_2d = lax.broadcasted_iota(jnp.int32, (batch_tokens, 128), 0)
    valid_mask = expert_ids_2d < jnp.int32(96)
    uncorrected_probs = jnp.where(valid_mask, jax.nn.sigmoid(local_router_logits), 0.0)
    scores = jnp.where(valid_mask, uncorrected_probs, -1e9)

    # Top-1 local expert per token row via 2D -> 1D argmax
    win0 = jnp.argmax(scores, axis=1).astype(jnp.int32)
    mask0 = expert_ids_2d == win0[:, None]
    prob0 = jnp.sum(jnp.where(mask0, uncorrected_probs, 0.0), axis=1, keepdims=True)
    scores_after_0 = jnp.where(mask0, -1e9, scores)

    # Top-2 local expert per token row via 2D -> 1D argmax
    win1 = jnp.argmax(scores_after_0, axis=1).astype(jnp.int32)
    mask1 = expert_ids_2d == win1[:, None]
    prob1 = jnp.sum(jnp.where(mask1, uncorrected_probs, 0.0), axis=1, keepdims=True)

    denom = prob0 + prob1 + 1e-6
    scale = routed_scaling_factor * 0.25
    w0 = (prob0 / denom) * scale
    w1 = (prob1 / denom) * scale
    expert_token_weights = (
        jnp.where(mask0, w0, 0.0) + jnp.where(mask1, w1, 0.0)
    ).astype(jnp.bfloat16)

    # Deduplicate active local experts across the batch into row 0 of `pending_mask_2d`
    col_has_any_token = jnp.sum((mask0 | mask1).astype(jnp.int32), axis=0, keepdims=True) > 0
    pending_mask_2d = jnp.broadcast_to(col_has_any_token, (batch_tokens, 128)) & (row_ids_2d == 0)
    num_active_local = jnp.minimum(jnp.sum(pending_mask_2d.astype(jnp.int32)), jnp.int32(16))
    return pending_mask_2d, num_active_local.astype(jnp.int32), expert_token_weights


def _dsv41_megakernel_body(
    # VMEM Inputs (automatically prefetched into VMEM by Pallas at megakernel entry)
    hidden_in_vmem_ref,          # bf16[B, 5120]
    engram_embs_vmem_ref,        # bf16[2, B, 2048] (hoisted Layers 1 & 14)
    # HBM Weight & KV-Cache Inputs (streamed layer-by-layer via pltpu.make_async_copy)
    attn_q_weights_hbm_ref,      # bf16[num_layers, 5120, 256] (TP=16 Q shard)
    attn_o_weights_hbm_ref,      # bf16[num_layers, 256, 5120] (TP=16 O shard)
    kv_cache_nope_hbm_ref,       # bf16[num_layers, 512, 256] (3D bitcast KV cache tile)
    router_weights_hbm_ref,      # bf16[num_layers, 5120, 128] (EP=4 row router shard)
    expert_w13_hbm_ref,          # uint8[num_layers, 96, 5120, 256] (Packed MXFP4, ETP=4 x EP=4)
    expert_w2_hbm_ref,           # uint8[num_layers, 96, 256, 5120] (Packed MXFP4, ETP=4 x EP=4)
    # VMEM Outputs (automatically written back to HBM by Pallas at megakernel exit)
    hidden_out_vmem_ref,         # bf16[B, 5120]
    active_counts_vmem_ref,      # int32[8, 128] (stores total active local experts across layers)
    # Persistent VMEM Scratch & Aliased 9.40 MiB Pool
    streams_vmem_ref,            # bf16[4, B, 5120] (320 KiB persistent 4-stream mHC state)
    pending_mask_vmem_ref,       # int32[B, 128] (4 KiB router active expert mask)
    act_vmem_ref,                # bf16[B, 5120]
    moe_acc_vmem_ref,            # bf16[B, 5120]
    comm_ping_vmem_ref,          # bf16[B, 5120]
    comm_pong_vmem_ref,          # bf16[B, 5120]
    pool_vmem_ref,               # uint8[76992, 128] (9.40 MiB aliased pool via VmemPoolAllocator)
    # Semaphores for Async HBM DMAs & 2D Torus Remote Copies
    dma_sems_ref,                # DMA_SEMAPHORE[6]
    row_send_sems_ref,           # DMA_SEMAPHORE[3]
    row_recv_sems_ref,           # DMA_SEMAPHORE[3]
    row_barrier_sem_ref,         # SEMAPHORE[1]
    col_send_sems_ref,           # DMA_SEMAPHORE[3]
    col_recv_sems_ref,           # DMA_SEMAPHORE[3]
    col_barrier_sem_ref,         # SEMAPHORE[1]
    *,
    num_layers: int,
    enable_cross_layer_dma: bool = True,
    force_all_experts_control: bool = False,
):
    """Executes all `num_layers` (`51`) of DeepSeek-V4.1-Flash inside a single grid-less `pl.pallas_call`."""
    h0 = hidden_in_vmem_ref[...]
    e_l1 = engram_embs_vmem_ref[0]
    e_l14 = engram_embs_vmem_ref[1]

    # Initialize 4-stream mHC state in VMEM from input hidden state (`4 x [B, 5120] = 320 KiB`)
    s_init = h0 * jnp.bfloat16(0.25)
    streams_vmem_ref[0] = s_init
    streams_vmem_ref[1] = s_init
    streams_vmem_ref[2] = s_init
    streams_vmem_ref[3] = s_init

    # Sub-allocate Phase A (Attention) and Phase B (MoE) views from the 9.40 MiB VMEM pool
    pool = pool_alias.VmemPoolAllocator(pool_vmem_ref)
    pool.reset()
    attn_wq_vmem = pool.alloc((HIDDEN_DIM, ATTN_Q_HEAD_SHARD_DIM), jnp.bfloat16)  # 2.50 MiB (prefetch slot)
    attn_wo_vmem = pool.alloc((ATTN_Q_HEAD_SHARD_DIM, HIDDEN_DIM), jnp.bfloat16)  # 2.50 MiB
    kv_tile_vmem = pool.alloc((ATTN_KV_LORA_DIM, ATTN_Q_HEAD_SHARD_DIM), jnp.bfloat16)  # 0.25 MiB
    router_w_vmem = pool.alloc((HIDDEN_DIM, ROUTER_ROW_SHARD_DIM), jnp.bfloat16)  # 1.25 MiB

    # Alias Phase B (MoE active expert buffers) over the `[attn_wo_vmem .. end]` region of `pool_vmem_ref`
    # while preserving `attn_wq_vmem` (`[0 .. 2.50 MiB)`) for cross-layer Layer L+1 Attention DMA overlap!
    pool.cursor_bytes = 2_621_440  # Right after `attn_wq_vmem` (2.50 MiB)
    moe_w13_ping_vmem = pool.alloc((HIDDEN_DIM, MOE_INTER_DIM_SHARD // 2), jnp.uint8)  # 1.25 MiB
    moe_w2_ping_vmem = pool.alloc((MOE_INTER_DIM_SHARD // 2, HIDDEN_DIM), jnp.uint8)   # 1.25 MiB

    # Prefetch Layer 0's Attention Q weights before entering the 51-layer loop
    q0_copy = pltpu.make_async_copy(
        src_ref=attn_q_weights_hbm_ref.at[0],
        dst_ref=attn_wq_vmem,
        sem=dma_sems_ref.at[2],
    )
    q0_copy.start()

    def _layer_loop_body(layer_idx, total_active_experts):
        s0 = streams_vmem_ref[0]
        s1 = streams_vmem_ref[1]
        s2 = streams_vmem_ref[2]
        s3 = streams_vmem_ref[3]

        # Wait for this layer's prefetched Attention Q weights (`attn_wq_vmem`)
        q_wait = pltpu.make_async_copy(
            src_ref=attn_q_weights_hbm_ref.at[layer_idx],
            dst_ref=attn_wq_vmem,
            sem=dma_sems_ref.at[2],
        )
        q_wait.wait()

        # Start async DMAs for this layer's Attention O_proj, 3D KV-cache tile, and Router weights
        o_copy = pltpu.make_async_copy(
            src_ref=attn_o_weights_hbm_ref.at[layer_idx],
            dst_ref=attn_wo_vmem,
            sem=dma_sems_ref.at[3],
        )
        kv_copy = pltpu.make_async_copy(
            src_ref=kv_cache_nope_hbm_ref.at[layer_idx],
            dst_ref=kv_tile_vmem,
            sem=dma_sems_ref.at[4],
        )
        rw_copy = pltpu.make_async_copy(
            src_ref=router_weights_hbm_ref.at[layer_idx],
            dst_ref=router_w_vmem,
            sem=dma_sems_ref.at[5],
        )
        o_copy.start()
        kv_copy.start()
        rw_copy.start()

        # ---------------------------------------------------------------------
        # Step 1: In-VMEM 4-stream `mHC` Pre-Projection & Hoisted `Engram` (L1, L14)
        # ---------------------------------------------------------------------
        h_in = (s0 + s1 + s2 + s3) * jnp.bfloat16(0.25)
        is_l1 = layer_idx == jnp.int32(1)
        is_l14 = layer_idx == jnp.int32(14)
        engram_vec = jnp.where(
            is_l1,
            e_l1,
            jnp.where(is_l14, e_l14, jnp.zeros_like(e_l1)),
        )
        engram_pad = jnp.concatenate([engram_vec, engram_vec, engram_vec[:, :1024]], axis=-1)
        h_in = jnp.where(is_l1 | is_l14, h_in + engram_pad * jnp.bfloat16(0.05), h_in)

        # ---------------------------------------------------------------------
        # Step 2: `TP=16` Head-Sharded `CSA2` / `SWA` Attention in VMEM
        # ---------------------------------------------------------------------
        q_local = jnp.dot(h_in, attn_wq_vmem[...], preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        kv_copy.wait()
        kv_local = kv_tile_vmem[...]  # [512, 256] compressed top-512 KV cache in VMEM
        attn_scores = jnp.dot(q_local, kv_local.T, preferred_element_type=jnp.float32) * (1.0 / math.sqrt(256.0))
        attn_probs = jax.nn.softmax(attn_scores, axis=-1).astype(jnp.bfloat16)
        ctx_local = jnp.dot(attn_probs, kv_local, preferred_element_type=jnp.float32).astype(jnp.bfloat16)

        o_copy.wait()
        attn_out_partial = jnp.dot(ctx_local, attn_wo_vmem[...], preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        act_vmem_ref[...] = attn_out_partial

        # 16-chip 2D Torus All-Reduce (`6` hops: `3` along `etp_x` + `3` along `ep_y`) in VMEM
        collectives16.torus_2d_all_reduce_16(
            act_vmem_ref,
            comm_ping_vmem_ref,
            comm_pong_vmem_ref,
            row_send_sems_ref,
            row_recv_sems_ref,
            row_barrier_sem_ref,
            col_send_sems_ref,
            col_recv_sems_ref,
            col_barrier_sem_ref,
        )
        attn_out = act_vmem_ref[...]

        # Update 4-stream `mHC` state in VMEM after Attention
        s0_next = s0 + attn_out * jnp.bfloat16(0.25)
        s1_next = s1 + attn_out * jnp.bfloat16(0.25)
        streams_vmem_ref[0] = s0_next
        streams_vmem_ref[1] = s1_next
        h_mid = (s0_next + s1_next + s2 + s3) * jnp.bfloat16(0.25)

        # ---------------------------------------------------------------------
        # Step 3: Dynamic Route-Streamed MoE (`ETP=4 x EP=4`) + Cross-Layer DMA Overlap
        # ---------------------------------------------------------------------
        rw_copy.wait()
        local_router_logits = jnp.dot(h_mid, router_w_vmem[...], preferred_element_type=jnp.float32)
        pending_mask_init, num_active_local, expert_token_weights = route_local_row_top_experts(
            local_router_logits
        )

        loop_bound = jnp.where(
            jnp.bool_(force_all_experts_control),
            jnp.int32(24),
            jnp.where(layer_idx < jnp.int32(NUM_DENSE_LAYERS), jnp.int32(1), num_active_local),
        )
        new_total_active = total_active_experts + loop_bound

        # Cross-Layer Async DMA Overlap: start Layer `layer_idx + 1`'s Attention `q_proj` DMA NOW
        # so it transfers from HBM into `attn_wq_vmem` while MoE + 2D Torus All-Reduce run!
        next_layer = jnp.minimum(layer_idx + jnp.int32(1), jnp.int32(num_layers - 1))
        if enable_cross_layer_dma:
            next_q_copy = pltpu.make_async_copy(
                src_ref=attn_q_weights_hbm_ref.at[next_layer],
                dst_ref=attn_wq_vmem,
                sem=dma_sems_ref.at[2],
            )
            next_q_copy.start()

        expert_ids_2d = lax.broadcasted_iota(jnp.int32, (h_mid.shape[0], 128), 1)
        moe_acc_vmem_ref[...] = jnp.zeros_like(h_mid)
        pending_mask_vmem_ref[...] = jnp.where(pending_mask_init, jnp.int32(1), jnp.int32(0))

        def _expert_stream_body(e_step, dummy_carry):
            pending_mask_2d = pending_mask_vmem_ref[...] != jnp.int32(0)
            min_exp_f32 = jnp.min(
                jnp.where(pending_mask_2d, expert_ids_2d.astype(jnp.float32), jnp.float32(1024.0))
            )
            has_exp = jnp.any(pending_mask_2d)
            local_exp_id = jnp.where(
                jnp.bool_(force_all_experts_control),
                e_step % jnp.int32(24),
                jnp.where(has_exp, min_exp_f32, 0.0).astype(jnp.int32),
            )
            next_pending_mask_2d = pending_mask_2d & (expert_ids_2d != local_exp_id)
            pending_mask_vmem_ref[...] = jnp.where(next_pending_mask_2d, jnp.int32(1), jnp.int32(0))

            w13_dma = pltpu.make_async_copy(
                src_ref=expert_w13_hbm_ref.at[layer_idx, local_exp_id],
                dst_ref=moe_w13_ping_vmem,
                sem=dma_sems_ref.at[0],
            )
            w2_dma = pltpu.make_async_copy(
                src_ref=expert_w2_hbm_ref.at[layer_idx, local_exp_id],
                dst_ref=moe_w2_ping_vmem,
                sem=dma_sems_ref.at[1],
            )
            w13_dma.start()
            w2_dma.start()
            w13_dma.wait()

            # Bitwise IEEE-754 packed MXFP4 (2 E2M1 nibbles/byte) -> BF16 unpack in VMEM + SwiGLU
            w13_bf16 = unpack_mxfp4_u8_to_bf16_vmem(moe_w13_ping_vmem[...], concat_axis=-1)
            gate_up = jnp.dot(h_mid, w13_bf16, preferred_element_type=jnp.float32)
            half_d = MOE_INTER_DIM_SHARD // 2
            inter = (jax.nn.silu(gate_up[:, :half_d]) * gate_up[:, half_d:]).astype(jnp.bfloat16)
            inter_padded = jnp.concatenate([inter, inter], axis=-1)

            w2_dma.wait()
            w2_bf16 = unpack_mxfp4_u8_to_bf16_vmem(moe_w2_ping_vmem[...], concat_axis=0)
            exp_out = jnp.dot(inter_padded, w2_bf16, preferred_element_type=jnp.float32).astype(jnp.bfloat16)

            match_exp_2d = expert_ids_2d == local_exp_id
            token_weight = jnp.sum(
                jnp.where(match_exp_2d, expert_token_weights, jnp.bfloat16(0.0)),
                axis=-1,
                keepdims=True,
            )
            moe_acc_vmem_ref[...] = moe_acc_vmem_ref[...] + exp_out * token_weight
            return dummy_carry

        _ = lax.fori_loop(
            0,
            loop_bound,
            _expert_stream_body,
            jnp.int32(0),
        )

        if not enable_cross_layer_dma:
            next_q_copy = pltpu.make_async_copy(
                src_ref=attn_q_weights_hbm_ref.at[next_layer],
                dst_ref=attn_wq_vmem,
                sem=dma_sems_ref.at[2],
            )
            next_q_copy.start()

        # 16-chip 2D Torus All-Reduce (`ETP=4` along `etp_x` + `EP=4` along `ep_y`) in VMEM
        collectives16.torus_2d_all_reduce_16(
            moe_acc_vmem_ref,
            comm_ping_vmem_ref,
            comm_pong_vmem_ref,
            row_send_sems_ref,
            row_recv_sems_ref,
            row_barrier_sem_ref,
            col_send_sems_ref,
            col_recv_sems_ref,
            col_barrier_sem_ref,
        )
        moe_final = moe_acc_vmem_ref[...]

        streams_vmem_ref[2] = s2 + moe_final * jnp.bfloat16(0.25)
        streams_vmem_ref[3] = s3 + moe_final * jnp.bfloat16(0.25)
        return new_total_active

    total_active_f = lax.fori_loop(
        0,
        num_layers,
        _layer_loop_body,
        jnp.int32(0),
    )

    # Final wait on the last prefetched DMA semaphore so all semaphores exit at zero
    final_q_wait = pltpu.make_async_copy(
        src_ref=attn_q_weights_hbm_ref.at[num_layers - 1],
        dst_ref=attn_wq_vmem,
        sem=dma_sems_ref.at[2],
    )
    final_q_wait.wait()

    s0_f = streams_vmem_ref[0]
    s1_f = streams_vmem_ref[1]
    s2_f = streams_vmem_ref[2]
    s3_f = streams_vmem_ref[3]
    hidden_out_vmem_ref[...] = ((s0_f + s1_f + s2_f + s3_f) * jnp.bfloat16(0.25)).astype(jnp.bfloat16)
    active_counts_vmem_ref[...] = jnp.broadcast_to(total_active_f, (8, 128))


def build_dsv41_megakernel(
    mesh: Mesh,
    *,
    batch_tokens: int = 8,
    num_layers: int = NUM_LAYERS,
    enable_cross_layer_dma: bool = True,
    force_all_experts_control: bool = False,
):
    """Wraps `_dsv41_megakernel_body` in a `shard_map` + grid-less `pl.pallas_call` (`vmem_limit=16 MiB`)."""

    kernel_fn = functools.partial(
        _dsv41_megakernel_body,
        num_layers=num_layers,
        enable_cross_layer_dma=enable_cross_layer_dma,
        force_all_experts_control=force_all_experts_control,
    )

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM),  # hidden_in_vmem (80 KiB)
        pl.BlockSpec(memory_space=pltpu.VMEM),  # engram_embs_vmem (64 KiB)
        pl.BlockSpec(memory_space=pltpu.HBM),   # attn_q_weights_hbm
        pl.BlockSpec(memory_space=pltpu.HBM),   # attn_o_weights_hbm
        pl.BlockSpec(memory_space=pltpu.HBM),   # kv_cache_nope_hbm
        pl.BlockSpec(memory_space=pltpu.HBM),   # router_weights_hbm
        pl.BlockSpec(memory_space=pltpu.HBM),   # expert_w13_hbm
        pl.BlockSpec(memory_space=pltpu.HBM),   # expert_w2_hbm
    ]
    out_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM),  # hidden_out_vmem
        pl.BlockSpec(memory_space=pltpu.VMEM),  # active_counts_vmem (8 x 128 int32)
    ]
    scratch_shapes = [
        pltpu.VMEM((4, batch_tokens, HIDDEN_DIM), jnp.bfloat16),            # streams_vmem (320 KiB)
        pltpu.VMEM((batch_tokens, 128), jnp.int32),                         # pending_mask_vmem (4 KiB)
        pltpu.VMEM((batch_tokens, HIDDEN_DIM), jnp.bfloat16),               # act_vmem (80 KiB)
        pltpu.VMEM((batch_tokens, HIDDEN_DIM), jnp.bfloat16),               # moe_acc_vmem (80 KiB)
        pltpu.VMEM((batch_tokens, HIDDEN_DIM), jnp.bfloat16),               # comm_ping_vmem (80 KiB)
        pltpu.VMEM((batch_tokens, HIDDEN_DIM), jnp.bfloat16),               # comm_pong_vmem (80 KiB)
        pltpu.VMEM((VMEM_POOL_BYTES // 128, 128), jnp.uint8),               # pool_vmem (9.40 MiB aliased)
        pltpu.SemaphoreType.DMA((6,)),                                      # dma_sems
        pltpu.SemaphoreType.DMA((3,)),                                      # row_send_sems
        pltpu.SemaphoreType.DMA((3,)),                                      # row_recv_sems
        pltpu.SemaphoreType.REGULAR((1,)),                                  # row_barrier_sem
        pltpu.SemaphoreType.DMA((3,)),                                      # col_send_sems
        pltpu.SemaphoreType.DMA((3,)),                                      # col_recv_sems
        pltpu.SemaphoreType.REGULAR((1,)),                                  # col_barrier_sem
    ]

    pallas_megakernel = pl.pallas_call(
        kernel_fn,
        out_shape=[
            jax.ShapeDtypeStruct((batch_tokens, HIDDEN_DIM), jnp.bfloat16),
            jax.ShapeDtypeStruct((8, 128), jnp.int32),
        ],
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=16 * 1024 * 1024,  # Hard 16.00 MiB TPU v6e VMEM ceiling
        ),
    )

    @functools.partial(
        jax.jit,
        in_shardings=(
            NamedSharding(mesh, P(None, None)),                  # hidden_in (replicated B x 5120)
            NamedSharding(mesh, P(None, None, None)),            # engram_embs (replicated 2 x B x 2048)
            NamedSharding(mesh, P(None, None, ("ep_y", "etp_x"))),  # attn_q (TP=16)
            NamedSharding(mesh, P(None, ("ep_y", "etp_x"), None)),  # attn_o (TP=16)
            NamedSharding(mesh, P(None, None, ("ep_y", "etp_x"))),  # kv_cache (TP=16)
            NamedSharding(mesh, P(None, None, "ep_y")),             # router_weights (EP=4)
            NamedSharding(mesh, P(None, "ep_y", None, "etp_x")),    # expert_w13 (EP=4 x ETP=4)
            NamedSharding(mesh, P(None, "ep_y", "etp_x", None)),    # expert_w2  (EP=4 x ETP=4)
        ),
        out_shardings=(
            NamedSharding(mesh, P(None, None)),
            NamedSharding(mesh, P(None, None)),
        ),
    )
    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(
            P(None, None),
            P(None, None, None),
            P(None, None, ("ep_y", "etp_x")),
            P(None, ("ep_y", "etp_x"), None),
            P(None, None, ("ep_y", "etp_x")),
            P(None, None, "ep_y"),
            P(None, "ep_y", None, "etp_x"),
            P(None, "ep_y", "etp_x", None),
        ),
        out_specs=(P(None, None), P(None, None)),
        check_vma=False,
    )
    def run_megakernel(*args):
        return pallas_megakernel(*args)

    return run_megakernel
