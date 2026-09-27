"""Real 40-layer TPU v6e-16 Pallas decode megakernel for DeepSeek-V4.1-Flash.

Implements:
- In-kernel block-32x32 FP8 (`E4M3` + `UE8M0`) and MXFP4 (`E2M1x2` + `UE8M0`) dequantizing MXU dots.
- In-kernel `mHC` (`hc_mult=4` streams, delayed `pre_mix`, 20-step Sinkhorn-Knopp) in VMEM.
- In-kernel GPT-J interleaved RoPE (Plain `theta=10000` for layers 0..1, YaRN `theta=160000` for layers 2..39).
- In-kernel `Engram` gated cross-attention at layers 1 and 14 using TP-sharded `wkv` + `allreduce_2d`.
- In-kernel `Compressor` (`ratio=2` 2-token softmax pooling at layers 2, 8, 14; `ratio=1` at layer 20) + block-64 E8M0 roundtrip.
- In-kernel `Indexer` (`build_k` at KV sources + 16-chip head-parallel `select_topk` at index sources `[2, 8, 14, 20, 24, 28, 32, 36]`).
- In-kernel joint `SWA + CSA + Sink` attention + inverse RoPE + `wo_a` / `wo_b` + `allreduce_2d`.
- In-kernel `sqrtsoftplus` + `noaux_tc` top-6 of 384 router + deduplicated dynamic HBM->VMEM expert gather + `swiglu_limit=10.0` + shared expert + `allreduce_2d`.
- Single `pl.pallas_call` persistent 40-layer decode megakernel covering all 40 layers + final RMSNorm + LM head.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import gc
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import ml_dtypes
import numpy as np

from . import collectives16
from .config import DSV41Config
from .engine_jax import to_host_np
from .load import EngramHostTables, LayerWeights, ModelWeights

os.makedirs("/dev/shm/jax_comp_cache_mk", exist_ok=True)
jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_comp_cache_mk")


def build_pallas_constants(
    cfg: DSV41Config, batch_tile: int = 8
) -> Dict[str, np.ndarray]:
    """Build static VMEM constant matrices for RoPE (`p_swap`, `inv_freq`, `rope_sign`) and SwiGLU (`s32`)."""
    p_swap_np = np.zeros((128, 128), dtype=np.float32)
    for i in range(32):
        p_swap_np[64 + 2 * i + 1, 64 + 2 * i] = 1.0
        p_swap_np[64 + 2 * i, 64 + 2 * i + 1] = 1.0

    # Shift-32 matrix [128, 128] to move cols 32..63 to 0..31 for 160-channel SwiGLU tail (160 = 128 + 32)
    s32_np = np.zeros((128, 128), dtype=np.float32)
    for i in range(32):
        s32_np[32 + i, i] = 1.0

    dim = cfg.qk_rope_head_dim  # 64
    inv_freq_plain_32 = 1.0 / (
        cfg.rope_theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
    )

    base = cfg.compress_rope_theta
    factor = cfg.rope_factor
    orig_max = cfg.rope_original_max_pos
    beta_fast = cfg.rope_beta_fast
    beta_slow = cfg.rope_beta_slow
    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
    inv_extrap = 1.0 / pos_freqs
    inv_interp = 1.0 / (factor * pos_freqs)
    low = max(
        math.floor(
            dim * math.log(orig_max / (beta_fast * 2.0 * math.pi)) / (2.0 * math.log(base))
        ),
        0,
    )
    high = min(
        math.ceil(
            dim * math.log(orig_max / (beta_slow * 2.0 * math.pi)) / (2.0 * math.log(base))
        ),
        dim // 2 - 1,
    )
    ramp = np.clip(
        (np.arange(dim // 2, dtype=np.float32) - low) / max(high - low, 0.001),
        0.0,
        1.0,
    )
    extrap_factor = 1.0 - ramp
    inv_freq_yarn_32 = inv_interp * (1.0 - extrap_factor) + inv_extrap * extrap_factor

    inv_freq_plain_vec = np.zeros((batch_tile, 128), dtype=np.float32)
    inv_freq_yarn_vec = np.zeros((batch_tile, 128), dtype=np.float32)
    rope_sign_vec = np.zeros((batch_tile, 128), dtype=np.float32)
    for i in range(32):
        inv_freq_plain_vec[:, 64 + 2 * i] = inv_freq_plain_32[i]
        inv_freq_plain_vec[:, 64 + 2 * i + 1] = inv_freq_plain_32[i]
        inv_freq_yarn_vec[:, 64 + 2 * i] = inv_freq_yarn_32[i]
        inv_freq_yarn_vec[:, 64 + 2 * i + 1] = inv_freq_yarn_32[i]
        rope_sign_vec[:, 64 + 2 * i] = -1.0
        rope_sign_vec[:, 64 + 2 * i + 1] = 1.0

    return {
        "p_swap": p_swap_np,
        "s32": s32_np,
        "inv_freq_plain": inv_freq_plain_vec,
        "inv_freq_yarn": inv_freq_yarn_vec,
        "rope_sign": rope_sign_vec,
    }


def pack_fp8_kn(w_f8: np.ndarray, s_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transpose `w_f8` [N, K] -> [K, N] and expand `s_u8` [N//32, K//32] -> [K//32, N]."""
    w_t = np.ascontiguousarray(w_f8.T)
    s_t = np.ascontiguousarray(np.repeat(s_u8.T, 32, axis=1))
    return w_t, s_t


def pack_shared_w13(
    w1_f8: np.ndarray, s1_u8: np.ndarray, w3_f8: np.ndarray, s3_u8: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack 160-channel shared expert `w1` & `w3` [160, 5120] into `[5120, 384]` f8 and `[160, 384]` u8."""
    w13 = np.zeros((5120, 384), dtype=ml_dtypes.float8_e4m3fn)
    s13 = np.zeros((160, 384), dtype=np.uint8)
    w1_t = w1_f8.T  # [5120, 160]
    w3_t = w3_f8.T  # [5120, 160]
    s1_exp = np.repeat(s1_u8.T, 32, axis=1)  # [160, 160]
    s3_exp = np.repeat(s3_u8.T, 32, axis=1)  # [160, 160]
    w13[:, 0:128] = w1_t[:, 0:128]
    w13[:, 128:256] = w3_t[:, 0:128]
    w13[:, 256:288] = w1_t[:, 128:160]
    w13[:, 288:320] = w3_t[:, 128:160]
    s13[:, 0:128] = s1_exp[:, 0:128]
    s13[:, 128:256] = s3_exp[:, 0:128]
    s13[:, 256:288] = s1_exp[:, 128:160]
    s13[:, 288:320] = s3_exp[:, 128:160]
    return w13, s13


def pack_shared_w2(
    w2_f8: np.ndarray, s2_u8: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack 160-channel shared expert `w2` [5120, 160] into `[256, 5120]` f8 and `[8, 5120]` u8."""
    w2_p = np.zeros((256, 5120), dtype=ml_dtypes.float8_e4m3fn)
    s2_p = np.zeros((8, 5120), dtype=np.uint8)
    w2_t = w2_f8.T  # [160, 5120]
    s2_exp = np.repeat(s2_u8.T, 32, axis=1)  # [5, 5120]
    w2_p[0:160, :] = w2_t[0:160, :]
    s2_p[0:5, :] = s2_exp[0:5, :]
    return w2_p, s2_p


def pack_routed_w13(
    w1_i8: np.ndarray, s1_u8: np.ndarray, w3_i8: np.ndarray, s3_u8: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack 160-channel routed experts `w1` & `w3` [384, 160, 2560] into `[384, 640, 384]` u32 and `[384, 160, 384]` u8."""
    e_cnt = w1_i8.shape[0]
    w13_u32 = np.zeros((e_cnt, 640, 384), dtype=np.uint32)
    s13_u8 = np.zeros((e_cnt, 160, 384), dtype=np.uint8)

    def _unpack_and_repack(arr_i8: np.ndarray) -> np.ndarray:
        u8 = arr_i8.view(np.uint8)
        low = u8 & 0x0F
        high = (u8 >> 4) & 0x0F
        nib = np.stack([low, high], axis=-1).reshape(e_cnt, 160, 5120)
        nib_kn = np.transpose(nib, (0, 2, 1)).reshape(e_cnt, 640, 8, 160).astype(np.uint32)
        packed = np.zeros((e_cnt, 640, 160), dtype=np.uint32)
        for b in range(8):
            packed |= nib_kn[:, :, b, :] << np.uint32(4 * b)
        return packed

    p1 = _unpack_and_repack(w1_i8)
    p3 = _unpack_and_repack(w3_i8)
    w13_u32[:, :, 0:128] = p1[:, :, 0:128]
    w13_u32[:, :, 128:256] = p3[:, :, 0:128]
    w13_u32[:, :, 256:288] = p1[:, :, 128:160]
    w13_u32[:, :, 288:320] = p3[:, :, 128:160]

    s1_t = np.transpose(s1_u8, (0, 2, 1))  # [384, 160, 160]
    s3_t = np.transpose(s3_u8, (0, 2, 1))  # [384, 160, 160]
    s13_u8[:, :, 0:128] = s1_t[:, :, 0:128]
    s13_u8[:, :, 128:256] = s3_t[:, :, 0:128]
    s13_u8[:, :, 256:288] = s1_t[:, :, 128:160]
    s13_u8[:, :, 288:320] = s3_t[:, :, 128:160]
    return w13_u32, s13_u8


def pack_routed_w2(
    w2_i8: np.ndarray, s2_u8: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack 160-channel routed experts `w2` [384, 5120, 80] into `[384, 24, 5120]` u32 and `[384, 8, 5120]` u8."""
    e_cnt = w2_i8.shape[0]
    u8 = w2_i8.view(np.uint8)
    low = u8 & 0x0F
    high = (u8 >> 4) & 0x0F
    nib = np.stack([low, high], axis=-1).reshape(e_cnt, 5120, 160)
    nib_kn = np.transpose(nib, (0, 2, 1))  # [384, 160, 5120]

    nib_192 = np.zeros((e_cnt, 192, 5120), dtype=np.uint32)
    nib_192[:, 0:160, :] = nib_kn.astype(np.uint32)
    nib_24 = nib_192.reshape(e_cnt, 24, 8, 5120)
    w2_u32 = np.zeros((e_cnt, 24, 5120), dtype=np.uint32)
    for b in range(8):
        w2_u32 |= nib_24[:, :, b, :] << np.uint32(4 * b)

    s2_p = np.zeros((e_cnt, 8, 5120), dtype=np.uint8)
    s2_t = np.transpose(s2_u8, (0, 2, 1))  # [384, 5, 5120]
    s2_p[:, 0:5, :] = s2_t[:, 0:5, :]
    return w2_u32, s2_p


@jax.jit
def _pack_routed_w13_dev_jax(
    w1_i8: jax.Array, s1_u8: jax.Array, w3_i8: jax.Array, s3_u8: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Single-device TPU JIT repacking of `w1` & `w3` [384, 160, 2560] into `[1, 384, 640, 384]` u32 and `[1, 384, 160, 384]` u8."""
    def _repack_one(arr_i8: jax.Array) -> jax.Array:
        u32 = jax.lax.bitcast_convert_type(arr_i8, jnp.uint8).astype(jnp.uint32)
        low = u32 & jnp.uint32(0x0F)
        high = (u32 >> jnp.uint32(4)) & jnp.uint32(0x0F)
        nib = jnp.stack([low, high], axis=-1).reshape(384, 160, 5120)
        nib_kn = jnp.transpose(nib, (0, 2, 1)).reshape(384, 640, 8, 160)
        packed = jnp.zeros((384, 640, 160), dtype=jnp.uint32)
        for b in range(8):
            packed = packed | (nib_kn[:, :, b, :] << jnp.uint32(4 * b))
        return packed

    p1 = _repack_one(w1_i8)
    p3 = _repack_one(w3_i8)
    z64 = jnp.zeros((384, 640, 64), dtype=jnp.uint32)
    w13 = jnp.concatenate(
        [p1[:, :, :128], p3[:, :, :128], p1[:, :, 128:160], p3[:, :, 128:160], z64],
        axis=2,
    )
    s1_t = jnp.transpose(s1_u8, (0, 2, 1))
    s3_t = jnp.transpose(s3_u8, (0, 2, 1))
    zs64 = jnp.zeros((384, 160, 64), dtype=jnp.uint8)
    s13 = jnp.concatenate(
        [s1_t[:, :, :128], s3_t[:, :, :128], s1_t[:, :, 128:160], s3_t[:, :, 128:160], zs64],
        axis=2,
    )
    return w13[None, ...], s13[None, ...]


@jax.jit
def _pack_routed_w2_dev_jax(
    w2_i8: jax.Array, s2_u8: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Single-device TPU JIT repacking of `w2` [384, 5120, 80] into `[1, 384, 24, 5120]` u32 and `[1, 384, 8, 5120]` u8."""
    u32 = jax.lax.bitcast_convert_type(w2_i8, jnp.uint8).astype(jnp.uint32)
    low = u32 & jnp.uint32(0x0F)
    high = (u32 >> jnp.uint32(4)) & jnp.uint32(0x0F)
    nib = jnp.stack([low, high], axis=-1).reshape(384, 5120, 160)
    nib_kn = jnp.transpose(nib, (0, 2, 1))
    nib_192 = jnp.pad(nib_kn, ((0, 0), (0, 32), (0, 0))).reshape(384, 24, 8, 5120)
    w2_u32 = jnp.zeros((384, 24, 5120), dtype=jnp.uint32)
    for b in range(8):
        w2_u32 = w2_u32 | (nib_192[:, :, b, :] << jnp.uint32(4 * b))
    s2_t = jnp.transpose(s2_u8, (0, 2, 1))
    s2_p = jnp.pad(s2_t, ((0, 0), (0, 3), (0, 0)))
    return w2_u32[None, ...], s2_p[None, ...]


@jax.jit
def _stack_group5_dev_jax(
    a0: jax.Array, a1: jax.Array, a2: jax.Array, a3: jax.Array, a4: jax.Array
) -> jax.Array:
    """Stack 5 `[1, ...]` single-device arrays along axis 1 -> `[1, 5, ...]`."""
    return jnp.stack([a0[0], a1[0], a2[0], a3[0], a4[0]], axis=0)[None, ...]


def pack_hc_params(
    hc_fn: np.ndarray, hc_scale: np.ndarray, hc_base: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack `hc_fn` [24, 20480] into `[5120, 128]` f32 (cols `32*s : 32*s+24` for stream `s`) and `hc_mb` `[1, 128]` f32."""
    hc_4 = hc_fn.reshape(24, 4, 5120)
    fn_packed = np.zeros((5120, 128), dtype=np.float32)
    for s in range(4):
        fn_packed[:, s * 32 : s * 32 + 24] = hc_4[:, s, :].T
    mb = np.zeros((1, 128), dtype=np.float32)
    mb[:, 0:4] = hc_scale[0]
    mb[:, 4:8] = hc_scale[1]
    mb[:, 8:24] = hc_scale[2]
    mb[:, 64:88] = hc_base[None, :]
    return fn_packed, mb


# ============================================================================
# VMEM Pallas Primitives
# ============================================================================

def decode_e2m1_vmem(code_i4: jax.Array, dtype: Any = jnp.bfloat16) -> jax.Array:
    """Branchless IEEE-754 bit-manipulation decoding of signed 4-bit E2M1 codes in VMEM."""
    wide = code_i4.astype(jnp.int32)
    index = jax.lax.bitwise_and(wide, 7)
    sign_bit = jax.lax.bitwise_and(wide, 8) << 28
    f32_bits = jnp.where(
        index <= 1,
        (-index) & 0x3F000000,
        0x3F000000 + (index << 22),
    ) | sign_bit
    return jax.lax.bitcast_convert_type(f32_bits, jnp.float32).astype(dtype)


def ue8m0_u8_to_f32_vmem(s_u8: jax.Array) -> jax.Array:
    """Convert unsigned 8-bit E8M0 exponent bias 127 to IEEE-754 float32 in VMEM."""
    return jax.lax.bitcast_convert_type(s_u8.astype(jnp.uint32) << 23, jnp.float32)


def fp8_dot_vmem(
    act_ref: Any, w_vmem: Any, s_vmem: Any, acc_ref: Any, tile_size: int = 512
) -> None:
    """In-kernel block-32x32 FP8 dequantization + BF16 MXU dot accumulating in FP32."""
    acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
    k_dim = w_vmem.shape[0]
    n_dim = w_vmem.shape[1]
    num_blocks = tile_size // 32
    for start in range(0, k_dim, tile_size):
        w_tile = w_vmem[pl.ds(start, tile_size), :].astype(jnp.float32)
        s_tile = ue8m0_u8_to_f32_vmem(s_vmem[pl.ds(start // 32, num_blocks), :])
        w_deq = (
            (w_tile.reshape(num_blocks, 32, n_dim) * s_tile[:, None, :])
            .reshape(tile_size, n_dim)
            .astype(jnp.bfloat16)
        )
        act = act_ref[:, pl.ds(start, tile_size)].astype(jnp.bfloat16)
        acc_ref[...] += jnp.dot(act, w_deq, preferred_element_type=jnp.float32)


def mxfp4_dot_vmem(
    act_ref: Any,
    w_vmem: Any,
    s_vmem: Any,
    acc_ref: Any,
    k_dim: int,
    tile_size: int = 512,
) -> None:
    """In-kernel block-32 MXFP4 (`uint32` -> `int4` bitcast + E2M1 decode + UE8M0 scale) MXU dot."""
    acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
    n_dim = w_vmem.shape[1]
    num_blocks = tile_size // 32
    for start in range(0, k_dim, tile_size):
        packed_tile = w_vmem[pl.ds(start // 8, tile_size // 8), :]
        code_i4 = pltpu.bitcast(packed_tile, jnp.int4)
        vals = decode_e2m1_vmem(code_i4, jnp.float32)
        s_tile = ue8m0_u8_to_f32_vmem(s_vmem[pl.ds(start // 32, num_blocks), :])
        w_deq = (
            (vals.reshape(num_blocks, 32, n_dim) * s_tile[:, None, :])
            .reshape(tile_size, n_dim)
            .astype(jnp.bfloat16)
        )
        act = act_ref[:, pl.ds(start, tile_size)].astype(jnp.bfloat16)
        acc_ref[...] += jnp.dot(act, w_deq, preferred_element_type=jnp.float32)


def swiglu_384_to_256(
    gu_ref: Any, h_256_ref: Any, s32_ref: Any, limit: float = 10.0
) -> None:
    """Clamped SwiGLU on 160 intermediate channels packed in `[B_tile, 384]` -> `[B_tile, 256]` bf16."""
    b_tile = gu_ref.shape[0]
    g_main = jnp.minimum(gu_ref[:, pl.ds(0, 128)], limit)
    u_main = jnp.clip(gu_ref[:, pl.ds(128, 128)], -limit, limit)
    h_main = (jax.nn.silu(g_main) * u_main).astype(jnp.bfloat16)

    tail_blk = gu_ref[:, pl.ds(256, 128)]
    mask32 = (jax.lax.broadcasted_iota(jnp.int32, (b_tile, 128), 1) < 32).astype(
        jnp.float32
    )
    g_tail = jnp.minimum(tail_blk * mask32, limit)
    u_tail = jnp.clip(
        jnp.dot(tail_blk, s32_ref[...], precision=jax.lax.Precision.HIGHEST) * mask32,
        -limit,
        limit,
    )
    h_tail = ((jax.nn.silu(g_tail) * u_tail) * mask32).astype(jnp.bfloat16)

    h_256_ref[:, pl.ds(0, 128)] = h_main
    h_256_ref[:, pl.ds(128, 128)] = h_tail


def rms_norm_vmem(
    x_ref: Any, w_ref: Any, out_bf16_ref: Any, eps: float = 1e-20
) -> None:
    """In-kernel FP32 RMSNorm writing BF16 to `out_bf16_ref`."""
    xf = x_ref[...].astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(xf * xf, axis=1, keepdims=True) + eps)
    out_bf16_ref[...] = (xf * inv_rms * w_ref[...].astype(jnp.float32)).astype(
        jnp.bfloat16
    )


def apply_rope_128_vmem(
    blk128_f32: jax.Array,
    cos_v: jax.Array,
    sin_v: jax.Array,
    p_swap_f32: jax.Array,
    inverse: bool = False,
) -> jax.Array:
    """Apply GPT-J interleaved RoPE on trailing 64 channels of `[B_tile, 128]`."""
    swapped = jnp.dot(blk128_f32, p_swap_f32, precision=jax.lax.Precision.HIGHEST)
    if inverse:
        return (blk128_f32 * cos_v - swapped * sin_v).astype(jnp.bfloat16)
    return (blk128_f32 * cos_v + swapped * sin_v).astype(jnp.bfloat16)


def quant_e8m0_64_roundtrip_128(
    blk128_bf16: jax.Array, keep_right_half: bool = False
) -> jax.Array:
    """Block-64 FP8 E4M3 + UE8M0 roundtrip on a `[B_tile, 128]` VMEM block."""
    b_tile = blk128_bf16.shape[0]
    xf = blk128_bf16.astype(jnp.float32)
    abs_x = jnp.abs(xf)
    mask_l = jax.lax.broadcasted_iota(jnp.int32, (b_tile, 128), 1) < 64
    amax_l = jnp.maximum(
        jnp.max(jnp.where(mask_l, abs_x, 0.0), axis=1, keepdims=True), 1e-4
    )
    amax_r = jnp.maximum(
        jnp.max(jnp.where(~mask_l, abs_x, 0.0), axis=1, keepdims=True), 1e-4
    )
    amax = jnp.where(mask_l, amax_l, amax_r)
    exp = jnp.ceil(jnp.log2(amax / 448.0))
    q = jnp.clip(xf * jnp.exp2(-exp), -448.0, 448.0).astype(jnp.float8_e4m3fn)
    sc_u8 = jnp.clip(exp + 127.0, 0.0, 255.0).astype(jnp.uint8)
    s_bf16 = ue8m0_u8_to_f32_vmem(sc_u8).astype(jnp.bfloat16)
    deq = (q.astype(jnp.bfloat16) * s_bf16).astype(jnp.bfloat16)
    if keep_right_half:
        return jnp.where(mask_l, deq, blk128_bf16)
    return deq


def mhc_pre_and_post_coeffs_vmem(
    res_vmem: Any,
    prev_pm_vmem: Any,
    hc_fn_vmem: Any,
    hc_mb_vmem: Any,
    xin_out_vmem: Any,
    next_pm_out_vmem: Any,
) -> Tuple[List[jax.Array], List[List[jax.Array]]]:
    """Compute `mhc_pre_delayed` (`xin_out_vmem`, `next_pm_out_vmem`) and return `(post_list, K)` for `mhc_post`."""
    b_tile = xin_out_vmem.shape[0]
    sq_sum = jnp.zeros((b_tile, 1), jnp.float32)
    xin_acc = jnp.zeros((b_tile, 5120), jnp.float32)
    stream_mixes: List[jax.Array] = []

    for s in range(4):
        r_s = res_vmem[s].astype(jnp.float32)
        pm_s = prev_pm_vmem[s]
        xin_acc += r_s * pm_s[:, :1]
        sq_sum += jnp.mean(r_s * r_s, axis=1, keepdims=True)
        m_s = jnp.zeros((b_tile, 128), jnp.float32)
        for tile_idx in range(10):
            st = tile_idx * 512
            r_tile = res_vmem[s, :, pl.ds(st, 512)].astype(jnp.float32)
            w_tile = hc_fn_vmem[pl.ds(st, 512), :]
            m_s += jnp.dot(r_tile, w_tile, precision=jax.lax.Precision.HIGHEST)
        stream_mixes.append(m_s)

    xin_out_vmem[...] = xin_acc.astype(jnp.bfloat16)
    inv_rms = jax.lax.rsqrt(sq_sum * 0.25 + 1e-20)

    mb = hc_mb_vmem[...]
    col_iota = jax.lax.broadcasted_iota(jnp.int32, (b_tile, 128), 1)

    def _get_col_sum(c: int) -> jax.Array:
        v0 = jnp.where(col_iota == c, stream_mixes[0], 0.0)
        v1 = jnp.where(col_iota == (32 + c), stream_mixes[1], 0.0)
        v2 = jnp.where(col_iota == (64 + c), stream_mixes[2], 0.0)
        v3 = jnp.where(col_iota == (96 + c), stream_mixes[3], 0.0)
        return jnp.sum(v0 + v1 + v2 + v3, axis=1, keepdims=True) * inv_rms

    def _get_mb(c: int) -> jax.Array:
        return jnp.sum(jnp.where(col_iota == c, mb, 0.0), axis=1, keepdims=True)

    sc0 = _get_mb(0)
    sc1 = _get_mb(4)
    sc2 = _get_mb(8)

    post_list: List[jax.Array] = []
    k_mat: List[List[jax.Array]] = [[None] * 4 for _ in range(4)]  # type: ignore[list-item]
    for i in range(4):
        p_i = jax.nn.sigmoid(_get_col_sum(i) * sc0 + _get_mb(64 + i)) + 1e-6
        next_pm_out_vmem[i] = jnp.broadcast_to(p_i, (b_tile, 128))
        po_i = 2.0 * jax.nn.sigmoid(_get_col_sum(4 + i) * sc1 + _get_mb(68 + i))
        post_list.append(po_i)
        for j in range(4):
            idx = 8 + i * 4 + j
            k_mat[i][j] = _get_col_sum(idx) * sc2 + _get_mb(64 + idx)

    for i in range(4):
        m_i = jnp.maximum(
            jnp.maximum(k_mat[i][0], k_mat[i][1]),
            jnp.maximum(k_mat[i][2], k_mat[i][3]),
        )
        e0 = jnp.exp(k_mat[i][0] - m_i)
        e1 = jnp.exp(k_mat[i][1] - m_i)
        e2 = jnp.exp(k_mat[i][2] - m_i)
        e3 = jnp.exp(k_mat[i][3] - m_i)
        s_i = e0 + e1 + e2 + e3
        k_mat[i][0] = e0 / s_i + 1e-6
        k_mat[i][1] = e1 / s_i + 1e-6
        k_mat[i][2] = e2 / s_i + 1e-6
        k_mat[i][3] = e3 / s_i + 1e-6
    for j in range(4):
        c_sum = k_mat[0][j] + k_mat[1][j] + k_mat[2][j] + k_mat[3][j] + 1e-6
        for i in range(4):
            k_mat[i][j] = k_mat[i][j] / c_sum

    for _ in range(19):
        for i in range(4):
            r_sum = k_mat[i][0] + k_mat[i][1] + k_mat[i][2] + k_mat[i][3] + 1e-6
            for j in range(4):
                k_mat[i][j] = k_mat[i][j] / r_sum
        for j in range(4):
            c_sum = k_mat[0][j] + k_mat[1][j] + k_mat[2][j] + k_mat[3][j] + 1e-6
            for i in range(4):
                k_mat[i][j] = k_mat[i][j] / c_sum

    return post_list, k_mat


def mhc_post_apply_vmem(
    res_vmem: Any,
    layer_out_vmem: Any,
    post_list: List[jax.Array],
    k_mat: List[List[jax.Array]],
) -> None:
    """Apply `mhc_post` in-place onto `res_vmem` `[4, B_tile, 5120]`."""
    lout_f = layer_out_vmem[...].astype(jnp.float32)
    r0 = res_vmem[0].astype(jnp.float32)
    r1 = res_vmem[1].astype(jnp.float32)
    r2 = res_vmem[2].astype(jnp.float32)
    r3 = res_vmem[3].astype(jnp.float32)
    for j in range(4):
        res_vmem[j] = (
            k_mat[0][j] * r0
            + k_mat[1][j] * r1
            + k_mat[2][j] * r2
            + k_mat[3][j] * r3
            + post_list[j] * lout_f
        ).astype(jnp.bfloat16)


def allreduce_2d_f32(
    val_f32: jax.Array,
    scratch_bf16_ref: Any,
    local_buf: Any,
    host_buf: Any,
    sends: Any,
    recvs: Any,
    iter_base: Any,
) -> jax.Array:
    """High-precision 16-chip all-reduce decomposing `val_f32` `[B_tile, 5120]` into high + low BF16 passes."""
    hi = val_f32.astype(jnp.bfloat16)
    lo = (val_f32 - hi.astype(jnp.float32)).astype(jnp.bfloat16)
    scratch_bf16_ref[...] = hi
    collectives16.allreduce_2d(
        scratch_bf16_ref, scratch_bf16_ref, local_buf, host_buf, sends, recvs, iter_base
    )
    hi_sum = scratch_bf16_ref[...].astype(jnp.float32)
    scratch_bf16_ref[...] = lo
    collectives16.allreduce_2d(
        scratch_bf16_ref,
        scratch_bf16_ref,
        local_buf,
        host_buf,
        sends,
        recvs,
        iter_base + 1,
    )
    lo_sum = scratch_bf16_ref[...].astype(jnp.float32)
    return hi_sum + lo_sum


def pallas_engram_sublayer(
    res_vmem: Any,
    eng_rows_vmem: Any,
    eng_wkv_hbm: Any,
    eng_s_hbm: Any,
    eng_qk_hbm: Any,
    w_eng_vmem: Any,
    s_eng_vmem: Any,
    eng_qk_vmem: Any,
    out_f32_vmem: Any,
    scratch_bf16_vmem: Any,
    sem: Any,
    local_buf: Any,
    host_buf: Any,
    sends: Any,
    recvs: Any,
    iter_base: int = 0,
) -> None:
    """In-kernel Engram gated cross-attention on `res_vmem` `[4, B_tile, 5120]`."""
    c_qk = pltpu.make_async_copy(eng_qk_hbm, eng_qk_vmem, sem.at[0])
    c_qk.start()
    c_qk.wait()

    gates: List[jax.Array] = []
    for s in range(4):
        c1 = pltpu.make_async_copy(eng_wkv_hbm.at[s], w_eng_vmem, sem.at[0])
        c2 = pltpu.make_async_copy(eng_s_hbm.at[s], s_eng_vmem, sem.at[1])
        c1.start()
        c2.start()
        c1.wait()
        c2.wait()
        fp8_dot_vmem(eng_rows_vmem, w_eng_vmem, s_eng_vmem, out_f32_vmem, tile_size=384)
        key_s = allreduce_2d_f32(
            out_f32_vmem[...],
            scratch_bf16_vmem,
            local_buf,
            host_buf,
            sends,
            recvs,
            jnp.int32(iter_base + 2 * s),
        )
        hidden_s = res_vmem[s].astype(jnp.float32)
        h_rms = jax.lax.rsqrt(jnp.mean(hidden_s * hidden_s, axis=1, keepdims=True) + 1e-20)
        k_rms = jax.lax.rsqrt(jnp.mean(key_s * key_s, axis=1, keepdims=True) + 1e-20)
        qk_s = eng_qk_vmem[s]
        dot_s = (
            jnp.sum(hidden_s * qk_s * key_s, axis=1, keepdims=True)
            * h_rms
            * k_rms
            * (5120.0 ** -0.5)
        )
        g_mag = jnp.sqrt(jnp.maximum(jnp.abs(dot_s), 1e-6))
        g_s = jax.nn.sigmoid(jnp.where(dot_s < 0.0, -g_mag, g_mag))
        gates.append(g_s)

    c1 = pltpu.make_async_copy(eng_wkv_hbm.at[4], w_eng_vmem, sem.at[0])
    c2 = pltpu.make_async_copy(eng_s_hbm.at[4], s_eng_vmem, sem.at[1])
    c1.start()
    c2.start()
    c1.wait()
    c2.wait()
    fp8_dot_vmem(eng_rows_vmem, w_eng_vmem, s_eng_vmem, out_f32_vmem, tile_size=384)
    val_f32 = allreduce_2d_f32(
        out_f32_vmem[...],
        scratch_bf16_vmem,
        local_buf,
        host_buf,
        sends,
        recvs,
        jnp.int32(iter_base + 8),
    )
    for s in range(4):
        res_vmem[s] = (res_vmem[s].astype(jnp.float32) + gates[s] * val_f32).astype(
            jnp.bfloat16
        )


def pallas_attention_sublayer(
    compress_ratio: int,
    is_kv_source: bool,
    is_index_source: bool,
    x_norm_vmem: Any,
    pos_vmem: Any,
    inv_freq_vmem: Any,
    rope_sign_vmem: Any,
    p_swap_vmem: Any,
    wqa_vmem: Any,
    wqa_s_vmem: Any,
    qn_vmem: Any,
    wkv_vmem: Any,
    wkv_s_vmem: Any,
    kvn_vmem: Any,
    wqb_vmem: Any,
    wqb_s_vmem: Any,
    sink_vmem: Any,
    woa_vmem: Any,
    woa_s_vmem: Any,
    wob_vmem: Any,
    wob_s_vmem: Any,
    comp_w_vmem: Any,
    comp_n_vmem: Any,
    idx_wk_vmem: Any,
    idx_knorm_vmem: Any,
    idx_wqb_vmem: Any,
    idx_wqb_s_vmem: Any,
    idx_wproj_vmem: Any,
    swa_kv_vmem: Any,
    swa_pos_vmem: Any,
    comp_kv_vmem: Any,
    ik_f8_vmem: Any,
    ik_sc_vmem: Any,
    comp_tail_vmem: Any,
    csa_mask_vmem: Any,
    qr_f32_vmem: Any,
    qr_vmem: Any,
    kv_f32_vmem: Any,
    kv_vmem: Any,
    q_f32_vmem: Any,
    q_vmem: Any,
    o_vmem: Any,
    z_f32_vmem: Any,
    z_vmem: Any,
    out_f32_vmem: Any,
    attn_out_vmem: Any,
    lat_f32_vmem: Any,
    lat_vmem: Any,
    idx_q_f32_vmem: Any,
    local_buf: Any,
    host_buf: Any,
    sends: Any,
    recvs: Any,
    iter_base: int = 0,
    spec_k: int = 0,
) -> None:
    """In-kernel Attention + Compressor + Indexer + `wo_a` / `wo_b` + `allreduce_2d`."""
    b_tile = x_norm_vmem.shape[0]
    max_comp = comp_kv_vmem.shape[0]

    pos_i32 = pos_vmem[...].astype(jnp.int32)  # [B_tile, 128]
    pos_f32 = pos_i32.astype(jnp.float32)
    inv_f = inv_freq_vmem[...]
    r_sign = rope_sign_vmem[...]
    p_swap = p_swap_vmem[...]

    cos_v = jnp.cos(pos_f32 * inv_f)
    sin_v = jnp.sin(pos_f32 * inv_f) * r_sign

    fp8_dot_vmem(x_norm_vmem, wqa_vmem, wqa_s_vmem, qr_f32_vmem, tile_size=512)
    rms_norm_vmem(qr_f32_vmem, qn_vmem, qr_vmem)

    fp8_dot_vmem(x_norm_vmem, wkv_vmem, wkv_s_vmem, kv_f32_vmem, tile_size=512)
    rms_norm_vmem(kv_f32_vmem, kvn_vmem, kv_vmem)
    kv_vmem[:, pl.ds(384, 128)] = apply_rope_128_vmem(
        kv_vmem[:, pl.ds(384, 128)].astype(jnp.float32),
        cos_v,
        sin_v,
        p_swap,
        inverse=False,
    )

    fp8_dot_vmem(qr_vmem, wqb_vmem, wqb_s_vmem, q_f32_vmem, tile_size=256)
    q_vmem[...] = q_f32_vmem[...].astype(jnp.bfloat16)
    for h in range(8):
        off = h * 512 + 384
        q_vmem[:, pl.ds(off, 128)] = apply_rope_128_vmem(
            q_vmem[:, pl.ds(off, 128)].astype(jnp.float32),
            cos_v,
            sin_v,
            p_swap,
            inverse=False,
        )

    col_iota_128 = jax.lax.broadcasted_iota(jnp.int32, (b_tile, 128), 1)
    if spec_k <= 0:
        # Update SWA ring buffer from row 0
        row0_mask_128 = jax.lax.broadcasted_iota(jnp.int32, (b_tile, 128), 0) == 0
        p0_vec = jnp.sum(jnp.where(row0_mask_128, pos_i32, 0), axis=0, keepdims=True)  # [1, 128]
        slot_vec = p0_vec % 128

        row0_mask_512 = jax.lax.broadcasted_iota(jnp.int32, (b_tile, 512), 0) == 0
        kv_row0 = jnp.sum(
            jnp.where(row0_mask_512, kv_vmem[...].astype(jnp.float32), 0.0),
            axis=0,
            keepdims=True,
        ).astype(jnp.bfloat16)
        slot_512 = (
            jnp.sum(
                jnp.where(row0_mask_512, jnp.broadcast_to(pos_i32[:, :1], (b_tile, 512)), 0),
                axis=0,
                keepdims=True,
            )
            % 128
        )
        slot_iota = jax.lax.broadcasted_iota(jnp.int32, (128, 512), 0)
        swa_kv_vmem[...] = jnp.where(slot_iota == slot_512, kv_row0, swa_kv_vmem[...])
        swa_pos_vmem[...] = jnp.where(col_iota_128 == slot_vec, p0_vec, swa_pos_vmem[...])

    if is_kv_source and compress_ratio > 0:
        cur_score = jnp.zeros((b_tile, 1024), jnp.float32)
        for start in range(0, 5120, 512):
            act = x_norm_vmem[:, pl.ds(start, 512)]
            wt = comp_w_vmem[pl.ds(start, 512), :]
            cur_score += jnp.dot(act, wt, preferred_element_type=jnp.float32)

        if compress_ratio == 2:
            if spec_k > 1:
                row0_1024 = jax.lax.broadcasted_iota(jnp.int32, (b_tile, 1024), 0) == 0
                tail_row0 = jnp.sum(
                    jnp.where(row0_1024, comp_tail_vmem[...], 0.0),
                    axis=0,
                    keepdims=True,
                )
                tail_before = jnp.broadcast_to(tail_row0, (b_tile, 1024))
                row_iota_1024 = jax.lax.broadcasted_iota(jnp.int32, (b_tile, 1024), 0)
                for r_ct in range(1, spec_k):
                    prev_score_r = jnp.sum(
                        jnp.where(row_iota_1024 == (r_ct - 1), cur_score, 0.0),
                        axis=0,
                        keepdims=True,
                    )
                    tail_before = jnp.where(
                        row_iota_1024 == r_ct,
                        jnp.broadcast_to(prev_score_r, (b_tile, 1024)),
                        tail_before,
                    )
            else:
                tail_before = comp_tail_vmem[...]
            v0 = tail_before[:, :512]
            g0 = tail_before[:, 512:]
            v1 = cur_score[:, :512]
            g1 = cur_score[:, 512:]
            mg = jnp.maximum(g0, g1)
            e0 = jnp.exp(g0 - mg)
            e1 = jnp.exp(g1 - mg)
            sg = e0 + e1
            lat_f32_vmem[...] = v0 * (e0 / sg) + v1 * (e1 / sg)
            c_slot_i32 = pos_i32 // 2
            c_pos_f32 = (c_slot_i32 * 2).astype(jnp.float32)
            is_odd_i32 = jnp.where(pos_i32 >= 0, pos_i32 % 2, jnp.int32(0))
            keep_tail_i32 = jnp.where(pos_i32 >= 0, pos_i32 % 2, jnp.int32(1))
            keep_tail_1024 = jnp.broadcast_to(keep_tail_i32[:, :1], (b_tile, 1024)) != 0
            comp_tail_vmem[...] = jnp.where(keep_tail_1024, tail_before, cur_score)
        else:
            lat_f32_vmem[...] = cur_score[:, :512]
            c_slot_i32 = pos_i32
            c_pos_f32 = pos_i32.astype(jnp.float32)
            is_odd_i32 = jnp.where(pos_i32 >= 0, jnp.int32(1), jnp.int32(0))

        rms_norm_vmem(lat_f32_vmem, comp_n_vmem, lat_vmem)
        cos_c = jnp.cos(c_pos_f32 * inv_f)
        sin_c = jnp.sin(c_pos_f32 * inv_f) * r_sign

        # Build indexer K (`new_ik_f8`, `new_ik_sc`) from unrotated `lat_vmem`
        k_proj = jnp.dot(
            lat_vmem[...], idx_wk_vmem[...], preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
        k_f32 = k_proj.astype(jnp.float32)
        k_inv_rms = jax.lax.rsqrt(jnp.mean(k_f32 * k_f32, axis=1, keepdims=True) + 1e-20)
        k_normed = (k_f32 * k_inv_rms * idx_knorm_vmem[...].astype(jnp.float32)).astype(
            jnp.bfloat16
        )
        k_rot = apply_rope_128_vmem(
            k_normed.astype(jnp.float32), cos_c, sin_c, p_swap, inverse=False
        )
        k_abs = jnp.abs(k_rot.astype(jnp.float32))
        k_amax = jnp.maximum(jnp.max(k_abs, axis=1, keepdims=True), 1e-4)
        k_exp = jnp.ceil(jnp.log2(k_amax / 448.0))
        new_ik_f8 = jnp.clip(
            k_rot.astype(jnp.float32) * jnp.exp2(-k_exp), -448.0, 448.0
        ).astype(jnp.float8_e4m3fn)
        k_sc_u8 = jnp.clip(k_exp + 127.0, 0.0, 255.0).astype(jnp.uint8)
        new_ik_sc = ue8m0_u8_to_f32_vmem(k_sc_u8)  # [B_tile, 1]

        # Rotate `lat_vmem` and apply block-64 E8M0 roundtrip on `0..448`
        lat_vmem[:, pl.ds(384, 128)] = apply_rope_128_vmem(
            lat_vmem[:, pl.ds(384, 128)].astype(jnp.float32),
            cos_c,
            sin_c,
            p_swap,
            inverse=False,
        )
        for b128 in range(4):
            blk = lat_vmem[:, pl.ds(b128 * 128, 128)]
            lat_vmem[:, pl.ds(b128 * 128, 128)] = quant_e8m0_64_roundtrip_128(
                blk, keep_right_half=(b128 == 3)
            )

        comp_iota_512 = jax.lax.broadcasted_iota(jnp.int32, (max_comp, 512), 0)
        comp_iota_128 = jax.lax.broadcasted_iota(jnp.int32, (max_comp, 128), 0)
        c_col_iota = jax.lax.broadcasted_iota(jnp.int32, (b_tile, max_comp), 1)

        for r_kv in range(max(1, spec_k)):
            row_r_mask_128 = jax.lax.broadcasted_iota(jnp.int32, (b_tile, 128), 0) == r_kv
            row_r_mask_512 = jax.lax.broadcasted_iota(jnp.int32, (b_tile, 512), 0) == r_kv
            do_write_0_i32 = jnp.sum(
                jnp.where(row_r_mask_128, is_odd_i32, 0),
                axis=0,
                keepdims=True,
            )  # [1, 128]
            c_slot_0_128 = jnp.sum(
                jnp.where(row_r_mask_128, c_slot_i32, 0), axis=0, keepdims=True
            )  # [1, 128]
            c_slot_0_512 = jnp.sum(
                jnp.where(row_r_mask_512, jnp.broadcast_to(c_slot_i32[:, :1], (b_tile, 512)), 0),
                axis=0,
                keepdims=True,
            )  # [1, 512]
            do_write_512_i32 = jnp.sum(
                jnp.where(
                    row_r_mask_512,
                    jnp.broadcast_to(is_odd_i32[:, :1], (b_tile, 512)),
                    0,
                ),
                axis=0,
                keepdims=True,
            )  # [1, 512]

            target_slot_512 = jnp.where(
                do_write_512_i32 != 0, c_slot_0_512, jnp.int32(-1)
            )
            comp_row0 = jnp.sum(
                jnp.where(row_r_mask_512, lat_vmem[...].astype(jnp.float32), 0.0),
                axis=0,
                keepdims=True,
            ).astype(jnp.bfloat16)
            comp_kv_vmem[...] = jnp.where(
                comp_iota_512 == target_slot_512,
                comp_row0,
                comp_kv_vmem[...],
            )

            target_slot_128 = jnp.where(
                do_write_0_i32 != 0, c_slot_0_128, jnp.int32(-1)
            )
            ik_f8_row0 = jnp.sum(
                jnp.where(row_r_mask_128, new_ik_f8.astype(jnp.float32), 0.0),
                axis=0,
                keepdims=True,
            ).astype(jnp.float8_e4m3fn)
            ik_f8_vmem[...] = jnp.where(
                comp_iota_128 == target_slot_128,
                ik_f8_row0,
                ik_f8_vmem[...],
            )

            ik_sc_0 = jnp.sum(
                jnp.where(row_r_mask_128, jnp.broadcast_to(new_ik_sc, (b_tile, 128)), 0.0),
                axis=0,
                keepdims=True,
            )[:, :1]  # [1, 1]
            target_slot_mc = jnp.broadcast_to(target_slot_128[:, :1], (b_tile, max_comp))
            ik_sc_vmem[...] = jnp.where(
                c_col_iota == target_slot_mc,
                jnp.broadcast_to(ik_sc_0, (b_tile, max_comp)),
                ik_sc_vmem[...],
            )

    if is_index_source and compress_ratio > 0:
        avail_i32 = jnp.broadcast_to(
            (pos_i32[:, :1] + 1) // compress_ratio, (b_tile, max_comp)
        )
        c_col_iota = jax.lax.broadcasted_iota(jnp.int32, (b_tile, max_comp), 1)
        valid_c = c_col_iota < avail_i32

        if max_comp <= 512:
            csa_mask_vmem[...] = jnp.where(valid_c, jnp.int32(1), jnp.int32(0))
        else:
            fp8_dot_vmem(
                qr_vmem, idx_wqb_vmem, idx_wqb_s_vmem, idx_q_f32_vmem, tile_size=256
            )
            w_proj = jnp.zeros((b_tile, 128), jnp.float32)
            for st in range(0, 5120, 512):
                act = x_norm_vmem[:, pl.ds(st, 512)].astype(jnp.bfloat16)
                wt = idx_wproj_vmem[pl.ds(st, 512), :]
                w_proj += jnp.dot(act, wt, preferred_element_type=jnp.float32)
            w_proj = w_proj.astype(jnp.bfloat16).astype(jnp.float32) * (32.0 ** -0.5)

            out_f32_vmem[...] = jnp.zeros((b_tile, 5120), jnp.float32)
            for local_h in range(2):
                qh_bf16 = idx_q_f32_vmem[:, pl.ds(local_h * 128, 128)].astype(
                    jnp.bfloat16
                )
                qh_rot = apply_rope_128_vmem(
                    qh_bf16.astype(jnp.float32), cos_v, sin_v, p_swap, inverse=False
                )
                qf = qh_rot.astype(jnp.float32)
                q_amax = jnp.maximum(jnp.max(jnp.abs(qf), axis=1, keepdims=True), 1e-4)
                q_exp = jnp.ceil(jnp.log2(q_amax / 448.0))
                q_f8 = jnp.clip(qf * jnp.exp2(-q_exp), -448.0, 448.0).astype(
                    jnp.float8_e4m3fn
                ).astype(jnp.float32)
                q_sc_u8 = jnp.clip(q_exp + 127.0, 0.0, 255.0).astype(jnp.uint8)
                q_sc = ue8m0_u8_to_f32_vmem(q_sc_u8)
                w_h = (
                    jnp.sum(
                        jnp.where(col_iota_128 == local_h, w_proj, 0.0),
                        axis=1,
                        keepdims=True,
                    )
                    * q_sc
                    * (128.0 ** -0.5)
                )
                for c_blk in range(max_comp // 128):
                    ik_tile = ik_f8_vmem[pl.ds(c_blk * 128, 128), :].astype(
                        jnp.float32
                    )
                    ik_sc_tile = ik_sc_vmem[:, pl.ds(c_blk * 128, 128)]
                    dots = jax.lax.dot_general(
                        q_f8,
                        ik_tile,
                        (((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    out_f32_vmem[:, pl.ds(c_blk * 128, 128)] += (
                        jax.nn.relu(dots * ik_sc_tile) * w_h
                    )

            total_scores_5120 = allreduce_2d_f32(
                out_f32_vmem[...],
                attn_out_vmem,
                local_buf,
                host_buf,
                sends,
                recvs,
                jnp.int32(iter_base),
            )
            scores = total_scores_5120[:, :max_comp]
            adj_scores = scores - (c_col_iota.astype(jnp.float32) / float(max_comp)) * 1e-6
            safe_s = jnp.where(valid_c, adj_scores, 0.0)
            lo = jnp.min(safe_s, axis=1, keepdims=True) - 1.0
            hi = jnp.max(safe_s, axis=1, keepdims=True) + 1.0
            masked_s = jnp.where(valid_c, adj_scores, -jnp.inf)
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                mid_b = jnp.broadcast_to(mid, (b_tile, max_comp))
                cnt = jnp.sum(
                    jnp.where(masked_s >= mid_b, jnp.int32(1), jnp.int32(0)),
                    axis=1,
                    keepdims=True,
                )
                lo = jnp.where(cnt > 512, mid, lo)
                hi = jnp.where(cnt > 512, hi, mid)
            hi_b = jnp.broadcast_to(hi, (b_tile, max_comp))
            thresh_b = jnp.where(avail_i32 <= 512, -1e30, hi_b)
            csa_mask_vmem[...] = jnp.where(
                masked_s >= thresh_b, jnp.int32(1), jnp.int32(0)
            )

    # Joint SWA + CSA + Sink attention
    sm_scale = 512.0 ** -0.5
    swa_kv_f = swa_kv_vmem[...].astype(jnp.float32)
    swa_p = swa_pos_vmem[...]
    swa_mask = (
        (swa_p >= 0) & (swa_p <= pos_i32) & (swa_p >= jnp.maximum(0, pos_i32 - 127))
    )
    slot_iota = jax.lax.broadcasted_iota(jnp.int32, (128, 512), 0)
    spec_swa_states = []
    if spec_k > 0:
        swa_kv_cur = swa_kv_f
        swa_p_cur = swa_p
        for r_kv in range(spec_k):
            row_r_128 = jax.lax.broadcasted_iota(jnp.int32, (b_tile, 128), 0) == r_kv
            row_r_512 = jax.lax.broadcasted_iota(jnp.int32, (b_tile, 512), 0) == r_kv
            pr_vec = jnp.sum(jnp.where(row_r_128, pos_i32, 0), axis=0, keepdims=True)
            slot_vec_r = jnp.where(pr_vec >= 0, pr_vec % 128, jnp.int32(-1))
            kv_row_r = jnp.sum(
                jnp.where(row_r_512, kv_vmem[...].astype(jnp.float32), 0.0),
                axis=0,
                keepdims=True,
            )
            pr_512 = jnp.sum(
                jnp.where(row_r_512, jnp.broadcast_to(pos_i32[:, :1], (b_tile, 512)), 0),
                axis=0,
                keepdims=True,
            )
            slot_512_r = jnp.where(pr_512 >= 0, pr_512 % 128, jnp.int32(-1))
            swa_kv_cur = jnp.where(slot_iota == slot_512_r, kv_row_r, swa_kv_cur)
            swa_p_cur = jnp.where(col_iota_128 == slot_vec_r, pr_vec, swa_p_cur)
            mask_cur = (
                (swa_p_cur >= 0)
                & (swa_p_cur <= pos_i32)
                & (swa_p_cur >= jnp.maximum(0, pos_i32 - 127))
            )
            spec_swa_states.append((row_r_128, row_r_512, swa_kv_cur, mask_cur))
        swa_kv_vmem[...] = swa_kv_cur.astype(jnp.bfloat16)

    for h in range(8):
        qh = q_vmem[:, pl.ds(h * 512, 512)].astype(jnp.float32)
        if spec_k <= 0:
            swa_l = (
                jax.lax.dot_general(
                    qh,
                    swa_kv_f,
                    (((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                * sm_scale
            )
            swa_l = jnp.where(swa_mask, swa_l, -jnp.inf)
        else:
            swa_l = jnp.full((b_tile, 128), -jnp.inf, dtype=jnp.float32)
            for row_r_128, _, kv_snap, mask_cur in spec_swa_states:
                l_cur = (
                    jax.lax.dot_general(
                        qh,
                        kv_snap,
                        (((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    * sm_scale
                )
                l_cur = jnp.where(mask_cur, l_cur, -jnp.inf)
                swa_l = jnp.where(row_r_128, l_cur, swa_l)

        sink_h = jnp.sum(
            jnp.where(col_iota_128 == h, sink_vmem[...], 0.0), axis=1, keepdims=True
        )
        m_h = jnp.maximum(jnp.max(swa_l, axis=1, keepdims=True), sink_h)

        if compress_ratio > 0:
            for c_blk in range(max_comp // 128):
                c_tile = comp_kv_vmem[pl.ds(c_blk * 128, 128), :].astype(jnp.float32)
                csa_l = (
                    jax.lax.dot_general(
                        qh,
                        c_tile,
                        (((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    * sm_scale
                )
                c_valid = csa_mask_vmem[:, pl.ds(c_blk * 128, 128)] != 0
                csa_l = jnp.where(c_valid, csa_l, -jnp.inf)
                m_h = jnp.maximum(m_h, jnp.max(csa_l, axis=1, keepdims=True))

        w_swa = jnp.exp(swa_l - m_h)
        denom = jnp.sum(w_swa, axis=1, keepdims=True) + jnp.exp(sink_h - m_h)
        if spec_k <= 0:
            num = jnp.dot(w_swa, swa_kv_f, precision=jax.lax.Precision.HIGHEST)
        else:
            num = jnp.zeros((b_tile, 512), dtype=jnp.float32)
            for _, row_r_512, kv_snap, _ in spec_swa_states:
                num_cur = jnp.dot(w_swa, kv_snap, precision=jax.lax.Precision.HIGHEST)
                num = jnp.where(row_r_512, num_cur, num)

        if compress_ratio > 0:
            for c_blk in range(max_comp // 128):
                c_tile = comp_kv_vmem[pl.ds(c_blk * 128, 128), :].astype(jnp.float32)
                csa_l = (
                    jax.lax.dot_general(
                        qh,
                        c_tile,
                        (((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    * sm_scale
                )
                c_valid = csa_mask_vmem[:, pl.ds(c_blk * 128, 128)] != 0
                csa_l = jnp.where(c_valid, csa_l, -jnp.inf)
                w_csa = jnp.exp(csa_l - m_h)
                denom += jnp.sum(w_csa, axis=1, keepdims=True)
                num += jnp.dot(w_csa, c_tile, precision=jax.lax.Precision.HIGHEST)

        oh = (num / denom).astype(jnp.bfloat16)
        o_vmem[:, pl.ds(h * 512, 512)] = oh
        off = h * 512 + 384
        o_vmem[:, pl.ds(off, 128)] = apply_rope_128_vmem(
            o_vmem[:, pl.ds(off, 128)].astype(jnp.float32),
            cos_v,
            sin_v,
            p_swap,
            inverse=True,
        )

    fp8_dot_vmem(o_vmem, woa_vmem, woa_s_vmem, z_f32_vmem, tile_size=512)
    z_vmem[...] = z_f32_vmem[...].astype(jnp.bfloat16)
    fp8_dot_vmem(z_vmem, wob_vmem, wob_s_vmem, out_f32_vmem, tile_size=512)
    attn_out_vmem[...] = out_f32_vmem[...].astype(jnp.bfloat16)
    collectives16.allreduce_2d(
        attn_out_vmem,
        attn_out_vmem,
        local_buf,
        host_buf,
        sends,
        recvs,
        jnp.int32(iter_base + 1),
    )


def pallas_moe_sublayer(
    active_b: int,
    x_norm_vmem: Any,
    gate_w_vmem: Any,
    gate_b_vmem: Any,
    s32_vmem: Any,
    sw13_vmem: Any,
    ss13_vmem: Any,
    sw2_vmem: Any,
    ss2_vmem: Any,
    rw13_hbm_layer: Any,
    rs13_hbm_layer: Any,
    rw2_hbm_layer: Any,
    rs2_hbm_layer: Any,
    w13_vmem: Any,
    s13_vmem: Any,
    w2_vmem: Any,
    s2_vmem: Any,
    gu_acc_vmem: Any,
    h256_vmem: Any,
    down_acc_vmem: Any,
    routed_acc_vmem: Any,
    sel_exp_vmem: Any,
    sel_prob_vmem: Any,
    proc_vmem: Any,
    moe_out_vmem: Any,
    sem: Any,
    local_buf: Any,
    host_buf: Any,
    sends: Any,
    recvs: Any,
    iter_base: Any = 0,
    sub_idx: Optional[Any] = None,
    pos_vmem: Optional[Any] = None,
) -> None:
    """In-kernel MoE sublayer: `sqrtsoftplus` top-6 router + shared expert + deduplicated routed experts + `allreduce_2d`."""
    b_tile = x_norm_vmem.shape[0]
    logits = jnp.zeros((b_tile, 384), jnp.float32)
    for start in range(0, 5120, 512):
        x_t = x_norm_vmem[:, pl.ds(start, 512)]
        w_t = gate_w_vmem[pl.ds(start, 512), :]
        logits += jnp.dot(x_t, w_t, preferred_element_type=jnp.float32)
    scores = jnp.sqrt(
        jnp.log1p(jnp.exp(-jnp.abs(logits))) + jnp.maximum(logits, 0.0)
    )
    biased = scores + gate_b_vmem[...]

    expert_ids = jax.lax.broadcasted_iota(jnp.int32, (b_tile, 384), 1)
    slot_ids = jax.lax.broadcasted_iota(jnp.int32, (b_tile, 128), 1)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, (b_tile, 128), 0)
    sel_exp = jnp.zeros((b_tile, 128), jnp.int32)
    sel_prob = jnp.zeros((b_tile, 128), jnp.float32)
    prob_tot = jnp.zeros((b_tile, 1), jnp.float32)
    for k in range(6):
        winners = jnp.argmax(biased, axis=1).astype(jnp.int32)
        w_mask = expert_ids == winners[:, None]
        probs = jnp.sum(jnp.where(w_mask, scores, 0.0), axis=1, keepdims=True)
        prob_tot += probs
        biased = jnp.where(w_mask, -jnp.inf, biased)
        k_mask = slot_ids == k
        sel_exp = jnp.where(k_mask, winners[:, None], sel_exp)
        sel_prob = jnp.where(k_mask, probs, sel_prob)

    sel_prob = (
        ((sel_prob / (prob_tot + 1e-20)) * 1.5).astype(jnp.bfloat16).astype(jnp.float32)
    )
    sel_exp_vmem[...] = sel_exp
    sel_prob_vmem[...] = sel_prob

    # Shared expert
    fp8_dot_vmem(x_norm_vmem, sw13_vmem, ss13_vmem, gu_acc_vmem, tile_size=512)
    swiglu_384_to_256(gu_acc_vmem, h256_vmem, s32_vmem, limit=10.0)
    fp8_dot_vmem(h256_vmem, sw2_vmem, ss2_vmem, down_acc_vmem, tile_size=256)
    moe_out_vmem[...] = down_acc_vmem[...].astype(jnp.bfloat16)
    collectives16.allreduce_2d(
        moe_out_vmem,
        moe_out_vmem,
        local_buf,
        host_buf,
        sends,
        recvs,
        jnp.int32(iter_base),
    )

    # Routed experts (deduplicated across active batch rows)
    w2_vmem[...] = jnp.zeros(w2_vmem.shape, jnp.uint32)
    routed_acc_vmem[...] = jnp.zeros((b_tile, 5120), jnp.float32)
    proc_vmem[...] = jnp.zeros((b_tile, 128), jnp.int32)

    if pos_vmem is not None and active_b <= 0:
        valid_slots = (pos_vmem[...] >= 0) & (slot_ids < 6)
        num_active = jnp.sum(
            jnp.where((pos_vmem[...] >= 0) & (slot_ids == 0), jnp.int32(1), jnp.int32(0))
        )
        max_routes = num_active * 6
    else:
        valid_slots = (row_ids < active_b) & (slot_ids < 6)
        max_routes = active_b * 6

    @pl.loop(0, max_routes)
    def _expert_loop(_):
        proc = proc_vmem[...] != 0
        pending = valid_slots & ~proc
        has_any = jnp.any(pending)

        @pl.when(has_any)
        def _do_one():
            min_e = jnp.min(
                jnp.where(pending, sel_exp_vmem[...].astype(jnp.float32), jnp.inf)
            )
            e = jnp.where(has_any, min_e, 0.0).astype(jnp.int32)
            match_mask = pending & (sel_exp_vmem[...] == e)

            if sub_idx is None:
                c1 = pltpu.make_async_copy(rw13_hbm_layer.at[e], w13_vmem, sem.at[0])
                c2 = pltpu.make_async_copy(rs13_hbm_layer.at[e], s13_vmem, sem.at[1])
                c3 = pltpu.make_async_copy(
                    rw2_hbm_layer.at[e], w2_vmem.at[pl.ds(0, 24), :], sem.at[2]
                )
                c4 = pltpu.make_async_copy(rs2_hbm_layer.at[e], s2_vmem, sem.at[3])
            else:
                c1 = pltpu.make_async_copy(rw13_hbm_layer.at[sub_idx, e], w13_vmem, sem.at[0])
                c2 = pltpu.make_async_copy(rs13_hbm_layer.at[sub_idx, e], s13_vmem, sem.at[1])
                c3 = pltpu.make_async_copy(
                    rw2_hbm_layer.at[sub_idx, e], w2_vmem.at[pl.ds(0, 24), :], sem.at[2]
                )
                c4 = pltpu.make_async_copy(rs2_hbm_layer.at[sub_idx, e], s2_vmem, sem.at[3])
            c1.start()
            c2.start()
            c3.start()
            c4.start()
            c1.wait()
            c2.wait()
            mxfp4_dot_vmem(
                x_norm_vmem, w13_vmem, s13_vmem, gu_acc_vmem, k_dim=5120, tile_size=512
            )
            swiglu_384_to_256(gu_acc_vmem, h256_vmem, s32_vmem, limit=10.0)
            c3.wait()
            c4.wait()
            mxfp4_dot_vmem(
                h256_vmem, w2_vmem, s2_vmem, down_acc_vmem, k_dim=256, tile_size=256
            )

            row_w = jnp.sum(
                jnp.where(match_mask, sel_prob_vmem[...], 0.0), axis=1, keepdims=True
            )
            y_bf16 = down_acc_vmem[...].astype(jnp.bfloat16).astype(jnp.float32)
            routed_acc_vmem[...] += y_bf16 * row_w
            proc_vmem[...] = jnp.where(match_mask, jnp.int32(1), proc_vmem[...])

    shared_total = moe_out_vmem[...]
    moe_out_vmem[...] = routed_acc_vmem[...].astype(jnp.bfloat16)
    collectives16.allreduce_2d(
        moe_out_vmem,
        moe_out_vmem,
        local_buf,
        host_buf,
        sends,
        recvs,
        jnp.int32(iter_base + 1),
    )
    moe_out_vmem[...] = (shared_total + moe_out_vmem[...]).astype(jnp.bfloat16)


@dataclass
class PallasLayerWeights:
    """Single-layer sharded weight container in Pallas VMEM/HBM layout."""
    layer_id: int
    compress_ratio: int
    is_kv_source: bool
    is_index_source: bool
    has_engram: bool

    attn_norm: jax.Array        # [B_tile, 5120] bf16
    ffn_norm: jax.Array         # [B_tile, 5120] bf16
    hc_attn_fn: jax.Array       # [5120, 128] f32
    hc_attn_mb: jax.Array       # [B_tile, 128] f32
    hc_ffn_fn: jax.Array        # [5120, 128] f32
    hc_ffn_mb: jax.Array        # [B_tile, 128] f32

    wqa: jax.Array              # [5120, 1280] f8
    wqa_s: jax.Array            # [160, 1280] u8
    qn: jax.Array               # [B_tile, 1280] bf16
    wkv: jax.Array              # [5120, 512] f8
    wkv_s: jax.Array            # [160, 512] u8
    kvn: jax.Array              # [B_tile, 512] bf16

    wqb: jax.Array              # P("tp", None, None): per-chip [1280, 4096] f8
    wqb_s: jax.Array            # P("tp", None, None): per-chip [40, 4096] u8
    sink: jax.Array             # P("tp", None, None): per-chip [B_tile, 128] f32

    woa: jax.Array              # P("tp", None, None): per-chip [4096, 512] f8
    woa_s: jax.Array            # P("tp", None, None): per-chip [128, 512] u8
    wob: jax.Array              # P("tp", None, None): per-chip [512, 5120] f8
    wob_s: jax.Array            # P("tp", None, None): per-chip [16, 5120] u8

    gate_w: jax.Array           # [5120, 384] bf16
    gate_b: jax.Array           # [B_tile, 384] f32
    sw13: jax.Array             # P("tp", None, None): per-chip [5120, 384] f8
    ss13: jax.Array             # P("tp", None, None): per-chip [160, 384] u8
    sw2: jax.Array              # P("tp", None, None): per-chip [256, 5120] f8
    ss2: jax.Array              # P("tp", None, None): per-chip [8, 5120] u8

    rw13: jax.Array             # P("tp", None, None, None): per-chip [384, 640, 384] u32
    rs13: jax.Array             # P("tp", None, None, None): per-chip [384, 160, 384] u8
    rw2: jax.Array              # P("tp", None, None, None): per-chip [384, 24, 5120] u32
    rs2: jax.Array              # P("tp", None, None, None): per-chip [384, 8, 5120] u8

    comp_w: jax.Array           # [5120, 1024] bf16
    comp_n: jax.Array           # [B_tile, 512] bf16
    idx_wk: jax.Array           # [512, 128] bf16
    idx_knorm: jax.Array        # [B_tile, 128] bf16

    idx_wqb: jax.Array          # P("tp", None, None): per-chip [1280, 256] f8
    idx_wqb_s: jax.Array        # P("tp", None, None): per-chip [40, 256] u8
    idx_wproj: jax.Array        # P("tp", None, None): per-chip [5120, 128] bf16

    eng_wkv: jax.Array          # P("tp", None, None, None): per-chip [5, 384, 5120] f8
    eng_s: jax.Array            # P("tp", None, None, None): per-chip [5, 12, 5120] u8
    eng_qk: jax.Array           # [4, B_tile, 5120] f32


def pack_pallas_layer_weights(
    lw: LayerWeights,
    mesh: Mesh,
    local_devs: List[Any],
    batch_tile: int = 8,
) -> PallasLayerWeights:
    """Convert a `LayerWeights` instance into `PallasLayerWeights` across the 4 local chips of this host."""
    rep_s = NamedSharding(mesh, P())
    tp3_s = NamedSharding(mesh, P("tp", None, None))
    tp4_s = NamedSharding(mesh, P("tp", None, None, None))

    def _put_rep(arr: np.ndarray) -> jax.Array:
        a = np.ascontiguousarray(arr)
        return jax.make_array_from_single_device_arrays(
            a.shape, rep_s, [jax.device_put(a, d) for d in local_devs]
        )

    def _put_tp3(per_chip: List[np.ndarray]) -> jax.Array:
        s0 = per_chip[0].shape
        return jax.make_array_from_single_device_arrays(
            (16, *s0),
            tp3_s,
            [jax.device_put(np.ascontiguousarray(per_chip[i][None, ...]), local_devs[i]) for i in range(4)],
        )

    def _put_tp4(per_chip: List[np.ndarray]) -> jax.Array:
        s0 = per_chip[0].shape
        return jax.make_array_from_single_device_arrays(
            (16, *s0),
            tp4_s,
            [jax.device_put(np.ascontiguousarray(per_chip[i][None, ...]), local_devs[i]) for i in range(4)],
        )

    def _bcast_row(vec_1d: np.ndarray) -> np.ndarray:
        return np.broadcast_to(vec_1d[None, :], (batch_tile, vec_1d.shape[0])).copy()

    attn_norm_np = _bcast_row(to_host_np(lw.attn_norm))
    ffn_norm_np = _bcast_row(to_host_np(lw.ffn_norm))
    hca_fn_np, hca_mb_1 = pack_hc_params(
        to_host_np(lw.hc_attn_fn), to_host_np(lw.hc_attn_scale), to_host_np(lw.hc_attn_base)
    )
    hcf_fn_np, hcf_mb_1 = pack_hc_params(
        to_host_np(lw.hc_ffn_fn), to_host_np(lw.hc_ffn_scale), to_host_np(lw.hc_ffn_base)
    )
    hca_mb_np = np.broadcast_to(hca_mb_1, (batch_tile, 128)).copy()
    hcf_mb_np = np.broadcast_to(hcf_mb_1, (batch_tile, 128)).copy()

    wqa_t, wqa_st = pack_fp8_kn(to_host_np(lw.wq_a), to_host_np(lw.wq_a_scale))
    qn_np = _bcast_row(to_host_np(lw.q_norm))
    wkv_t, wkv_st = pack_fp8_kn(to_host_np(lw.wkv), to_host_np(lw.wkv_scale))
    kvn_np = _bcast_row(to_host_np(lw.kv_norm))

    wqb_chips, wqb_s_chips, sink_chips = [], [], []
    woa_chips, woa_s_chips, wob_chips, wob_s_chips = [], [], [], []
    sw13_chips, ss13_chips, sw2_chips, ss2_chips = [], [], [], []
    rw13_chips, rs13_chips, rw2_chips, rs2_chips = [], [], [], []

    for i in range(4):
        wqb_i, wqb_si = pack_fp8_kn(
            np.asarray(lw.wq_b.addressable_shards[i].data),
            np.asarray(lw.wq_b_scale.addressable_shards[i].data),
        )
        wqb_chips.append(wqb_i)
        wqb_s_chips.append(wqb_si)
        sk_i = np.zeros((batch_tile, 128), dtype=np.float32)
        sk_i[:, :8] = np.asarray(lw.attn_sink.addressable_shards[i].data)[None, :]
        sink_chips.append(sk_i)

        woa_i, woa_si = pack_fp8_kn(
            np.asarray(lw.wo_a.addressable_shards[i].data),
            np.asarray(lw.wo_a_scale.addressable_shards[i].data),
        )
        woa_chips.append(woa_i)
        woa_s_chips.append(woa_si)

        wob_i, wob_si = pack_fp8_kn(
            np.asarray(lw.wo_b.addressable_shards[i].data),
            np.asarray(lw.wo_b_scale.addressable_shards[i].data),
        )
        wob_chips.append(wob_i)
        wob_s_chips.append(wob_si)

        sw13_i, ss13_i = pack_shared_w13(
            np.asarray(lw.shared_w1.addressable_shards[i].data),
            np.asarray(lw.shared_w1_scale.addressable_shards[i].data),
            np.asarray(lw.shared_w3.addressable_shards[i].data),
            np.asarray(lw.shared_w3_scale.addressable_shards[i].data),
        )
        sw13_chips.append(sw13_i)
        ss13_chips.append(ss13_i)

        sw2_i, ss2_i = pack_shared_w2(
            np.asarray(lw.shared_w2.addressable_shards[i].data),
            np.asarray(lw.shared_w2_scale.addressable_shards[i].data),
        )
        sw2_chips.append(sw2_i)
        ss2_chips.append(ss2_i)

        rw13_i, rs13_i = _pack_routed_w13_dev_jax(
            lw.routed_w1.addressable_shards[i].data,
            lw.routed_w1_scale.addressable_shards[i].data,
            lw.routed_w3.addressable_shards[i].data,
            lw.routed_w3_scale.addressable_shards[i].data,
        )
        rw13_chips.append(rw13_i)
        rs13_chips.append(rs13_i)

        rw2_i, rs2_i = _pack_routed_w2_dev_jax(
            lw.routed_w2.addressable_shards[i].data,
            lw.routed_w2_scale.addressable_shards[i].data,
        )
        rw2_chips.append(rw2_i)
        rs2_chips.append(rs2_i)

    gate_w_np = np.ascontiguousarray(to_host_np(lw.gate_weight).T)
    gate_b_np = _bcast_row(to_host_np(lw.gate_bias))

    comp_w_np = np.zeros((5120, 1024), dtype=ml_dtypes.bfloat16)
    comp_n_np = np.ones((batch_tile, 512), dtype=ml_dtypes.bfloat16)
    idx_wk_np = np.zeros((512, 128), dtype=ml_dtypes.bfloat16)
    idx_knorm_np = np.ones((batch_tile, 128), dtype=ml_dtypes.bfloat16)
    if lw.is_kv_source and lw.comp_wkv is not None:
        comp_w_np[:, :512] = to_host_np(lw.comp_wkv).T
        if lw.compress_ratio == 2 and lw.comp_wgate is not None:
            comp_w_np[:, 512:] = to_host_np(lw.comp_wgate).T
        comp_n_np = _bcast_row(to_host_np(lw.comp_norm))
        idx_wk_np = np.ascontiguousarray(to_host_np(lw.idx_wk).T)
        idx_knorm_np = _bcast_row(to_host_np(lw.idx_k_norm))

    idx_wqb_chips, idx_wqb_s_chips, idx_wproj_chips = [], [], []
    host_idx = jax.process_index()
    if lw.is_index_source and lw.idx_wq_b is not None:
        iwqb_full, iwqb_s_full = pack_fp8_kn(
            to_host_np(lw.idx_wq_b), to_host_np(lw.idx_wq_b_scale)
        )  # [1280, 4096], [40, 4096]
        iwproj_full = to_host_np(lw.idx_weights_proj).T  # [5120, 32]
        for i in range(4):
            c = host_idx * 4 + i
            idx_wqb_chips.append(iwqb_full[:, c * 256 : (c + 1) * 256].copy())
            idx_wqb_s_chips.append(iwqb_s_full[:, c * 256 : (c + 1) * 256].copy())
            wp_i = np.zeros((5120, 128), dtype=ml_dtypes.bfloat16)
            wp_i[:, :2] = iwproj_full[:, c * 2 : (c + 1) * 2]
            idx_wproj_chips.append(wp_i)
    else:
        for _ in range(4):
            idx_wqb_chips.append(np.zeros((1280, 256), dtype=ml_dtypes.float8_e4m3fn))
            idx_wqb_s_chips.append(np.zeros((40, 256), dtype=np.uint8))
            idx_wproj_chips.append(np.zeros((5120, 128), dtype=ml_dtypes.bfloat16))

    eng_wkv_chips, eng_s_chips = [], []
    eng_qk_np = np.zeros((4, batch_tile, 5120), dtype=np.float32)
    if lw.has_engram and lw.engram_wkv is not None:
        ewkv_t, ewkv_st = pack_fp8_kn(
            to_host_np(lw.engram_wkv), to_host_np(lw.engram_wkv_scale)
        )  # [6144, 25600], [192, 25600]
        eqw = to_host_np(lw.engram_q_weight, dtype=np.float32)  # [4, 5120]
        ekw = to_host_np(lw.engram_k_weight, dtype=np.float32)  # [4, 5120]
        qk_prod = eqw * ekw
        for s in range(4):
            eng_qk_np[s] = qk_prod[s : s + 1]
        for i in range(4):
            c = host_idx * 4 + i
            w_slice = ewkv_t[c * 384 : (c + 1) * 384, :]  # [384, 25600]
            s_slice = ewkv_st[c * 12 : (c + 1) * 12, :]   # [12, 25600]
            w_5 = np.stack(
                [w_slice[:, b * 5120 : (b + 1) * 5120] for b in range(5)], axis=0
            ).copy()
            s_5 = np.stack(
                [s_slice[:, b * 5120 : (b + 1) * 5120] for b in range(5)], axis=0
            ).copy()
            eng_wkv_chips.append(w_5)
            eng_s_chips.append(s_5)
    else:
        for _ in range(4):
            eng_wkv_chips.append(np.zeros((5, 384, 5120), dtype=ml_dtypes.float8_e4m3fn))
            eng_s_chips.append(np.zeros((5, 12, 5120), dtype=np.uint8))

    return PallasLayerWeights(
        layer_id=lw.layer_id,
        compress_ratio=lw.compress_ratio,
        is_kv_source=lw.is_kv_source,
        is_index_source=lw.is_index_source,
        has_engram=lw.has_engram,
        attn_norm=_put_rep(attn_norm_np),
        ffn_norm=_put_rep(ffn_norm_np),
        hc_attn_fn=_put_rep(hca_fn_np),
        hc_attn_mb=_put_rep(hca_mb_np),
        hc_ffn_fn=_put_rep(hcf_fn_np),
        hc_ffn_mb=_put_rep(hcf_mb_np),
        wqa=_put_rep(wqa_t),
        wqa_s=_put_rep(wqa_st),
        qn=_put_rep(qn_np),
        wkv=_put_rep(wkv_t),
        wkv_s=_put_rep(wkv_st),
        kvn=_put_rep(kvn_np),
        wqb=_put_tp3(wqb_chips),
        wqb_s=_put_tp3(wqb_s_chips),
        sink=_put_tp3(sink_chips),
        woa=_put_tp3(woa_chips),
        woa_s=_put_tp3(woa_s_chips),
        wob=_put_tp3(wob_chips),
        wob_s=_put_tp3(wob_s_chips),
        gate_w=_put_rep(gate_w_np),
        gate_b=_put_rep(gate_b_np),
        sw13=_put_tp3(sw13_chips),
        ss13=_put_tp3(ss13_chips),
        sw2=_put_tp3(sw2_chips),
        ss2=_put_tp3(ss2_chips),
        rw13=jax.make_array_from_single_device_arrays((16, 384, 640, 384), tp4_s, rw13_chips),
        rs13=jax.make_array_from_single_device_arrays((16, 384, 160, 384), tp4_s, rs13_chips),
        rw2=jax.make_array_from_single_device_arrays((16, 384, 24, 5120), tp4_s, rw2_chips),
        rs2=jax.make_array_from_single_device_arrays((16, 384, 8, 5120), tp4_s, rs2_chips),
        comp_w=_put_rep(comp_w_np),
        comp_n=_put_rep(comp_n_np),
        idx_wk=_put_rep(idx_wk_np),
        idx_knorm=_put_rep(idx_knorm_np),
        idx_wqb=_put_tp3(idx_wqb_chips),
        idx_wqb_s=_put_tp3(idx_wqb_s_chips),
        idx_wproj=_put_tp3(idx_wproj_chips),
        eng_wkv=_put_tp4(eng_wkv_chips),
        eng_s=_put_tp4(eng_s_chips),
        eng_qk=_put_rep(eng_qk_np),
    )


def execute_pallas_layer_in_vmem(
    active_b: int,
    compress_ratio: int,
    is_kv_source: bool,
    is_index_source: bool,
    has_engram: bool,
    res_vmem: Any,
    pm_vmem: Any,
    pos_vmem: Any,
    inv_freq_vmem: Any,
    rope_sign_vmem: Any,
    p_swap_vmem: Any,
    s32_vmem: Any,
    eng_rows_vmem: Any,
    eng_wkv_hbm: Any,
    eng_s_hbm: Any,
    eng_qk_hbm: Any,
    attn_norm_hbm: Any,
    ffn_norm_hbm: Any,
    hc_attn_fn_hbm: Any,
    hc_attn_mb_hbm: Any,
    hc_ffn_fn_hbm: Any,
    hc_ffn_mb_hbm: Any,
    wqa_hbm: Any,
    wqa_s_hbm: Any,
    qn_hbm: Any,
    wkv_hbm: Any,
    wkv_s_hbm: Any,
    kvn_hbm: Any,
    wqb_hbm: Any,
    wqb_s_hbm: Any,
    sink_hbm: Any,
    woa_hbm: Any,
    woa_s_hbm: Any,
    wob_hbm: Any,
    wob_s_hbm: Any,
    gate_w_hbm: Any,
    gate_b_hbm: Any,
    sw13_hbm: Any,
    ss13_hbm: Any,
    sw2_hbm: Any,
    ss2_hbm: Any,
    rw13_hbm: Any,
    rs13_hbm: Any,
    rw2_hbm: Any,
    rs2_hbm: Any,
    comp_w_hbm: Any,
    comp_n_hbm: Any,
    idx_wk_hbm: Any,
    idx_knorm_hbm: Any,
    idx_wqb_hbm: Any,
    idx_wqb_s_hbm: Any,
    idx_wproj_hbm: Any,
    swa_kv_vmem: Any,
    swa_pos_vmem: Any,
    comp_kv_vmem: Any,
    ik_f8_vmem: Any,
    ik_sc_vmem: Any,
    comp_tail_vmem: Any,
    csa_mask_vmem: Any,
    # Scratch buffers
    w_eng_vmem: Any,
    s_eng_vmem: Any,
    eng_qk_vmem: Any,
    norm_vmem: Any,
    hc_fn_vmem: Any,
    hc_mb_vmem: Any,
    xin_vmem: Any,
    x_norm_vmem: Any,
    next_pm_vmem: Any,
    wqa_vmem: Any,
    wqa_s_vmem: Any,
    qn_vmem: Any,
    wkv_vmem: Any,
    wkv_s_vmem: Any,
    kvn_vmem: Any,
    wqb_vmem: Any,
    wqb_s_vmem: Any,
    sink_vmem: Any,
    woa_vmem: Any,
    woa_s_vmem: Any,
    wob_vmem: Any,
    wob_s_vmem: Any,
    comp_w_vmem: Any,
    comp_n_vmem: Any,
    idx_wk_vmem: Any,
    idx_knorm_vmem: Any,
    idx_wqb_vmem: Any,
    idx_wqb_s_vmem: Any,
    idx_wproj_vmem: Any,
    qr_f32_vmem: Any,
    qr_vmem: Any,
    kv_f32_vmem: Any,
    kv_vmem: Any,
    q_f32_vmem: Any,
    q_vmem: Any,
    o_vmem: Any,
    z_f32_vmem: Any,
    z_vmem: Any,
    out_f32_vmem: Any,
    attn_out_vmem: Any,
    lat_f32_vmem: Any,
    lat_vmem: Any,
    idx_q_f32_vmem: Any,
    gate_w_vmem: Any,
    gate_b_vmem: Any,
    sw13_vmem: Any,
    ss13_vmem: Any,
    sw2_vmem: Any,
    ss2_vmem: Any,
    w13_vmem: Any,
    s13_vmem: Any,
    w2_vmem: Any,
    s2_vmem: Any,
    gu_acc_vmem: Any,
    h256_vmem: Any,
    down_acc_vmem: Any,
    routed_acc_vmem: Any,
    sel_exp_vmem: Any,
    sel_prob_vmem: Any,
    proc_vmem: Any,
    sem: Any,
    local_buf: Any,
    host_buf: Any,
    sends: Any,
    recvs: Any,
    iter_base: Any = 0,
    sub_idx: Optional[Any] = None,
    spec_k: int = 0,
) -> None:
    """Execute one full DSV4.1-Flash decoder layer (`Engram` + `mHC` + `Attn/Comp/Idx` + `mHC` + `MoE`) in VMEM."""
    if has_engram:
        pallas_engram_sublayer(
            res_vmem,
            eng_rows_vmem,
            eng_wkv_hbm,
            eng_s_hbm,
            eng_qk_hbm,
            w_eng_vmem,
            s_eng_vmem,
            eng_qk_vmem,
            out_f32_vmem,
            attn_out_vmem,
            sem,
            local_buf,
            host_buf,
            sends,
            recvs,
            iter_base=iter_base,
        )

    attn_copies = [
        (hc_attn_fn_hbm, hc_fn_vmem),
        (hc_attn_mb_hbm, hc_mb_vmem),
        (attn_norm_hbm, norm_vmem),
        (wqa_hbm, wqa_vmem),
        (wqa_s_hbm, wqa_s_vmem),
        (qn_hbm, qn_vmem),
        (wkv_hbm, wkv_vmem),
        (wkv_s_hbm, wkv_s_vmem),
        (kvn_hbm, kvn_vmem),
        (wqb_hbm, wqb_vmem),
        (wqb_s_hbm, wqb_s_vmem),
        (sink_hbm, sink_vmem),
        (woa_hbm, woa_vmem),
        (woa_s_hbm, woa_s_vmem),
        (wob_hbm, wob_vmem),
        (wob_s_hbm, wob_s_vmem),
    ]
    if is_kv_source and compress_ratio > 0:
        attn_copies.extend(
            [
                (comp_w_hbm, comp_w_vmem),
                (comp_n_hbm, comp_n_vmem),
                (idx_wk_hbm, idx_wk_vmem),
                (idx_knorm_hbm, idx_knorm_vmem),
            ]
        )
    if is_index_source and compress_ratio > 0:
        attn_copies.extend(
            [
                (idx_wqb_hbm, idx_wqb_vmem),
                (idx_wqb_s_hbm, idx_wqb_s_vmem),
                (idx_wproj_hbm, idx_wproj_vmem),
            ]
        )

    for idx_c, (src, dst) in enumerate(attn_copies):
        c = pltpu.make_async_copy(src, dst, sem.at[idx_c % 16])
        c.start()
        c.wait()

    attn_post, attn_k = mhc_pre_and_post_coeffs_vmem(
        res_vmem, pm_vmem, hc_fn_vmem, hc_mb_vmem, xin_vmem, next_pm_vmem
    )
    rms_norm_vmem(xin_vmem, norm_vmem, x_norm_vmem)

    pallas_attention_sublayer(
        compress_ratio=compress_ratio,
        is_kv_source=is_kv_source,
        is_index_source=is_index_source,
        x_norm_vmem=x_norm_vmem,
        pos_vmem=pos_vmem,
        inv_freq_vmem=inv_freq_vmem,
        rope_sign_vmem=rope_sign_vmem,
        p_swap_vmem=p_swap_vmem,
        wqa_vmem=wqa_vmem,
        wqa_s_vmem=wqa_s_vmem,
        qn_vmem=qn_vmem,
        wkv_vmem=wkv_vmem,
        wkv_s_vmem=wkv_s_vmem,
        kvn_vmem=kvn_vmem,
        wqb_vmem=wqb_vmem,
        wqb_s_vmem=wqb_s_vmem,
        sink_vmem=sink_vmem,
        woa_vmem=woa_vmem,
        woa_s_vmem=woa_s_vmem,
        wob_vmem=wob_vmem,
        wob_s_vmem=wob_s_vmem,
        comp_w_vmem=comp_w_vmem,
        comp_n_vmem=comp_n_vmem,
        idx_wk_vmem=idx_wk_vmem,
        idx_knorm_vmem=idx_knorm_vmem,
        idx_wqb_vmem=idx_wqb_vmem,
        idx_wqb_s_vmem=idx_wqb_s_vmem,
        idx_wproj_vmem=idx_wproj_vmem,
        swa_kv_vmem=swa_kv_vmem,
        swa_pos_vmem=swa_pos_vmem,
        comp_kv_vmem=comp_kv_vmem,
        ik_f8_vmem=ik_f8_vmem,
        ik_sc_vmem=ik_sc_vmem,
        comp_tail_vmem=comp_tail_vmem,
        csa_mask_vmem=csa_mask_vmem,
        qr_f32_vmem=qr_f32_vmem,
        qr_vmem=qr_vmem,
        kv_f32_vmem=kv_f32_vmem,
        kv_vmem=kv_vmem,
        q_f32_vmem=q_f32_vmem,
        q_vmem=q_vmem,
        o_vmem=o_vmem,
        z_f32_vmem=z_f32_vmem,
        z_vmem=z_vmem,
        out_f32_vmem=out_f32_vmem,
        attn_out_vmem=attn_out_vmem,
        lat_f32_vmem=lat_f32_vmem,
        lat_vmem=lat_vmem,
        idx_q_f32_vmem=idx_q_f32_vmem,
        local_buf=local_buf,
        host_buf=host_buf,
        sends=sends,
        recvs=recvs,
        iter_base=iter_base + 10,
        spec_k=spec_k,
    )

    mhc_post_apply_vmem(res_vmem, attn_out_vmem, attn_post, attn_k)

    ffn_copies = [
        (hc_ffn_fn_hbm, hc_fn_vmem),
        (hc_ffn_mb_hbm, hc_mb_vmem),
        (ffn_norm_hbm, norm_vmem),
        (gate_w_hbm, gate_w_vmem),
        (gate_b_hbm, gate_b_vmem),
        (sw13_hbm, sw13_vmem),
        (ss13_hbm, ss13_vmem),
        (sw2_hbm, sw2_vmem),
        (ss2_hbm, ss2_vmem),
    ]
    for idx_c, (src, dst) in enumerate(ffn_copies):
        c = pltpu.make_async_copy(src, dst, sem.at[idx_c])
        c.start()
        c.wait()

    ffn_post, ffn_k = mhc_pre_and_post_coeffs_vmem(
        res_vmem, next_pm_vmem, hc_fn_vmem, hc_mb_vmem, xin_vmem, pm_vmem
    )
    rms_norm_vmem(xin_vmem, norm_vmem, x_norm_vmem)

    pallas_moe_sublayer(
        active_b=active_b,
        x_norm_vmem=x_norm_vmem,
        gate_w_vmem=gate_w_vmem,
        gate_b_vmem=gate_b_vmem,
        s32_vmem=s32_vmem,
        sw13_vmem=sw13_vmem,
        ss13_vmem=ss13_vmem,
        sw2_vmem=sw2_vmem,
        ss2_vmem=ss2_vmem,
        rw13_hbm_layer=rw13_hbm,
        rs13_hbm_layer=rs13_hbm,
        rw2_hbm_layer=rw2_hbm,
        rs2_hbm_layer=rs2_hbm,
        w13_vmem=w13_vmem,
        s13_vmem=s13_vmem,
        w2_vmem=w2_vmem,
        s2_vmem=s2_vmem,
        gu_acc_vmem=gu_acc_vmem,
        h256_vmem=h256_vmem,
        down_acc_vmem=down_acc_vmem,
        routed_acc_vmem=routed_acc_vmem,
        sel_exp_vmem=sel_exp_vmem,
        sel_prob_vmem=sel_prob_vmem,
        proc_vmem=proc_vmem,
        moe_out_vmem=attn_out_vmem,
        sem=sem,
        local_buf=local_buf,
        host_buf=host_buf,
        sends=sends,
        recvs=recvs,
        iter_base=iter_base + 14,
        sub_idx=sub_idx,
        pos_vmem=pos_vmem,
    )

    mhc_post_apply_vmem(res_vmem, attn_out_vmem, ffn_post, ffn_k)


def _build_common_scratch_shapes(batch_tile: int = 8) -> List[Any]:
    """Return the shared VMEM + Semaphore scratch shapes used by both single-layer and 40-layer Pallas kernels."""
    return [
        pltpu.VMEM((384, 5120), jnp.float8_e4m3fn),   # w_eng_vmem
        pltpu.VMEM((12, 5120), jnp.uint8),            # s_eng_vmem
        pltpu.VMEM((4, batch_tile, 5120), jnp.float32),  # eng_qk_vmem
        pltpu.VMEM((batch_tile, 5120), jnp.bfloat16),    # norm_vmem
        pltpu.VMEM((5120, 128), jnp.float32),         # hc_fn_vmem
        pltpu.VMEM((batch_tile, 128), jnp.float32),   # hc_mb_vmem
        pltpu.VMEM((batch_tile, 5120), jnp.bfloat16), # xin_vmem
        pltpu.VMEM((batch_tile, 5120), jnp.bfloat16), # x_norm_vmem
        pltpu.VMEM((4, batch_tile, 128), jnp.float32),# next_pm_vmem
        pltpu.VMEM((5120, 1280), jnp.float8_e4m3fn),  # wqa_vmem
        pltpu.VMEM((160, 1280), jnp.uint8),           # wqa_s_vmem
        pltpu.VMEM((batch_tile, 1280), jnp.bfloat16), # qn_vmem
        pltpu.VMEM((5120, 512), jnp.float8_e4m3fn),   # wkv_vmem
        pltpu.VMEM((160, 512), jnp.uint8),            # wkv_s_vmem
        pltpu.VMEM((batch_tile, 512), jnp.bfloat16),  # kvn_vmem
        pltpu.VMEM((1280, 4096), jnp.float8_e4m3fn),  # wqb_vmem
        pltpu.VMEM((40, 4096), jnp.uint8),            # wqb_s_vmem
        pltpu.VMEM((batch_tile, 128), jnp.float32),   # sink_vmem
        pltpu.VMEM((4096, 512), jnp.float8_e4m3fn),   # woa_vmem
        pltpu.VMEM((128, 512), jnp.uint8),            # woa_s_vmem
        pltpu.VMEM((512, 5120), jnp.float8_e4m3fn),   # wob_vmem
        pltpu.VMEM((16, 5120), jnp.uint8),            # wob_s_vmem
        pltpu.VMEM((5120, 1024), jnp.bfloat16),       # comp_w_vmem
        pltpu.VMEM((batch_tile, 512), jnp.bfloat16),  # comp_n_vmem
        pltpu.VMEM((512, 128), jnp.bfloat16),         # idx_wk_vmem
        pltpu.VMEM((batch_tile, 128), jnp.bfloat16),  # idx_knorm_vmem
        pltpu.VMEM((1280, 256), jnp.float8_e4m3fn),   # idx_wqb_vmem
        pltpu.VMEM((40, 256), jnp.uint8),             # idx_wqb_s_vmem
        pltpu.VMEM((5120, 128), jnp.bfloat16),        # idx_wproj_vmem
        pltpu.VMEM((batch_tile, 1280), jnp.float32),  # qr_f32_vmem
        pltpu.VMEM((batch_tile, 1280), jnp.bfloat16), # qr_vmem
        pltpu.VMEM((batch_tile, 512), jnp.float32),   # kv_f32_vmem
        pltpu.VMEM((batch_tile, 512), jnp.bfloat16),  # kv_vmem
        pltpu.VMEM((batch_tile, 4096), jnp.float32),  # q_f32_vmem
        pltpu.VMEM((batch_tile, 4096), jnp.bfloat16), # q_vmem
        pltpu.VMEM((batch_tile, 4096), jnp.bfloat16), # o_vmem
        pltpu.VMEM((batch_tile, 512), jnp.float32),   # z_f32_vmem
        pltpu.VMEM((batch_tile, 512), jnp.bfloat16),  # z_vmem
        pltpu.VMEM((batch_tile, 5120), jnp.float32),  # out_f32_vmem
        pltpu.VMEM((batch_tile, 5120), jnp.bfloat16), # attn_out_vmem
        pltpu.VMEM((batch_tile, 512), jnp.float32),   # lat_f32_vmem
        pltpu.VMEM((batch_tile, 512), jnp.bfloat16),  # lat_vmem
        pltpu.VMEM((batch_tile, 256), jnp.float32),   # idx_q_f32_vmem
        pltpu.VMEM((5120, 384), jnp.bfloat16),        # gate_w_vmem
        pltpu.VMEM((batch_tile, 384), jnp.float32),   # gate_b_vmem
        pltpu.VMEM((5120, 384), jnp.float8_e4m3fn),   # sw13_vmem
        pltpu.VMEM((160, 384), jnp.uint8),            # ss13_vmem
        pltpu.VMEM((256, 5120), jnp.float8_e4m3fn),   # sw2_vmem
        pltpu.VMEM((8, 5120), jnp.uint8),             # ss2_vmem
        pltpu.VMEM((640, 384), jnp.uint32),           # w13_vmem
        pltpu.VMEM((160, 384), jnp.uint8),            # s13_vmem
        pltpu.VMEM((32, 5120), jnp.uint32),           # w2_vmem
        pltpu.VMEM((8, 5120), jnp.uint8),             # s2_vmem
        pltpu.VMEM((batch_tile, 384), jnp.float32),   # gu_acc_vmem
        pltpu.VMEM((batch_tile, 256), jnp.bfloat16),  # h256_vmem
        pltpu.VMEM((batch_tile, 5120), jnp.float32),  # down_acc_vmem
        pltpu.VMEM((batch_tile, 5120), jnp.float32),  # routed_acc_vmem
        pltpu.VMEM((batch_tile, 128), jnp.int32),     # sel_exp_vmem
        pltpu.VMEM((batch_tile, 128), jnp.float32),   # sel_prob_vmem
        pltpu.VMEM((batch_tile, 128), jnp.int32),     # proc_vmem
        pltpu.SemaphoreType.DMA((16,)),
        *collectives16.scratch_2d(shape=(batch_tile, 5120), dtype=jnp.bfloat16),
    ]


class DSV41PallasLayerRunner:
    """Single-layer Pallas decode runner for per-layer parity verification across all 40 layers."""

    def __init__(self, cfg: DSV41Config, mesh: Mesh, max_comp: int = 512, batch_tile: int = 8) -> None:
        self.cfg = cfg
        self.mesh = mesh
        self.max_comp = max_comp
        self.batch_tile = batch_tile
        self.local_devs = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))[
            jax.process_index() * 4 : (jax.process_index() + 1) * 4
        ]
        self.rep_s = NamedSharding(mesh, P())
        self.tp3_s = NamedSharding(mesh, P("tp", None, None))
        consts = build_pallas_constants(cfg, batch_tile=batch_tile)
        self.p_swap = self.replicate(consts["p_swap"])
        self.s32 = self.replicate(consts["s32"])
        self.inv_freq_plain = self.replicate(consts["inv_freq_plain"])
        self.inv_freq_yarn = self.replicate(consts["inv_freq_yarn"])
        self.rope_sign = self.replicate(consts["rope_sign"])
        self._compiled: Dict[Tuple[int, int, bool, bool, bool], Any] = {}

    def replicate(self, arr: np.ndarray) -> jax.Array:
        a = np.ascontiguousarray(arr)
        return jax.make_array_from_single_device_arrays(
            a.shape, self.rep_s, [jax.device_put(a, d) for d in self.local_devs]
        )

    def _get_kernel(
        self,
        active_b: int,
        compress_ratio: int,
        is_kv_source: bool,
        is_index_source: bool,
        has_engram: bool,
    ) -> Any:
        key = (active_b, compress_ratio, is_kv_source, is_index_source, has_engram)
        if key in self._compiled:
            return self._compiled[key]

        b_tile = self.batch_tile
        max_comp = self.max_comp

        def _kernel(
            res_in_ref,
            pm_in_ref,
            pos_ref,
            inv_freq_ref,
            rope_sign_ref,
            p_swap_ref,
            s32_ref,
            eng_rows_ref,
            swa_kv_in_ref,
            swa_pos_in_ref,
            comp_kv_in_ref,
            ik_f8_in_ref,
            ik_sc_in_ref,
            comp_tail_in_ref,
            csa_mask_in_ref,
            eng_wkv_hbm,
            eng_s_hbm,
            eng_qk_hbm,
            attn_norm_hbm,
            ffn_norm_hbm,
            hc_attn_fn_hbm,
            hc_attn_mb_hbm,
            hc_ffn_fn_hbm,
            hc_ffn_mb_hbm,
            wqa_hbm,
            wqa_s_hbm,
            qn_hbm,
            wkv_hbm,
            wkv_s_hbm,
            kvn_hbm,
            wqb_hbm,
            wqb_s_hbm,
            sink_hbm,
            woa_hbm,
            woa_s_hbm,
            wob_hbm,
            wob_s_hbm,
            gate_w_hbm,
            gate_b_hbm,
            sw13_hbm,
            ss13_hbm,
            sw2_hbm,
            ss2_hbm,
            rw13_hbm,
            rs13_hbm,
            rw2_hbm,
            rs2_hbm,
            comp_w_hbm,
            comp_n_hbm,
            idx_wk_hbm,
            idx_knorm_hbm,
            idx_wqb_hbm,
            idx_wqb_s_hbm,
            idx_wproj_hbm,
            # Outputs
            res_out_ref,
            pm_out_ref,
            swa_kv_out_ref,
            swa_pos_out_ref,
            comp_kv_out_ref,
            ik_f8_out_ref,
            ik_sc_out_ref,
            comp_tail_out_ref,
            csa_mask_out_ref,
            # Scratches
            *scratches,
        ):
            collectives16.barrier()
            res_out_ref[...] = res_in_ref[...]
            pm_out_ref[...] = pm_in_ref[...]
            swa_kv_out_ref[...] = swa_kv_in_ref[...]
            swa_pos_out_ref[...] = swa_pos_in_ref[...]
            comp_kv_out_ref[...] = comp_kv_in_ref[...]
            ik_f8_out_ref[...] = ik_f8_in_ref[...]
            ik_sc_out_ref[...] = ik_sc_in_ref[...]
            comp_tail_out_ref[...] = comp_tail_in_ref[...]
            csa_mask_out_ref[...] = csa_mask_in_ref[...]

            execute_pallas_layer_in_vmem(
                active_b,
                compress_ratio,
                is_kv_source,
                is_index_source,
                has_engram,
                res_out_ref,
                pm_out_ref,
                pos_ref,
                inv_freq_ref,
                rope_sign_ref,
                p_swap_ref,
                s32_ref,
                eng_rows_ref,
                eng_wkv_hbm,
                eng_s_hbm,
                eng_qk_hbm,
                attn_norm_hbm,
                ffn_norm_hbm,
                hc_attn_fn_hbm,
                hc_attn_mb_hbm,
                hc_ffn_fn_hbm,
                hc_ffn_mb_hbm,
                wqa_hbm,
                wqa_s_hbm,
                qn_hbm,
                wkv_hbm,
                wkv_s_hbm,
                kvn_hbm,
                wqb_hbm,
                wqb_s_hbm,
                sink_hbm,
                woa_hbm,
                woa_s_hbm,
                wob_hbm,
                wob_s_hbm,
                gate_w_hbm,
                gate_b_hbm,
                sw13_hbm,
                ss13_hbm,
                sw2_hbm,
                ss2_hbm,
                rw13_hbm,
                rs13_hbm,
                rw2_hbm,
                rs2_hbm,
                comp_w_hbm,
                comp_n_hbm,
                idx_wk_hbm,
                idx_knorm_hbm,
                idx_wqb_hbm,
                idx_wqb_s_hbm,
                idx_wproj_hbm,
                swa_kv_out_ref,
                swa_pos_out_ref,
                comp_kv_out_ref,
                ik_f8_out_ref,
                ik_sc_out_ref,
                comp_tail_out_ref,
                csa_mask_out_ref,
                *scratches,
                iter_base=0,
            )

        in_specs = [
            *(pl.BlockSpec(memory_space=pltpu.VMEM) for _ in range(15)),
            *(pl.BlockSpec(memory_space=pl.ANY) for _ in range(39)),
        ]
        out_shapes = (
            jax.ShapeDtypeStruct((4, b_tile, 5120), jnp.bfloat16),
            jax.ShapeDtypeStruct((4, b_tile, 128), jnp.float32),
            jax.ShapeDtypeStruct((128, 512), jnp.bfloat16),
            jax.ShapeDtypeStruct((b_tile, 128), jnp.int32),
            jax.ShapeDtypeStruct((max_comp, 512), jnp.bfloat16),
            jax.ShapeDtypeStruct((max_comp, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((b_tile, max_comp), jnp.float32),
            jax.ShapeDtypeStruct((b_tile, 1024), jnp.float32),
            jax.ShapeDtypeStruct((b_tile, max_comp), jnp.int32),
        )
        out_specs = tuple(pl.BlockSpec(memory_space=pltpu.VMEM) for _ in range(9))
        p_call = pl.pallas_call(
            _kernel,
            out_shape=out_shapes,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=_build_common_scratch_shapes(b_tile),
            compiler_params=pltpu.CompilerParams(
                collective_id=0, vmem_limit_bytes=104 * 1024 * 1024
            ),
        )

        # Shard specs: TP-sharded inputs have leading dim 16 which we squeeze inside shard_map
        tp_arg_indices = {
            7,   # eng_rows_ref: [16, B_tile, 384]
            15, 16,  # eng_wkv, eng_s
            30, 31, 32,  # wqb, wqb_s, sink
            33, 34, 35, 36,  # woa, woa_s, wob, wob_s
            39, 40, 41, 42,  # sw13, ss13, sw2, ss2
            43, 44, 45, 46,  # rw13, rs13, rw2, rs2
            51, 52, 53,      # idx_wqb, idx_wqb_s, idx_wproj
        }
        smap_in_specs = tuple(
            P("tp") if idx in tp_arg_indices else P()
            for idx in range(54)
        )

        def _smap_wrapper(*args):
            unwrapped = [
                args[idx][0] if idx in tp_arg_indices else args[idx]
                for idx in range(54)
            ]
            return p_call(*unwrapped)

        fn = jax.jit(
            jax.shard_map(
                _smap_wrapper,
                mesh=self.mesh,
                in_specs=smap_in_specs,
                out_specs=tuple(P() for _ in range(9)),
                check_vma=False,
            )
        )
        self._compiled[key] = fn
        return fn

    def run_layer(
        self,
        plw: PallasLayerWeights,
        active_b: int,
        res_vmem: jax.Array,
        pm_vmem: jax.Array,
        pos_vmem: jax.Array,
        eng_rows_tp: jax.Array,
        swa_kv: jax.Array,
        swa_pos: jax.Array,
        comp_kv: jax.Array,
        ik_f8: jax.Array,
        ik_sc: jax.Array,
        comp_tail: jax.Array,
        csa_mask: jax.Array,
    ) -> Tuple[jax.Array, ...]:
        fn = self._get_kernel(
            active_b=active_b,
            compress_ratio=plw.compress_ratio,
            is_kv_source=plw.is_kv_source,
            is_index_source=plw.is_index_source,
            has_engram=plw.has_engram,
        )
        inv_f = self.inv_freq_plain if plw.compress_ratio == 0 else self.inv_freq_yarn
        return fn(
            res_vmem,
            pm_vmem,
            pos_vmem,
            inv_f,
            self.rope_sign,
            self.p_swap,
            self.s32,
            eng_rows_tp,
            swa_kv,
            swa_pos,
            comp_kv,
            ik_f8,
            ik_sc,
            comp_tail,
            csa_mask,
            plw.eng_wkv,
            plw.eng_s,
            plw.eng_qk,
            plw.attn_norm,
            plw.ffn_norm,
            plw.hc_attn_fn,
            plw.hc_attn_mb,
            plw.hc_ffn_fn,
            plw.hc_ffn_mb,
            plw.wqa,
            plw.wqa_s,
            plw.qn,
            plw.wkv,
            plw.wkv_s,
            plw.kvn,
            plw.wqb,
            plw.wqb_s,
            plw.sink,
            plw.woa,
            plw.woa_s,
            plw.wob,
            plw.wob_s,
            plw.gate_w,
            plw.gate_b,
            plw.sw13,
            plw.ss13,
            plw.sw2,
            plw.ss2,
            plw.rw13,
            plw.rs13,
            plw.rw2,
            plw.rs2,
            plw.comp_w,
            plw.comp_n,
            plw.idx_wk,
            plw.idx_knorm,
            plw.idx_wqb,
            plw.idx_wqb_s,
            plw.idx_wproj,
        )


@dataclass
class PallasMegakernelWeights:
    """Full 40-layer sharded weight container for the single-`pallas_call` DSV4.1-Flash decode megakernel."""

    cfg: DSV41Config
    mesh: Mesh
    engram_host: Any

    embed_weight: jax.Array     # P(None, "tp"): [129280, 5120] bf16
    final_norm: jax.Array       # P(): [B_tile, 5120] bf16
    lm_head: jax.Array          # P("tp"): per-chip [8, 5120, 1024] bf16 (8080 padded to 8192)

    eng_wkv: jax.Array          # P("tp"): per-chip [2, 5, 384, 5120] f8
    eng_s: jax.Array            # P("tp"): per-chip [2, 5, 12, 5120] u8
    eng_qk: jax.Array           # P(): [2, 4, B_tile, 5120] f32

    attn_norm: jax.Array        # P(): [40, B_tile, 5120] bf16
    ffn_norm: jax.Array         # P(): [40, B_tile, 5120] bf16
    hc_attn_fn: jax.Array       # P(): [40, 5120, 128] f32
    hc_attn_mb: jax.Array       # P(): [40, B_tile, 128] f32
    hc_ffn_fn: jax.Array        # P(): [40, 5120, 128] f32
    hc_ffn_mb: jax.Array        # P(): [40, B_tile, 128] f32

    wqa: jax.Array              # P(): [40, 5120, 1280] f8
    wqa_s: jax.Array            # P(): [40, 160, 1280] u8
    qn: jax.Array               # P(): [40, B_tile, 1280] bf16
    wkv: jax.Array              # P(): [40, 5120, 512] f8
    wkv_s: jax.Array            # P(): [40, 160, 512] u8
    kvn: jax.Array              # P(): [40, B_tile, 512] bf16

    wqb: jax.Array              # P("tp"): per-chip [40, 1280, 4096] f8
    wqb_s: jax.Array            # P("tp"): per-chip [40, 40, 4096] u8
    sink: jax.Array             # P("tp"): per-chip [40, B_tile, 128] f32

    woa: jax.Array              # P("tp"): per-chip [40, 4096, 512] f8
    woa_s: jax.Array            # P("tp"): per-chip [40, 128, 512] u8
    wob: jax.Array              # P("tp"): per-chip [40, 512, 5120] f8
    wob_s: jax.Array            # P("tp"): per-chip [40, 16, 5120] u8

    gate_w: jax.Array           # P(): [40, 5120, 384] bf16
    gate_b: jax.Array           # P(): [40, B_tile, 384] f32
    sw13: jax.Array             # P("tp"): per-chip [40, 5120, 384] f8
    ss13: jax.Array             # P("tp"): per-chip [40, 160, 384] u8
    sw2: jax.Array              # P("tp"): per-chip [40, 256, 5120] f8
    ss2: jax.Array              # P("tp"): per-chip [40, 8, 5120] u8

    comp_w: jax.Array           # P(): [4, 5120, 1024] bf16 (layers 2, 8, 14, 20)
    comp_n: jax.Array           # P(): [4, B_tile, 512] bf16
    idx_wk: jax.Array           # P(): [4, 512, 128] bf16
    idx_knorm: jax.Array        # P(): [4, B_tile, 128] bf16

    idx_wqb: jax.Array          # P("tp"): per-chip [8, 1280, 256] f8 (layers 2, 8, 14, 20, 24, 28, 32, 36)
    idx_wqb_s: jax.Array        # P("tp"): per-chip [8, 40, 256] u8
    idx_wproj: jax.Array        # P("tp"): per-chip [8, 5120, 128] bf16

    rw13_groups: Tuple[jax.Array, ...]  # 8 x P("tp"): per-chip [5, 384, 640, 384] u32
    rs13_groups: Tuple[jax.Array, ...]  # 8 x P("tp"): per-chip [5, 384, 160, 384] u8
    rw2_groups: Tuple[jax.Array, ...]   # 8 x P("tp"): per-chip [5, 384, 24, 5120] u32
    rs2_groups: Tuple[jax.Array, ...]   # 8 x P("tp"): per-chip [5, 384, 8, 5120] u8


def pack_megakernel_weights(
    mw: ModelWeights,
    batch_tile: int = 8,
    free_source_layers: bool = True,
) -> PallasMegakernelWeights:
    """Pack a 40-layer `ModelWeights` into `PallasMegakernelWeights`, optionally freeing raw layers per group."""
    import gc

    cfg = mw.config
    mesh = mw.mesh
    host_idx = jax.process_index()
    local_devs = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))[
        host_idx * 4 : (host_idx + 1) * 4
    ]
    rep_s = NamedSharding(mesh, P())
    tp_s = NamedSharding(mesh, P("tp"))

    def _put_rep(arr: np.ndarray) -> jax.Array:
        a = np.ascontiguousarray(arr)
        return jax.make_array_from_single_device_arrays(
            a.shape, rep_s, [jax.device_put(a, d) for d in local_devs]
        )

    def _put_tp(per_chip: List[np.ndarray]) -> jax.Array:
        s0 = per_chip[0].shape
        return jax.make_array_from_single_device_arrays(
            (16, *s0),
            tp_s,
            [
                jax.device_put(np.ascontiguousarray(per_chip[i][None, ...]), local_devs[i])
                for i in range(4)
            ],
        )

    def _bcast_row(vec_1d: np.ndarray) -> np.ndarray:
        return np.broadcast_to(vec_1d[None, :], (batch_tile, vec_1d.shape[0])).copy()

    final_norm_np = _bcast_row(to_host_np(mw.norm_weight))
    lm_head_chips = []
    for i in range(4):
        hw_i = np.asarray(mw.head_weight.addressable_shards[i].data)  # [8080, 5120] bf16
        hw_pad = np.zeros((5120, 8192), dtype=ml_dtypes.bfloat16)
        hw_pad[:, :8080] = hw_i.T
        hw_blocks = np.stack(
            [hw_pad[:, b * 1024 : (b + 1) * 1024] for b in range(8)], axis=0
        ).copy()
        lm_head_chips.append(hw_blocks)

    attn_norm_l, ffn_norm_l = [], []
    hca_fn_l, hca_mb_l, hcf_fn_l, hcf_mb_l = [], [], [], []
    wqa_l, wqa_s_l, qn_l, wkv_l, wkv_s_l, kvn_l = [], [], [], [], [], []
    wqb_c = [[] for _ in range(4)]
    wqb_sc = [[] for _ in range(4)]
    sink_c = [[] for _ in range(4)]
    woa_c = [[] for _ in range(4)]
    woa_sc = [[] for _ in range(4)]
    wob_c = [[] for _ in range(4)]
    wob_sc = [[] for _ in range(4)]
    gate_w_l, gate_b_l = [], []
    sw13_c = [[] for _ in range(4)]
    ss13_c = [[] for _ in range(4)]
    sw2_c = [[] for _ in range(4)]
    ss2_c = [[] for _ in range(4)]

    comp_w_l, comp_n_l, idx_wk_l, idx_knorm_l = [], [], [], []
    idx_wqb_c = [[] for _ in range(4)]
    idx_wqb_sc = [[] for _ in range(4)]
    idx_wproj_c = [[] for _ in range(4)]
    eng_wkv_c = [[] for _ in range(4)]
    eng_s_c = [[] for _ in range(4)]
    eng_qk_l = []

    rw13_groups, rs13_groups, rw2_groups, rs2_groups = [], [], [], []
    layers_mut = list(mw.layers)

    for g in range(8):
        g_rw13 = [[] for _ in range(4)]
        g_rs13 = [[] for _ in range(4)]
        g_rw2 = [[] for _ in range(4)]
        g_rs2 = [[] for _ in range(4)]

        for sub in range(5):
            lid = g * 5 + sub
            lw = layers_mut[lid]

            attn_norm_l.append(_bcast_row(to_host_np(lw.attn_norm)))
            ffn_norm_l.append(_bcast_row(to_host_np(lw.ffn_norm)))
            hca_fn, hca_mb1 = pack_hc_params(
                to_host_np(lw.hc_attn_fn), to_host_np(lw.hc_attn_scale), to_host_np(lw.hc_attn_base)
            )
            hcf_fn, hcf_mb1 = pack_hc_params(
                to_host_np(lw.hc_ffn_fn), to_host_np(lw.hc_ffn_scale), to_host_np(lw.hc_ffn_base)
            )
            hca_fn_l.append(hca_fn)
            hca_mb_l.append(np.broadcast_to(hca_mb1, (batch_tile, 128)).copy())
            hcf_fn_l.append(hcf_fn)
            hcf_mb_l.append(np.broadcast_to(hcf_mb1, (batch_tile, 128)).copy())

            wqa_t, wqa_st = pack_fp8_kn(to_host_np(lw.wq_a), to_host_np(lw.wq_a_scale))
            wqa_l.append(wqa_t)
            wqa_s_l.append(wqa_st)
            qn_l.append(_bcast_row(to_host_np(lw.q_norm)))

            wkv_t, wkv_st = pack_fp8_kn(to_host_np(lw.wkv), to_host_np(lw.wkv_scale))
            wkv_l.append(wkv_t)
            wkv_s_l.append(wkv_st)
            kvn_l.append(_bcast_row(to_host_np(lw.kv_norm)))

            gate_w_l.append(np.ascontiguousarray(to_host_np(lw.gate_weight).T))
            gate_b_l.append(_bcast_row(to_host_np(lw.gate_bias)))

            for i in range(4):
                wqb_i, wqb_si = pack_fp8_kn(
                    np.asarray(lw.wq_b.addressable_shards[i].data),
                    np.asarray(lw.wq_b_scale.addressable_shards[i].data),
                )
                wqb_c[i].append(wqb_i)
                wqb_sc[i].append(wqb_si)

                sk_i = np.zeros((batch_tile, 128), dtype=np.float32)
                sk_i[:, :8] = np.asarray(lw.attn_sink.addressable_shards[i].data)[None, :]
                sink_c[i].append(sk_i)

                woa_i, woa_si = pack_fp8_kn(
                    np.asarray(lw.wo_a.addressable_shards[i].data),
                    np.asarray(lw.wo_a_scale.addressable_shards[i].data),
                )
                woa_c[i].append(woa_i)
                woa_sc[i].append(woa_si)

                wob_i, wob_si = pack_fp8_kn(
                    np.asarray(lw.wo_b.addressable_shards[i].data),
                    np.asarray(lw.wo_b_scale.addressable_shards[i].data),
                )
                wob_c[i].append(wob_i)
                wob_sc[i].append(wob_si)

                sw13_i, ss13_i = pack_shared_w13(
                    np.asarray(lw.shared_w1.addressable_shards[i].data),
                    np.asarray(lw.shared_w1_scale.addressable_shards[i].data),
                    np.asarray(lw.shared_w3.addressable_shards[i].data),
                    np.asarray(lw.shared_w3_scale.addressable_shards[i].data),
                )
                sw13_c[i].append(sw13_i)
                ss13_c[i].append(ss13_i)

                sw2_i, ss2_i = pack_shared_w2(
                    np.asarray(lw.shared_w2.addressable_shards[i].data),
                    np.asarray(lw.shared_w2_scale.addressable_shards[i].data),
                )
                sw2_c[i].append(sw2_i)
                ss2_c[i].append(ss2_i)

                rw13_i, rs13_i = _pack_routed_w13_dev_jax(
                    lw.routed_w1.addressable_shards[i].data,
                    lw.routed_w1_scale.addressable_shards[i].data,
                    lw.routed_w3.addressable_shards[i].data,
                    lw.routed_w3_scale.addressable_shards[i].data,
                )
                g_rw13[i].append(rw13_i)
                g_rs13[i].append(rs13_i)

                rw2_i, rs2_i = _pack_routed_w2_dev_jax(
                    lw.routed_w2.addressable_shards[i].data,
                    lw.routed_w2_scale.addressable_shards[i].data,
                )
                g_rw2[i].append(rw2_i)
                g_rs2[i].append(rs2_i)

            if lw.is_kv_source and lw.comp_wkv is not None:
                cw = np.zeros((5120, 1024), dtype=ml_dtypes.bfloat16)
                cw[:, :512] = to_host_np(lw.comp_wkv).T
                if lw.compress_ratio == 2 and lw.comp_wgate is not None:
                    cw[:, 512:] = to_host_np(lw.comp_wgate).T
                comp_w_l.append(cw)
                comp_n_l.append(_bcast_row(to_host_np(lw.comp_norm)))
                idx_wk_l.append(np.ascontiguousarray(to_host_np(lw.idx_wk).T))
                idx_knorm_l.append(_bcast_row(to_host_np(lw.idx_k_norm)))

            if lw.is_index_source and lw.idx_wq_b is not None:
                iwqb_full, iwqb_s_full = pack_fp8_kn(
                    to_host_np(lw.idx_wq_b), to_host_np(lw.idx_wq_b_scale)
                )
                iwproj_full = to_host_np(lw.idx_weights_proj).T
                for i in range(4):
                    c = host_idx * 4 + i
                    idx_wqb_c[i].append(iwqb_full[:, c * 256 : (c + 1) * 256].copy())
                    idx_wqb_sc[i].append(iwqb_s_full[:, c * 256 : (c + 1) * 256].copy())
                    wp_i = np.zeros((5120, 128), dtype=ml_dtypes.bfloat16)
                    wp_i[:, :2] = iwproj_full[:, c * 2 : (c + 1) * 2]
                    idx_wproj_c[i].append(wp_i)

            if lw.has_engram and lw.engram_wkv is not None:
                ewkv_t, ewkv_st = pack_fp8_kn(
                    to_host_np(lw.engram_wkv), to_host_np(lw.engram_wkv_scale)
                )
                eqw = to_host_np(lw.engram_q_weight, dtype=np.float32)
                ekw = to_host_np(lw.engram_k_weight, dtype=np.float32)
                qk_prod = eqw * ekw
                eqk = np.zeros((4, batch_tile, 5120), dtype=np.float32)
                for s in range(4):
                    eqk[s] = qk_prod[s : s + 1]
                eng_qk_l.append(eqk)
                for i in range(4):
                    c = host_idx * 4 + i
                    w_slice = ewkv_t[c * 384 : (c + 1) * 384, :]
                    s_slice = ewkv_st[c * 12 : (c + 1) * 12, :]
                    eng_wkv_c[i].append(
                        np.stack([w_slice[:, b * 5120 : (b + 1) * 5120] for b in range(5)], axis=0).copy()
                    )
                    eng_s_c[i].append(
                        np.stack([s_slice[:, b * 5120 : (b + 1) * 5120] for b in range(5)], axis=0).copy()
                    )

            if free_source_layers:
                layers_mut[lid] = None  # type: ignore[assignment]

        if free_source_layers:
            mw.layers = tuple(layers_mut)  # type: ignore[misc]
            gc.collect()

        rw13_groups.append(
            jax.make_array_from_single_device_arrays(
                (16, 5, 384, 640, 384),
                tp_s,
                [_stack_group5_dev_jax(*g_rw13[i]) for i in range(4)],
            )
        )
        rs13_groups.append(
            jax.make_array_from_single_device_arrays(
                (16, 5, 384, 160, 384),
                tp_s,
                [_stack_group5_dev_jax(*g_rs13[i]) for i in range(4)],
            )
        )
        rw2_groups.append(
            jax.make_array_from_single_device_arrays(
                (16, 5, 384, 24, 5120),
                tp_s,
                [_stack_group5_dev_jax(*g_rw2[i]) for i in range(4)],
            )
        )
        rs2_groups.append(
            jax.make_array_from_single_device_arrays(
                (16, 5, 384, 8, 5120),
                tp_s,
                [_stack_group5_dev_jax(*g_rs2[i]) for i in range(4)],
            )
        )

    return PallasMegakernelWeights(
        cfg=cfg,
        mesh=mesh,
        engram_host=mw.engram_host,
        embed_weight=mw.embed_weight,
        final_norm=_put_rep(final_norm_np),
        lm_head=_put_tp(lm_head_chips),
        eng_wkv=_put_tp([np.stack(eng_wkv_c[i], axis=0) for i in range(4)]),
        eng_s=_put_tp([np.stack(eng_s_c[i], axis=0) for i in range(4)]),
        eng_qk=_put_rep(np.stack(eng_qk_l, axis=0)),
        attn_norm=_put_rep(np.stack(attn_norm_l, axis=0)),
        ffn_norm=_put_rep(np.stack(ffn_norm_l, axis=0)),
        hc_attn_fn=_put_rep(np.stack(hca_fn_l, axis=0)),
        hc_attn_mb=_put_rep(np.stack(hca_mb_l, axis=0)),
        hc_ffn_fn=_put_rep(np.stack(hcf_fn_l, axis=0)),
        hc_ffn_mb=_put_rep(np.stack(hcf_mb_l, axis=0)),
        wqa=_put_rep(np.stack(wqa_l, axis=0)),
        wqa_s=_put_rep(np.stack(wqa_s_l, axis=0)),
        qn=_put_rep(np.stack(qn_l, axis=0)),
        wkv=_put_rep(np.stack(wkv_l, axis=0)),
        wkv_s=_put_rep(np.stack(wkv_s_l, axis=0)),
        kvn=_put_rep(np.stack(kvn_l, axis=0)),
        wqb=_put_tp([np.stack(wqb_c[i], axis=0) for i in range(4)]),
        wqb_s=_put_tp([np.stack(wqb_sc[i], axis=0) for i in range(4)]),
        sink=_put_tp([np.stack(sink_c[i], axis=0) for i in range(4)]),
        woa=_put_tp([np.stack(woa_c[i], axis=0) for i in range(4)]),
        woa_s=_put_tp([np.stack(woa_sc[i], axis=0) for i in range(4)]),
        wob=_put_tp([np.stack(wob_c[i], axis=0) for i in range(4)]),
        wob_s=_put_tp([np.stack(wob_sc[i], axis=0) for i in range(4)]),
        gate_w=_put_rep(np.stack(gate_w_l, axis=0)),
        gate_b=_put_rep(np.stack(gate_b_l, axis=0)),
        sw13=_put_tp([np.stack(sw13_c[i], axis=0) for i in range(4)]),
        ss13=_put_tp([np.stack(ss13_c[i], axis=0) for i in range(4)]),
        sw2=_put_tp([np.stack(sw2_c[i], axis=0) for i in range(4)]),
        ss2=_put_tp([np.stack(ss2_c[i], axis=0) for i in range(4)]),
        comp_w=_put_rep(np.stack(comp_w_l, axis=0)),
        comp_n=_put_rep(np.stack(comp_n_l, axis=0)),
        idx_wk=_put_rep(np.stack(idx_wk_l, axis=0)),
        idx_knorm=_put_rep(np.stack(idx_knorm_l, axis=0)),
        idx_wqb=_put_tp([np.stack(idx_wqb_c[i], axis=0) for i in range(4)]),
        idx_wqb_s=_put_tp([np.stack(idx_wqb_sc[i], axis=0) for i in range(4)]),
        idx_wproj=_put_tp([np.stack(idx_wproj_c[i], axis=0) for i in range(4)]),
        rw13_groups=tuple(rw13_groups),
        rs13_groups=tuple(rs13_groups),
        rw2_groups=tuple(rw2_groups),
        rs2_groups=tuple(rs2_groups),
    )


@dataclass
class PallasMegakernelDecodeState:
    """Mutable KV/Compressor/Indexer decode cache state for `DSV41PallasMegakernel`."""

    max_comp: int
    batch_tile: int
    swa_kv_all: jax.Array       # [40, 128, 512] bf16
    swa_pos: jax.Array          # [B_tile, 128] int32
    comp_kv_all: jax.Array      # [4, max_comp, 512] bf16
    ik_f8_all: jax.Array        # [4, max_comp, 128] f8
    ik_sc_all: jax.Array        # [4, B_tile, max_comp] f32
    comp_tail_all: jax.Array    # [4, B_tile, 1024] f32
    token_history: List[int]
    enable_engram: bool = True


class DSV41PallasMegakernel:
    """Single-`pallas_call` 40-layer persistent decode megakernel for DeepSeek-V4.1-Flash on TPU v6e-16."""

    def __init__(
        self,
        weights: PallasMegakernelWeights,
        batch_tile: int = 8,
    ) -> None:
        self.weights = weights
        self.cfg = weights.cfg
        self.mesh = weights.mesh
        self.batch_tile = batch_tile
        self.host_idx = jax.process_index()
        self.local_devs = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))[
            self.host_idx * 4 : (self.host_idx + 1) * 4
        ]
        self.rep_s = NamedSharding(self.mesh, P())
        self.tp_s = NamedSharding(self.mesh, P("tp"))

        consts = build_pallas_constants(self.cfg, batch_tile=batch_tile)
        self.p_swap = self.replicate(consts["p_swap"])
        self.s32 = self.replicate(consts["s32"])
        self.inv_freq_plain = self.replicate(consts["inv_freq_plain"])
        self.inv_freq_yarn = self.replicate(consts["inv_freq_yarn"])
        self.rope_sign = self.replicate(consts["rope_sign"])

        pm_init_np = np.zeros((4, batch_tile, 128), dtype=np.float32)
        pm_init_np[0, :, :] = 1.0
        self.pm_init = self.replicate(pm_init_np)
        self.zero_eng_rows = jax.make_array_from_single_device_arrays(
            (16, 2, batch_tile, 384),
            self.tp_s,
            [
                jax.device_put(np.zeros((1, 2, batch_tile, 384), dtype=ml_dtypes.bfloat16), d)
                for d in self.local_devs
            ],
        )
        self._compiled_steps: Dict[Tuple[int, int], Any] = {}
        self._compile_embed()

    def replicate(self, arr: np.ndarray) -> jax.Array:
        a = np.ascontiguousarray(arr)
        return jax.make_array_from_single_device_arrays(
            a.shape, self.rep_s, [jax.device_put(a, d) for d in self.local_devs]
        )

    def _compile_embed(self) -> None:
        mesh = self.mesh
        b_tile = self.batch_tile

        @functools.partial(
            jax.jit,
            in_shardings=(NamedSharding(mesh, P(None, "tp")), self.rep_s),
            out_shardings=self.rep_s,
        )
        def _embed_4stream(embed_w: jax.Array, tids: jax.Array) -> jax.Array:
            def _local(w_part, t):
                part = w_part[t]  # [B_tile, 320]
                emb = jax.lax.all_gather(part, "tp", axis=-1, tiled=True)  # [B_tile, 5120]
                return jnp.broadcast_to(emb[None, :, :], (4, b_tile, 5120))

            return jax.shard_map(
                _local,
                mesh=mesh,
                in_specs=(P(None, "tp"), P()),
                out_specs=P(),
                check_vma=False,
            )(embed_w, tids)

        @functools.partial(
            jax.jit,
            in_shardings=(
                self.rep_s,
                self.rep_s,
                self.rep_s,
                self.rep_s,
                self.rep_s,
                self.rep_s,
                self.rep_s,
            ),
            out_shardings=(self.rep_s, self.rep_s, self.rep_s),
        )
        def _commit_spec_state(
            old_swa_kv: jax.Array,
            old_swa_pos: jax.Array,
            new_swa_kv: jax.Array,
            new_swa_pos: jax.Array,
            new_comp_tail: jax.Array,
            keep_slot_mask_128: jax.Array,
            last_row_idx: jax.Array,
        ) -> Tuple[jax.Array, jax.Array, jax.Array]:
            mask_bool = keep_slot_mask_128 != 0
            swa_kv_out = jnp.where(mask_bool[None, :, None], new_swa_kv, old_swa_kv)
            swa_pos_out = jnp.where(mask_bool[None, :], new_swa_pos, old_swa_pos)
            r_last = last_row_idx[0]
            tail_row = jax.lax.dynamic_slice_in_dim(new_comp_tail, r_last, 1, axis=1)
            comp_tail_out = jnp.broadcast_to(tail_row, new_comp_tail.shape)
            return swa_kv_out, swa_pos_out, comp_tail_out

        self._embed_4stream = _embed_4stream
        self._commit_spec_state = _commit_spec_state

    def init_decode_state(
        self,
        max_comp: int = 512,
        enable_engram: bool = True,
    ) -> PallasMegakernelDecodeState:
        b_tile = self.batch_tile
        return PallasMegakernelDecodeState(
            max_comp=max_comp,
            batch_tile=b_tile,
            swa_kv_all=self.replicate(np.zeros((40, 128, 512), dtype=ml_dtypes.bfloat16)),
            swa_pos=self.replicate(np.full((b_tile, 128), -1, dtype=np.int32)),
            comp_kv_all=self.replicate(np.zeros((4, max_comp, 512), dtype=ml_dtypes.bfloat16)),
            ik_f8_all=self.replicate(np.zeros((4, max_comp, 128), dtype=ml_dtypes.float8_e4m3fn)),
            ik_sc_all=self.replicate(np.ones((4, b_tile, max_comp), dtype=np.float32)),
            comp_tail_all=self.replicate(np.zeros((4, b_tile, 1024), dtype=np.float32)),
            token_history=[],
            enable_engram=enable_engram,
        )

    def state_from_jax_prefill(
        self,
        jax_state: Dict[str, Any],
        max_comp: int = 512,
    ) -> PallasMegakernelDecodeState:
        """Convert a `DSV41JaxEngine.prefill()` state dict into `PallasMegakernelDecodeState`."""
        b_tile = self.batch_tile
        swa_kv_np = np.stack([to_host_np(x) for x in jax_state["swa_kv"]], axis=0)
        swa_pos_1d = to_host_np(jax_state["swa_pos"])
        swa_pos_np = np.broadcast_to(swa_pos_1d[None, :], (b_tile, 128)).copy()

        comp_kv_np = np.zeros((4, max_comp, 512), dtype=ml_dtypes.bfloat16)
        ik_f8_np = np.zeros((4, max_comp, 128), dtype=ml_dtypes.float8_e4m3fn)
        ik_sc_np = np.ones((4, b_tile, max_comp), dtype=np.float32)
        comp_tail_np = np.zeros((4, b_tile, 1024), dtype=np.float32)

        for idx_kv, lid in enumerate(self.cfg.kv_source_layer_ids):
            ckv = to_host_np(jax_state["comp_kv"][lid])
            n_copy = min(max_comp, ckv.shape[0])
            comp_kv_np[idx_kv, :n_copy] = ckv[:n_copy]
            ikf = to_host_np(jax_state["idx_k_f8"][lid])
            ik_f8_np[idx_kv, :n_copy] = ikf[:n_copy]
            iks = to_host_np(jax_state["idx_k_sc"][lid])[:, 0]
            ik_sc_np[idx_kv, :, :n_copy] = iks[None, :n_copy]
            ct = to_host_np(jax_state["comp_tail"][lid])
            comp_tail_np[idx_kv] = ct[:1]

        return PallasMegakernelDecodeState(
            max_comp=max_comp,
            batch_tile=b_tile,
            swa_kv_all=self.replicate(swa_kv_np),
            swa_pos=self.replicate(swa_pos_np),
            comp_kv_all=self.replicate(comp_kv_np),
            ik_f8_all=self.replicate(ik_f8_np),
            ik_sc_all=self.replicate(ik_sc_np),
            comp_tail_all=self.replicate(comp_tail_np),
            token_history=list(jax_state["token_history"]),
            enable_engram=bool(jax_state.get("enable_engram", True)),
        )

    def _gather_engram_rows_tp(self, windows_np: np.ndarray, active_b: int) -> jax.Array:
        """Gather local Engram rows for Layers 1 and 14 (`[16, 2, B_tile, 384]` bf16)."""
        b_tile = self.batch_tile
        chips = self.weights.engram_host.fast_gather_both_chips(
            windows_np[:active_b], b_tile=b_tile
        )
        return jax.make_array_from_single_device_arrays(
            (16, 2, b_tile, 384),
            self.tp_s,
            [jax.device_put(chips[i], self.local_devs[i]) for i in range(4)],
        )

    def _get_step_kernel(self, active_b: int, max_comp: int, spec_k: int = 0) -> Any:
        key = (active_b, max_comp, spec_k)
        if key in self._compiled_steps:
            return self._compiled_steps[key]

        b_tile = self.batch_tile

        def _megakernel(
            # 10 VMEM inputs (0..9)
            res_in_ref,
            pm_in_ref,
            pos_ref,
            inv_freq_plain_ref,
            inv_freq_yarn_ref,
            rope_sign_ref,
            p_swap_ref,
            s32_ref,
            eng_rows_ref,
            swa_pos_in_ref,
            # 5 HBM KV-state inputs (10..14)
            swa_kv_in_hbm,
            comp_kv_in_hbm,
            ik_f8_in_hbm,
            ik_sc_in_hbm,
            comp_tail_in_hbm,
            # 37 HBM non-routed weight inputs (15..51)
            eng_wkv_hbm,
            eng_s_hbm,
            eng_qk_hbm,
            attn_norm_hbm,
            ffn_norm_hbm,
            hc_attn_fn_hbm,
            hc_attn_mb_hbm,
            hc_ffn_fn_hbm,
            hc_ffn_mb_hbm,
            wqa_hbm,
            wqa_s_hbm,
            qn_hbm,
            wkv_hbm,
            wkv_s_hbm,
            kvn_hbm,
            wqb_hbm,
            wqb_s_hbm,
            sink_hbm,
            woa_hbm,
            woa_s_hbm,
            wob_hbm,
            wob_s_hbm,
            gate_w_hbm,
            gate_b_hbm,
            sw13_hbm,
            ss13_hbm,
            sw2_hbm,
            ss2_hbm,
            comp_w_hbm,
            comp_n_hbm,
            idx_wk_hbm,
            idx_knorm_hbm,
            idx_wqb_hbm,
            idx_wqb_s_hbm,
            idx_wproj_hbm,
            final_norm_hbm,
            lm_head_hbm,
            # 32 HBM routed-expert group inputs (52..83)
            rw13_g0, rs13_g0, rw2_g0, rs2_g0,
            rw13_g1, rs13_g1, rw2_g1, rs2_g1,
            rw13_g2, rs13_g2, rw2_g2, rs2_g2,
            rw13_g3, rs13_g3, rw2_g3, rs2_g3,
            rw13_g4, rs13_g4, rw2_g4, rs2_g4,
            rw13_g5, rs13_g5, rw2_g5, rs2_g5,
            rw13_g6, rs13_g6, rw2_g6, rs2_g6,
            rw13_g7, rs13_g7, rw2_g7, rs2_g7,
            # 8 Outputs (3 VMEM + 5 HBM)
            logits_out_ref,
            dspark_out_ref,
            swa_pos_out_ref,
            swa_kv_out_hbm,
            comp_kv_out_hbm,
            ik_f8_out_hbm,
            ik_sc_out_hbm,
            comp_tail_out_hbm,
            # Scratches
            res_vmem,
            pm_vmem,
            swa_kv_vmem,
            comp_kv_vmem,
            ik_f8_vmem,
            ik_sc_vmem,
            comp_tail_vmem,
            csa_mask_vmem,
            *common_scratches,
        ):
            collectives16.barrier()
            res_vmem[...] = res_in_ref[...]
            pm_vmem[...] = pm_in_ref[...]
            swa_pos_out_ref[...] = swa_pos_in_ref[...]
            csa_mask_vmem[...] = jnp.zeros((b_tile, max_comp), jnp.int32)

            sem = common_scratches[-5]

            def _load_kv_slot(slot_idx: int):
                c1 = pltpu.make_async_copy(comp_kv_in_hbm.at[slot_idx], comp_kv_vmem, sem.at[0])
                c2 = pltpu.make_async_copy(ik_f8_in_hbm.at[slot_idx], ik_f8_vmem, sem.at[1])
                c3 = pltpu.make_async_copy(ik_sc_in_hbm.at[slot_idx], ik_sc_vmem, sem.at[2])
                c4 = pltpu.make_async_copy(comp_tail_in_hbm.at[slot_idx], comp_tail_vmem, sem.at[3])
                c1.start(); c2.start(); c3.start(); c4.start()
                c1.wait(); c2.wait(); c3.wait(); c4.wait()

            def _store_kv_slot(slot_idx: int):
                c1 = pltpu.make_async_copy(comp_kv_vmem, comp_kv_out_hbm.at[slot_idx], sem.at[0])
                c2 = pltpu.make_async_copy(ik_f8_vmem, ik_f8_out_hbm.at[slot_idx], sem.at[1])
                c3 = pltpu.make_async_copy(ik_sc_vmem, ik_sc_out_hbm.at[slot_idx], sem.at[2])
                c4 = pltpu.make_async_copy(comp_tail_vmem, comp_tail_out_hbm.at[slot_idx], sem.at[3])
                c1.start(); c2.start(); c3.start(); c4.start()
                c1.wait(); c2.wait(); c3.wait(); c4.wait()

            def _exec_one(
                lid,
                sub_idx,
                c_ratio: int,
                is_kv_src: bool,
                is_idx_src: bool,
                has_eng: bool,
                kv_idx: int,
                idx_idx: int,
                eng_idx: int,
                g_rw13,
                g_rs13,
                g_rw2,
                g_rs2,
            ):
                c_swa_in = pltpu.make_async_copy(swa_kv_in_hbm.at[lid], swa_kv_vmem, sem.at[15])
                c_swa_in.start()
                c_swa_in.wait()

                inv_f = inv_freq_plain_ref if c_ratio == 0 else inv_freq_yarn_ref
                execute_pallas_layer_in_vmem(
                    active_b,
                    c_ratio,
                    is_kv_src,
                    is_idx_src,
                    has_eng,
                    res_vmem,
                    pm_vmem,
                    pos_ref,
                    inv_f,
                    rope_sign_ref,
                    p_swap_ref,
                    s32_ref,
                    eng_rows_ref.at[eng_idx],
                    eng_wkv_hbm.at[eng_idx],
                    eng_s_hbm.at[eng_idx],
                    eng_qk_hbm.at[eng_idx],
                    attn_norm_hbm.at[lid],
                    ffn_norm_hbm.at[lid],
                    hc_attn_fn_hbm.at[lid],
                    hc_attn_mb_hbm.at[lid],
                    hc_ffn_fn_hbm.at[lid],
                    hc_ffn_mb_hbm.at[lid],
                    wqa_hbm.at[lid],
                    wqa_s_hbm.at[lid],
                    qn_hbm.at[lid],
                    wkv_hbm.at[lid],
                    wkv_s_hbm.at[lid],
                    kvn_hbm.at[lid],
                    wqb_hbm.at[lid],
                    wqb_s_hbm.at[lid],
                    sink_hbm.at[lid],
                    woa_hbm.at[lid],
                    woa_s_hbm.at[lid],
                    wob_hbm.at[lid],
                    wob_s_hbm.at[lid],
                    gate_w_hbm.at[lid],
                    gate_b_hbm.at[lid],
                    sw13_hbm.at[lid],
                    ss13_hbm.at[lid],
                    sw2_hbm.at[lid],
                    ss2_hbm.at[lid],
                    g_rw13,
                    g_rs13,
                    g_rw2,
                    g_rs2,
                    comp_w_hbm.at[kv_idx],
                    comp_n_hbm.at[kv_idx],
                    idx_wk_hbm.at[kv_idx],
                    idx_knorm_hbm.at[kv_idx],
                    idx_wqb_hbm.at[idx_idx],
                    idx_wqb_s_hbm.at[idx_idx],
                    idx_wproj_hbm.at[idx_idx],
                    swa_kv_vmem,
                    swa_pos_out_ref,
                    comp_kv_vmem,
                    ik_f8_vmem,
                    ik_sc_vmem,
                    comp_tail_vmem,
                    csa_mask_vmem,
                    *common_scratches,
                    iter_base=lid * 32,
                    sub_idx=sub_idx,
                    spec_k=spec_k,
                )

                c_swa_out = pltpu.make_async_copy(swa_kv_vmem, swa_kv_out_hbm.at[lid], sem.at[15])
                c_swa_out.start()
                c_swa_out.wait()

            # Group 0: Layers 0..4
            _exec_one(0, 0, 0, False, False, False, 0, 0, 0, rw13_g0, rs13_g0, rw2_g0, rs2_g0)
            _exec_one(1, 1, 0, False, False, True, 0, 0, 0, rw13_g0, rs13_g0, rw2_g0, rs2_g0)
            _load_kv_slot(0)
            _exec_one(2, 2, 2, True, True, False, 0, 0, 0, rw13_g0, rs13_g0, rw2_g0, rs2_g0)
            _store_kv_slot(0)

            @pl.loop(3, 5)
            def _g0_loop(sub):
                _exec_one(sub, sub, 2, False, False, False, 0, 0, 0, rw13_g0, rs13_g0, rw2_g0, rs2_g0)

            # Group 1: Layers 5..9
            @pl.loop(0, 3)
            def _g1_loop(sub):
                _exec_one(5 + sub, sub, 2, False, False, False, 0, 0, 0, rw13_g1, rs13_g1, rw2_g1, rs2_g1)

            _load_kv_slot(1)
            _exec_one(8, 3, 2, True, True, False, 1, 1, 0, rw13_g1, rs13_g1, rw2_g1, rs2_g1)
            _store_kv_slot(1)
            _exec_one(9, 4, 2, False, False, False, 1, 1, 0, rw13_g1, rs13_g1, rw2_g1, rs2_g1)

            # Group 2: Layers 10..14
            @pl.loop(0, 4)
            def _g2_loop(sub):
                _exec_one(10 + sub, sub, 2, False, False, False, 1, 1, 0, rw13_g2, rs13_g2, rw2_g2, rs2_g2)

            _load_kv_slot(2)
            _exec_one(14, 4, 2, True, True, True, 2, 2, 1, rw13_g2, rs13_g2, rw2_g2, rs2_g2)
            _store_kv_slot(2)

            # Group 3: Layers 15..19
            @pl.loop(0, 5)
            def _g3_loop(sub):
                _exec_one(15 + sub, sub, 2, False, False, False, 2, 2, 0, rw13_g3, rs13_g3, rw2_g3, rs2_g3)

            # Group 4: Layers 20..24
            _load_kv_slot(3)
            _exec_one(20, 0, 1, True, True, False, 3, 3, 0, rw13_g4, rs13_g4, rw2_g4, rs2_g4)
            _store_kv_slot(3)

            @pl.loop(1, 4)
            def _g4_loop(sub):
                _exec_one(20 + sub, sub, 1, False, False, False, 3, 3, 0, rw13_g4, rs13_g4, rw2_g4, rs2_g4)

            _exec_one(24, 4, 1, False, True, False, 3, 4, 0, rw13_g4, rs13_g4, rw2_g4, rs2_g4)

            # Group 5: Layers 25..29
            @pl.loop(0, 3)
            def _g5_loop(sub):
                _exec_one(25 + sub, sub, 1, False, False, False, 3, 4, 0, rw13_g5, rs13_g5, rw2_g5, rs2_g5)

            _exec_one(28, 3, 1, False, True, False, 3, 5, 0, rw13_g5, rs13_g5, rw2_g5, rs2_g5)
            _exec_one(29, 4, 1, False, False, False, 3, 5, 0, rw13_g5, rs13_g5, rw2_g5, rs2_g5)

            # Group 6: Layers 30..34
            @pl.loop(0, 2)
            def _g6_loop_a(sub):
                _exec_one(30 + sub, sub, 1, False, False, False, 3, 5, 0, rw13_g6, rs13_g6, rw2_g6, rs2_g6)

            _exec_one(32, 2, 1, False, True, False, 3, 6, 0, rw13_g6, rs13_g6, rw2_g6, rs2_g6)

            @pl.loop(3, 5)
            def _g6_loop_b(sub):
                _exec_one(30 + sub, sub, 1, False, False, False, 3, 6, 0, rw13_g6, rs13_g6, rw2_g6, rs2_g6)

            # Group 7: Layers 35..39 (with DSpark target hidden capture before 37, 38, 39)
            _exec_one(35, 0, 1, False, False, False, 3, 6, 0, rw13_g7, rs13_g7, rw2_g7, rs2_g7)
            _exec_one(36, 1, 1, False, True, False, 3, 7, 0, rw13_g7, rs13_g7, rw2_g7, rs2_g7)

            @pl.loop(2, 5)
            def _g7_loop(sub):
                h_mean = (
                    (
                        res_vmem[0].astype(jnp.float32)
                        + res_vmem[1].astype(jnp.float32)
                        + res_vmem[2].astype(jnp.float32)
                        + res_vmem[3].astype(jnp.float32)
                    )
                    * 0.25
                ).astype(jnp.bfloat16)
                dspark_out_ref[sub - 2] = h_mean
                _exec_one(35 + sub, sub, 1, False, False, False, 3, 7, 0, rw13_g7, rs13_g7, rw2_g7, rs2_g7)

            if spec_k > 0:
                pos_all = pos_ref[...]
                col_iota_128 = jax.lax.broadcasted_iota(jnp.int32, (b_tile, 128), 1)
                swa_p_final = swa_pos_out_ref[...]
                for r_kv in range(spec_k):
                    row_r_128 = jax.lax.broadcasted_iota(jnp.int32, (b_tile, 128), 0) == r_kv
                    pr_vec = jnp.sum(jnp.where(row_r_128, pos_all, 0), axis=0, keepdims=True)
                    slot_vec_r = jnp.where(pr_vec >= 0, pr_vec % 128, jnp.int32(-1))
                    swa_p_final = jnp.where(col_iota_128 == slot_vec_r, pr_vec, swa_p_final)
                swa_pos_out_ref[...] = swa_p_final

            # Final mHC stream collapse + Final RMSNorm + Sharded LM Head inside the same pallas_call
            norm_vmem = common_scratches[3]
            xin_vmem = common_scratches[6]
            x_norm_vmem = common_scratches[7]
            comp_w_vmem = common_scratches[22]

            c_fnorm = pltpu.make_async_copy(final_norm_hbm, norm_vmem, sem.at[0])
            c_fnorm.start()
            h_coll_f32 = jnp.zeros((b_tile, 5120), jnp.float32)
            for s in range(4):
                h_coll_f32 += res_vmem[s].astype(jnp.float32) * pm_vmem[s, :, :1]
            xin_vmem[...] = h_coll_f32.astype(jnp.bfloat16)
            c_fnorm.wait()
            rms_norm_vmem(xin_vmem, norm_vmem, x_norm_vmem)

            for v_blk in range(8):
                c_lm = pltpu.make_async_copy(lm_head_hbm.at[v_blk], comp_w_vmem, sem.at[0])
                c_lm.start()
                c_lm.wait()
                acc_logits = jnp.zeros((b_tile, 1024), jnp.float32)
                for k_st in range(0, 5120, 512):
                    x_t = x_norm_vmem[:, pl.ds(k_st, 512)]
                    w_t = comp_w_vmem[pl.ds(k_st, 512), :]
                    acc_logits += jnp.dot(x_t, w_t, preferred_element_type=jnp.float32)
                logits_out_ref[:, pl.ds(v_blk * 1024, 1024)] = acc_logits

        in_specs = [
            *(pl.BlockSpec(memory_space=pltpu.VMEM) for _ in range(10)),
            *(pl.BlockSpec(memory_space=pl.ANY) for _ in range(74)),
        ]
        out_shapes = (
            jax.ShapeDtypeStruct((b_tile, 8192), jnp.float32),
            jax.ShapeDtypeStruct((3, b_tile, 5120), jnp.bfloat16),
            jax.ShapeDtypeStruct((b_tile, 128), jnp.int32),
            jax.ShapeDtypeStruct((40, 128, 512), jnp.bfloat16),
            jax.ShapeDtypeStruct((4, max_comp, 512), jnp.bfloat16),
            jax.ShapeDtypeStruct((4, max_comp, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((4, b_tile, max_comp), jnp.float32),
            jax.ShapeDtypeStruct((4, b_tile, 1024), jnp.float32),
        )
        out_specs = (
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        )
        megakernel_scratches = [
            pltpu.VMEM((4, b_tile, 5120), jnp.bfloat16),     # res_vmem
            pltpu.VMEM((4, b_tile, 128), jnp.float32),       # pm_vmem
            pltpu.VMEM((128, 512), jnp.bfloat16),            # swa_kv_vmem
            pltpu.VMEM((max_comp, 512), jnp.bfloat16),       # comp_kv_vmem
            pltpu.VMEM((max_comp, 128), jnp.float8_e4m3fn),  # ik_f8_vmem
            pltpu.VMEM((b_tile, max_comp), jnp.float32),     # ik_sc_vmem
            pltpu.VMEM((b_tile, 1024), jnp.float32),         # comp_tail_vmem
            pltpu.VMEM((b_tile, max_comp), jnp.int32),       # csa_mask_vmem
            *_build_common_scratch_shapes(b_tile),
        ]

        p_call = pl.pallas_call(
            _megakernel,
            out_shape=out_shapes,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=megakernel_scratches,
            compiler_params=pltpu.CompilerParams(
                collective_id=0, vmem_limit_bytes=112 * 1024 * 1024
            ),
        )

        tp_arg_indices = {
            8,               # eng_rows_ref
            15, 16,          # eng_wkv, eng_s
            30, 31, 32,      # wqb, wqb_s, sink
            33, 34, 35, 36,  # woa, woa_s, wob, wob_s
            39, 40, 41, 42,  # sw13, ss13, sw2, ss2
            47, 48, 49,      # idx_wqb, idx_wqb_s, idx_wproj
            51,              # lm_head
            *range(52, 84),  # rw13/rs13/rw2/rs2 groups 0..7
        }
        smap_in_specs = tuple(
            P("tp") if idx in tp_arg_indices else P()
            for idx in range(84)
        )

        def _smap_wrapper(*args):
            unwrapped = [
                args[idx][0] if idx in tp_arg_indices else args[idx]
                for idx in range(84)
            ]
            (
                logits_local,
                dspark_h,
                swa_pos_o,
                swa_kv_o,
                comp_kv_o,
                ik_f8_o,
                ik_sc_o,
                comp_tail_o,
            ) = p_call(*unwrapped)
            logits_8080 = logits_local[:, :8080]
            logits_full = jax.lax.all_gather(logits_8080, "tp", axis=-1, tiled=True)
            top1_ids = jnp.argmax(logits_full, axis=-1).astype(jnp.int32)
            return (
                logits_full,
                top1_ids,
                dspark_h,
                swa_pos_o,
                swa_kv_o,
                comp_kv_o,
                ik_f8_o,
                ik_sc_o,
                comp_tail_o,
            )

        fn = jax.jit(
            jax.shard_map(
                _smap_wrapper,
                mesh=self.mesh,
                in_specs=smap_in_specs,
                out_specs=tuple(P() for _ in range(9)),
                check_vma=False,
            )
        )
        self._compiled_steps[key] = fn
        return fn

    def _weight_args(self) -> Tuple[jax.Array, ...]:
        w = self.weights
        group_args: List[jax.Array] = []
        for g in range(8):
            group_args.extend(
                [
                    w.rw13_groups[g],
                    w.rs13_groups[g],
                    w.rw2_groups[g],
                    w.rs2_groups[g],
                ]
            )
        return (
            w.eng_wkv,
            w.eng_s,
            w.eng_qk,
            w.attn_norm,
            w.ffn_norm,
            w.hc_attn_fn,
            w.hc_attn_mb,
            w.hc_ffn_fn,
            w.hc_ffn_mb,
            w.wqa,
            w.wqa_s,
            w.qn,
            w.wkv,
            w.wkv_s,
            w.kvn,
            w.wqb,
            w.wqb_s,
            w.sink,
            w.woa,
            w.woa_s,
            w.wob,
            w.wob_s,
            w.gate_w,
            w.gate_b,
            w.sw13,
            w.ss13,
            w.sw2,
            w.ss2,
            w.comp_w,
            w.comp_n,
            w.idx_wk,
            w.idx_knorm,
            w.idx_wqb,
            w.idx_wqb_s,
            w.idx_wproj,
            w.final_norm,
            w.lm_head,
            *group_args,
        )

    def run_raw_step(
        self,
        active_b: int,
        res_in: jax.Array,
        pos_vmem: jax.Array,
        eng_rows_tp: jax.Array,
        state: PallasMegakernelDecodeState,
        spec_k: int = 0,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, PallasMegakernelDecodeState]:
        fn = self._get_step_kernel(
            active_b=active_b, max_comp=state.max_comp, spec_k=spec_k
        )
        (
            logits_full,
            top1_ids,
            dspark_h,
            swa_pos_o,
            swa_kv_o,
            comp_kv_o,
            ik_f8_o,
            ik_sc_o,
            comp_tail_o,
        ) = fn(
            res_in,
            self.pm_init,
            pos_vmem,
            self.inv_freq_plain,
            self.inv_freq_yarn,
            self.rope_sign,
            self.p_swap,
            self.s32,
            eng_rows_tp,
            state.swa_pos,
            state.swa_kv_all,
            state.comp_kv_all,
            state.ik_f8_all,
            state.ik_sc_all,
            state.comp_tail_all,
            *self._weight_args(),
        )
        new_state = PallasMegakernelDecodeState(
            max_comp=state.max_comp,
            batch_tile=state.batch_tile,
            swa_kv_all=swa_kv_o,
            swa_pos=swa_pos_o,
            comp_kv_all=comp_kv_o,
            ik_f8_all=ik_f8_o,
            ik_sc_all=ik_sc_o,
            comp_tail_all=comp_tail_o,
            token_history=state.token_history,
            enable_engram=state.enable_engram,
        )
        return logits_full, top1_ids, dspark_h, new_state

    def decode_step(
        self,
        next_token_id: int,
        state: PallasMegakernelDecodeState,
        enable_engram: Optional[bool] = None,
        return_dspark: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], PallasMegakernelDecodeState]:
        """Execute one 40-layer Pallas megakernel decode step for `next_token_id`."""
        if enable_engram is None:
            enable_engram = state.enable_engram
        b_tile = self.batch_tile
        hist = state.token_history
        p_cur = len(hist)
        hist.append(int(next_token_id))

        eng_rows_tp = self.zero_eng_rows
        if enable_engram:
            win = np.array(
                [[hist[p_cur - s] if (p_cur - s) >= 0 else -1 for s in range(4)]],
                dtype=np.int64,
            )
            eng_rows_tp = self._gather_engram_rows_tp(win, active_b=1)

        tids_np = np.zeros((b_tile,), dtype=np.int32)
        tids_np[0] = int(next_token_id)
        pos_np = np.zeros((b_tile, 128), dtype=np.int32)
        pos_np[0, :] = p_cur

        res_in = self._embed_4stream(self.weights.embed_weight, self.replicate(tids_np))
        logits_full, _, dspark_h, new_state = self.run_raw_step(
            active_b=1,
            res_in=res_in,
            pos_vmem=self.replicate(pos_np),
            eng_rows_tp=eng_rows_tp,
            state=state,
        )
        logits_np = np.asarray(logits_full.addressable_shards[0].data, dtype=np.float32)[:1]
        dspark_np = (
            np.asarray(dspark_h.addressable_shards[0].data, dtype=np.float32)[:, :1, :]
            if return_dspark
            else None
        )
        return logits_np, dspark_np, new_state

    def decode_step_top1(
        self,
        next_token_id: int,
        state: PallasMegakernelDecodeState,
        enable_engram: Optional[bool] = None,
    ) -> Tuple[int, PallasMegakernelDecodeState]:
        """Execute one 40-layer Pallas megakernel decode step and return only the greedy `top-1` token ID."""
        if enable_engram is None:
            enable_engram = state.enable_engram
        b_tile = self.batch_tile
        hist = state.token_history
        p_cur = len(hist)
        hist.append(int(next_token_id))

        eng_rows_tp = self.zero_eng_rows
        if enable_engram:
            win = np.array(
                [[hist[p_cur - s] if (p_cur - s) >= 0 else -1 for s in range(4)]],
                dtype=np.int64,
            )
            eng_rows_tp = self._gather_engram_rows_tp(win, active_b=1)

        tids_np = np.zeros((b_tile,), dtype=np.int32)
        tids_np[0] = int(next_token_id)
        pos_np = np.zeros((b_tile, 128), dtype=np.int32)
        pos_np[0, :] = p_cur

        res_in = self._embed_4stream(self.weights.embed_weight, self.replicate(tids_np))
        _, top1_ids, _, new_state = self.run_raw_step(
            active_b=1,
            res_in=res_in,
            pos_vmem=self.replicate(pos_np),
            eng_rows_tp=eng_rows_tp,
            state=state,
        )
        next_tok = int(np.asarray(top1_ids.addressable_shards[0].data)[0])
        return next_tok, new_state

    def verify_speculative_step(
        self,
        candidate_token_ids: Sequence[int],
        state: PallasMegakernelDecodeState,
        enable_engram: Optional[bool] = None,
        spec_k: int = 5,
    ) -> Tuple[List[int], int, jax.Array, PallasMegakernelDecodeState]:
        """Verify `candidate_token_ids` (`[root_tok, d_0, ..., d_{m-1}]`, `1 <= len <= spec_k`) in ONE 40-layer Pallas call.

        Returns:
          - `emitted_tokens`: `[d_0, ..., d_{n_acc - 1}, bonus_tok]` (`n_acc + 1` greedy tokens).
          - `n_acc`: number of accepted draft tokens (`0 <= n_acc <= len(candidate_token_ids) - 1`).
          - `dspark_h`: `[3, b_tile, 5120]` BF16 target hidden states at `[37, 38, 39]` for all verified rows.
          - `committed_state`: `PallasMegakernelDecodeState` committed to exactly `n_keep = 1 + n_acc` tokens.
        """
        if enable_engram is None:
            enable_engram = state.enable_engram
        b_tile = self.batch_tile
        k_actual = len(candidate_token_ids)
        assert 1 <= k_actual <= spec_k, f"Expected 1..{spec_k} tokens, got {k_actual}"

        hist_base = list(state.token_history)
        p_0 = len(hist_base)
        hist_ext = hist_base + [int(t) for t in candidate_token_ids]

        eng_rows_tp = self.zero_eng_rows
        if enable_engram:
            wins = np.full((k_actual, 4), -1, dtype=np.int64)
            for r in range(k_actual):
                p_r = p_0 + r
                for s in range(4):
                    if (p_r - s) >= 0:
                        wins[r, s] = hist_ext[p_r - s]
            eng_rows_tp = self._gather_engram_rows_tp(wins, active_b=k_actual)

        tids_np = np.zeros((b_tile,), dtype=np.int32)
        tids_np[:k_actual] = np.asarray(candidate_token_ids, dtype=np.int32)
        pos_np = np.full((b_tile, 128), -1, dtype=np.int32)
        for r in range(k_actual):
            pos_np[r, :] = p_0 + r

        res_in = self._embed_4stream(self.weights.embed_weight, self.replicate(tids_np))
        _, top1_ids, dspark_h, cand_state = self.run_raw_step(
            active_b=0,
            res_in=res_in,
            pos_vmem=self.replicate(pos_np),
            eng_rows_tp=eng_rows_tp,
            state=state,
            spec_k=spec_k,
        )
        preds = np.asarray(top1_ids.addressable_shards[0].data, dtype=np.int32)[:k_actual]

        n_acc = 0
        for j in range(k_actual - 1):
            if int(candidate_token_ids[j + 1]) == int(preds[j]):
                n_acc += 1
            else:
                break
        n_keep = 1 + n_acc
        emitted = [int(candidate_token_ids[1 + j]) for j in range(n_acc)] + [
            int(preds[n_acc])
        ]

        if k_actual == 1:
            swa_kv_c = cand_state.swa_kv_all
            swa_pos_c = cand_state.swa_pos
            comp_tail_c = cand_state.comp_tail_all
        else:
            keep_mask_np = np.ones((128,), dtype=np.int32)
            for r in range(n_keep, k_actual):
                keep_mask_np[(p_0 + r) % 128] = 0
            for r in range(n_keep):
                keep_mask_np[(p_0 + r) % 128] = 1
            last_idx_np = np.array([n_keep - 1], dtype=np.int32)
            swa_kv_c, swa_pos_c, comp_tail_c = self._commit_spec_state(
                state.swa_kv_all,
                state.swa_pos,
                cand_state.swa_kv_all,
                cand_state.swa_pos,
                cand_state.comp_tail_all,
                self.replicate(keep_mask_np),
                self.replicate(last_idx_np),
            )

        committed_state = PallasMegakernelDecodeState(
            max_comp=state.max_comp,
            batch_tile=state.batch_tile,
            swa_kv_all=swa_kv_c,
            swa_pos=swa_pos_c,
            comp_kv_all=cand_state.comp_kv_all,
            ik_f8_all=cand_state.ik_f8_all,
            ik_sc_all=cand_state.ik_sc_all,
            comp_tail_all=comp_tail_c,
            token_history=hist_ext[: p_0 + n_keep],
            enable_engram=state.enable_engram,
        )
        return emitted, n_acc, dspark_h, committed_state

    def prefill_and_logits(
        self,
        token_ids_np: np.ndarray,
        max_comp: int = 512,
        enable_engram: bool = True,
    ) -> Tuple[np.ndarray, PallasMegakernelDecodeState]:
        """Step the 40-layer Pallas megakernel across `token_ids_np` [T] and return `(logits_np [T, 129280], state)`."""
        state = self.init_decode_state(max_comp=max_comp, enable_engram=enable_engram)
        logits_list: List[np.ndarray] = []
        for tid in token_ids_np:
            l_step, _, state = self.decode_step(
                int(tid), state, enable_engram=enable_engram, return_dspark=False
            )
            logits_list.append(l_step[0])
        return np.stack(logits_list, axis=0), state

    def generate_greedy(
        self,
        prompt_ids_np: np.ndarray,
        max_new_tokens: int = 256,
        max_comp: int = 512,
        enable_engram: bool = True,
        spec_k: int = 0,
    ) -> Tuple[List[int], np.ndarray, PallasMegakernelDecodeState]:
        """Prefill `prompt_ids_np` and generate `max_new_tokens` greedy tokens through the 40-layer Pallas megakernel."""
        if spec_k <= 0:
            prompt_logits, state = self.prefill_and_logits(
                prompt_ids_np, max_comp=max_comp, enable_engram=enable_engram
            )
            generated: List[int] = [int(np.argmax(prompt_logits[-1]))]
            for _ in range(max_new_tokens - 1):
                nxt, state = self.decode_step_top1(
                    generated[-1], state, enable_engram=enable_engram
                )
                generated.append(nxt)
            return generated, prompt_logits, state

        state = self.init_decode_state(max_comp=max_comp, enable_engram=enable_engram)
        first_gen = 0
        for tid in prompt_ids_np:
            emitted, _, _, state = self.verify_speculative_step(
                [int(tid)], state, enable_engram=enable_engram, spec_k=spec_k
            )
            first_gen = emitted[0]
        generated = [first_gen]
        for _ in range(max_new_tokens - 1):
            emitted, _, _, state = self.verify_speculative_step(
                [generated[-1]], state, enable_engram=enable_engram, spec_k=spec_k
            )
            generated.append(emitted[0])
        return generated, np.zeros((0, 129280), dtype=np.float32), state

    def generate_speculative_greedy(
        self,
        prompt_ids_np: np.ndarray,
        dspark_engine: Any,
        max_new_tokens: int = 256,
        max_comp: int = 512,
        enable_engram: bool = True,
        spec_k: int = 5,
        num_draft: int = 4,
    ) -> Tuple[List[int], Dict[str, Any], PallasMegakernelDecodeState]:
        """Lossless speculative greedy decoding combining the 40-layer Pallas megakernel with the 3-layer `DSparkDraftEngine`.

        Uses the unified `(active_b=0, max_comp=max_comp, spec_k=spec_k)` Pallas megakernel and feeds
        target hidden states from layers `[37, 38, 39]` into `dspark_engine` (`mtp.0..2`) to propose
        `num_draft` draft tokens per step.
        """
        state = self.init_decode_state(max_comp=max_comp, enable_engram=enable_engram)
        dspark_kv, dspark_pos = dspark_engine.init_kv_state()

        t_len = len(prompt_ids_np)
        last_dspark_h: Optional[jax.Array] = None
        first_tok = 0
        for idx_p, tid in enumerate(prompt_ids_np):
            p_cur = len(state.token_history)
            emitted, _, dspark_h, state = self.verify_speculative_step(
                [int(tid)], state, enable_engram=enable_engram, spec_k=spec_k
            )
            first_tok = emitted[0]
            last_dspark_h = dspark_h
            # Seed the last 128 prompt positions into the DSpark SWA KV window (`[1, 15360]` bf16)
            if idx_p >= max(0, t_len - 128) and idx_p < t_len - 1:
                h_3x1x5120 = np.asarray(dspark_h.addressable_shards[0].data)[:, :1, :]
                h_1x15360 = np.concatenate(
                    [h_3x1x5120[0], h_3x1x5120[1], h_3x1x5120[2]], axis=-1
                )
                dspark_kv, dspark_pos = dspark_engine.update_kv_cache(
                    h_1x15360,
                    np.array([p_cur], dtype=np.int32),
                    dspark_kv,
                    dspark_pos,
                )

        # Update DSpark KV for the last prompt position (`t_len - 1`) and draft from `first_tok`
        assert last_dspark_h is not None
        prev_dspark_h = last_dspark_h
        prev_n_acc = 0
        prev_p0 = t_len - 1

        generated: List[int] = [first_tok]
        verify_steps = 0
        accepted_counts: List[int] = []
        draft_times_ms: List[float] = []
        verify_times_ms: List[float] = []

        t0_decode = time.perf_counter()
        while len(generated) < max_new_tokens:
            rem = max_new_tokens - len(generated)
            k_draft = min(num_draft, spec_k - 1, rem)
            root_tok = generated[-1]

            t0_d = time.perf_counter()
            draft_toks, dspark_kv, dspark_pos = dspark_engine.draft_block_on_device(
                root_token_id=root_tok,
                dspark_h=prev_dspark_h,
                prev_n_acc=prev_n_acc,
                prev_p0=prev_p0,
                swa_kv=dspark_kv,
                swa_pos=dspark_pos,
            )
            draft_times_ms.append((time.perf_counter() - t0_d) * 1000.0)

            cand = [int(root_tok)] + [int(x) for x in draft_toks[:k_draft]]
            p_step_0 = len(state.token_history)

            t0_v = time.perf_counter()
            emitted, n_acc, dspark_h, state = self.verify_speculative_step(
                cand, state, enable_engram=enable_engram, spec_k=spec_k
            )
            verify_times_ms.append((time.perf_counter() - t0_v) * 1000.0)

            verify_steps += 1
            accepted_counts.append(int(n_acc))

            prev_dspark_h = dspark_h
            prev_n_acc = int(n_acc)
            prev_p0 = p_step_0

            for tok in emitted:
                if len(generated) < max_new_tokens:
                    generated.append(int(tok))
        decode_elapsed_ms = (time.perf_counter() - t0_decode) * 1000.0

        stats = {
            "num_generated_tokens": len(generated),
            "num_verify_steps": int(verify_steps),
            "accepted_draft_counts": accepted_counts,
            "mean_accepted_draft_tokens": float(np.mean(accepted_counts))
            if accepted_counts
            else 0.0,
            "mean_tokens_per_verify_step": float(1.0 + np.mean(accepted_counts))
            if accepted_counts
            else 1.0,
            "median_draft_ms": float(np.median(draft_times_ms))
            if draft_times_ms
            else 0.0,
            "median_verify_ms": float(np.median(verify_times_ms))
            if verify_times_ms
            else 0.0,
            "decode_elapsed_ms": float(decode_elapsed_ms),
            "effective_tpot_ms": float(
                decode_elapsed_ms / max(1, len(generated) - 1)
            ),
        }
        return generated, stats, state

    def get_hlo_text(
        self, active_b: int = 1, max_comp: int = 512, spec_k: int = 0
    ) -> str:
        """Return the lowered HLO text for one 40-layer decode step to verify single `pallas_call`."""
        state = self.init_decode_state(max_comp=max_comp, enable_engram=False)
        b_tile = self.batch_tile
        res_in = self.replicate(np.zeros((4, b_tile, 5120), dtype=ml_dtypes.bfloat16))
        pos_in = self.replicate(np.zeros((b_tile, 128), dtype=np.int32))
        fn = self._get_step_kernel(
            active_b=active_b, max_comp=max_comp, spec_k=spec_k
        )
        lowered = fn.lower(
            res_in,
            self.pm_init,
            pos_in,
            self.inv_freq_plain,
            self.inv_freq_yarn,
            self.rope_sign,
            self.p_swap,
            self.s32,
            self.zero_eng_rows,
            state.swa_pos,
            state.swa_kv_all,
            state.comp_kv_all,
            state.ik_f8_all,
            state.ik_sc_all,
            state.comp_tail_all,
            *self._weight_args(),
        )
        return lowered.compiler_ir(dialect="hlo").as_hlo_text()



