"""Pure-JAX 16-chip reference engine for DeepSeek-V4.1-Flash on TPU v6e-16."""
from __future__ import annotations

import functools
import math
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

from .config import DSV41Config
from .load import (
    LayerWeights,
    ModelWeights,
    dequant_fp8_block32_jax,
    dequant_mxfp4_block32_jax,
    ue8m0_to_f32_scale,
)


def to_host_np(arr: jax.Array, dtype: Any = None) -> np.ndarray:
    """Read a replicated global `jax.Array` from local device 0 without multi-host blocking."""
    local_arr = np.asarray(arr.addressable_shards[0].data)
    return local_arr.astype(dtype) if dtype is not None else local_arr


def build_rope_caches(
    cfg: DSV41Config, max_seq_len: int = 8192
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build (cos_plain, sin_plain, cos_yarn, sin_yarn) tables of shape [max_seq_len, 32] in float32."""
    dim = cfg.qk_rope_head_dim  # 64
    pos = jnp.arange(max_seq_len, dtype=jnp.float32)

    # 1. Plain GPT-J RoPE (compress_ratio == 0: layers 0, 1, mtp.0..2)
    inv_freq_plain = 1.0 / (cfg.rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    freqs_plain = jnp.outer(pos, inv_freq_plain)
    cos_plain = jnp.cos(freqs_plain)
    sin_plain = jnp.sin(freqs_plain)

    # 2. DeepSeek YaRN GPT-J RoPE (compress_ratio > 0: layers 2..39, base = 160000.0)
    base = cfg.compress_rope_theta
    factor = cfg.rope_factor
    orig_max = cfg.rope_original_max_pos
    beta_fast = cfg.rope_beta_fast
    beta_slow = cfg.rope_beta_slow

    pos_freqs = base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    inv_extrap = 1.0 / pos_freqs
    inv_interp = 1.0 / (factor * pos_freqs)

    low = max(
        math.floor(dim * math.log(orig_max / (beta_fast * 2.0 * math.pi)) / (2.0 * math.log(base))),
        0,
    )
    high = min(
        math.ceil(dim * math.log(orig_max / (beta_slow * 2.0 * math.pi)) / (2.0 * math.log(base))),
        dim // 2 - 1,
    )
    ramp = jnp.clip(
        (jnp.arange(dim // 2, dtype=jnp.float32) - low) / max(high - low, 0.001),
        0.0,
        1.0,
    )
    extrap_factor = 1.0 - ramp
    inv_freq_yarn = inv_interp * (1.0 - extrap_factor) + inv_extrap * extrap_factor
    freqs_yarn = jnp.outer(pos, inv_freq_yarn)
    cos_yarn = jnp.cos(freqs_yarn)
    sin_yarn = jnp.sin(freqs_yarn)

    return cos_plain, sin_plain, cos_yarn, sin_yarn


def apply_gptj_rope(
    x: jax.Array,
    positions: jax.Array,
    cos_table: jax.Array,
    sin_table: jax.Array,
    inverse: bool = False,
) -> jax.Array:
    """Apply GPT-J interleaved RoPE on the trailing 64 channels of `x` ([T, D] or [T, H, D])."""
    x_nope = x[..., :-64]
    x_pe = x[..., -64:].astype(jnp.float32)
    orig_pe_shape = x_pe.shape
    x_pairs = x_pe.reshape(*orig_pe_shape[:-1], 32, 2)
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]

    c = cos_table[positions]  # [T, 32]
    s = sin_table[positions]  # [T, 32]
    if inverse:
        s = -s
    if x.ndim == 3:
        c = c[:, None, :]
        s = s[:, None, :]

    o0 = x0 * c - x1 * s
    o1 = x1 * c + x0 * s
    out_pe = jnp.stack([o0, o1], axis=-1).reshape(orig_pe_shape).astype(x.dtype)
    return jnp.concatenate([x_nope, out_pe], axis=-1)


def rms_norm_f32(x: jax.Array, weight: jax.Array, eps: float = 1e-20) -> jax.Array:
    """RMSNorm accumulated in float32 and cast back to bfloat16."""
    xf = x.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * inv_rms * weight.astype(jnp.float32)).astype(jnp.bfloat16)


def quantize_e8m0_roundtrip_bf16(x: jax.Array, block_size: int) -> jax.Array:
    """Exact FP8 E4M3 + UE8M0 block scale quantization/dequantization matching tpu_inference compressor."""
    orig_shape = x.shape
    x_blk = x.astype(jnp.float32).reshape(-1, block_size)
    amax = jnp.maximum(jnp.max(jnp.abs(x_blk), axis=-1, keepdims=True), 1e-4)
    exp = jnp.ceil(jnp.log2(amax / 448.0))
    q = jnp.clip(x_blk * jnp.exp2(-exp), -448.0, 448.0).astype(jnp.float8_e4m3fn)
    scale_u8 = jnp.clip(exp + 127.0, 0.0, 255.0).astype(jnp.uint8)
    s_bf16 = ue8m0_to_f32_scale(scale_u8).astype(jnp.bfloat16)
    return (q.astype(jnp.bfloat16) * s_bf16).reshape(orig_shape).astype(jnp.bfloat16)


def quantize_e8m0_pair(x: jax.Array, block_size: int) -> Tuple[jax.Array, jax.Array]:
    """Quantize `x` [..., D] to (q_f8 [..., D], scale_f32 [..., D // block_size])."""
    orig_shape = x.shape
    x_blk = x.astype(jnp.float32).reshape(-1, block_size)
    amax = jnp.maximum(jnp.max(jnp.abs(x_blk), axis=-1, keepdims=True), 1e-4)
    exp = jnp.ceil(jnp.log2(amax / 448.0))
    q = jnp.clip(x_blk * jnp.exp2(-exp), -448.0, 448.0).astype(jnp.float8_e4m3fn)
    scale_u8 = jnp.clip(exp + 127.0, 0.0, 255.0).astype(jnp.uint8)
    s_f32 = ue8m0_to_f32_scale(scale_u8)
    return (
        q.reshape(orig_shape),
        s_f32.reshape(*orig_shape[:-1], orig_shape[-1] // block_size),
    )


def mhc_pre_delayed_jax(
    residual: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    prev_pre_mix: jax.Array,
    rms_norm_eps: float = 1e-20,
    hc_eps: float = 1e-6,
    hc_post_alpha: float = 2.0,
    sinkhorn_iters: int = 20,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Exact JAX implementation of mhc_pre_delayed_torch from mhc_torch.py."""
    t_len = residual.shape[0]
    r_f32 = residual.astype(jnp.float32)  # [T, 4, 5120]
    x_flat = r_f32.reshape(t_len, -1)     # [T, 20480]
    inv_rms = jax.lax.rsqrt(jnp.mean(x_flat * x_flat, axis=-1, keepdims=True) + rms_norm_eps)
    mixes = jnp.dot(x_flat, hc_fn.T, precision=jax.lax.Precision.HIGHEST) * inv_rms

    pre_logits = mixes[:, :4] * hc_scale[0] + hc_base[:4]
    post_logits = mixes[:, 4:8] * hc_scale[1] + hc_base[4:8]
    res_logits = (mixes[:, 8:] * hc_scale[2] + hc_base[8:]).reshape(t_len, 4, 4)

    next_pre_mix = jax.nn.sigmoid(pre_logits) + hc_eps
    post_mix = hc_post_alpha * jax.nn.sigmoid(post_logits)

    comb0 = jax.nn.softmax(res_logits, axis=-1) + hc_eps
    comb0 = comb0 / (jnp.sum(comb0, axis=-2, keepdims=True) + hc_eps)

    def _sinkhorn_step(_: int, k_val: jax.Array) -> jax.Array:
        k_val = k_val / (jnp.sum(k_val, axis=-1, keepdims=True) + hc_eps)
        k_val = k_val / (jnp.sum(k_val, axis=-2, keepdims=True) + hc_eps)
        return k_val

    res_mix = jax.lax.fori_loop(0, sinkhorn_iters - 1, _sinkhorn_step, comb0)
    layer_in = jnp.sum(prev_pre_mix[:, :, None] * r_f32, axis=1).astype(jnp.bfloat16)

    return post_mix, res_mix, layer_in, next_pre_mix


def mhc_post_jax(
    layer_out: jax.Array,
    residual: jax.Array,
    post_mix: jax.Array,
    res_mix: jax.Array,
) -> jax.Array:
    """Exact JAX implementation of mhc_post_torch from mhc_torch.py."""
    r_f32 = residual.astype(jnp.float32)
    mixed = jnp.einsum("tij,tid->tjd", res_mix, r_f32, precision=jax.lax.Precision.HIGHEST)
    out = mixed + post_mix[:, :, None] * layer_out.astype(jnp.float32)[:, None, :]
    return out.astype(jnp.bfloat16)


def _psum_tp(x: jax.Array, use_bf16_mxu: bool = False) -> jax.Array:
    """All-reduce across 'tp': 1D XLA psum when use_bf16_mxu=False, or 2-stage 4x4 BF16 tree when use_bf16_mxu=True."""
    if not use_bf16_mxu:
        return jax.lax.psum(x, "tp")
    g = jax.lax.all_gather(x.astype(jnp.bfloat16), "tp", axis=0, tiled=False)
    g4 = g.reshape(4, 4, *x.shape)
    h = (
        (g4[:, 0] + g4[:, 1]).astype(jnp.bfloat16)
        + (g4[:, 2] + g4[:, 3]).astype(jnp.bfloat16)
    ).astype(jnp.bfloat16)
    return (
        (h[0] + h[1]).astype(jnp.bfloat16) + (h[2] + h[3]).astype(jnp.bfloat16)
    ).astype(x.dtype)


def fp8_linear_tpu_inference_jax(
    x: jax.Array,
    w_f8: jax.Array,
    w_scale_u8: jax.Array,
    sharded_k: bool = False,
    use_bf16_mxu: bool = False,
) -> jax.Array:
    """Exact block-32x32 FP8 dequantization + float32 or BF16 MXU dot product."""
    w_bf16 = dequant_fp8_block32_jax(w_f8, w_scale_u8)  # [out, in] bfloat16
    if use_bf16_mxu:
        out = jnp.dot(
            x.astype(jnp.bfloat16),
            w_bf16.T,
            preferred_element_type=jnp.bfloat16,
            precision=jax.lax.Precision.DEFAULT,
        ).astype(x.dtype)
    else:
        out = jnp.dot(
            x.astype(jnp.float32),
            w_bf16.astype(jnp.float32).T,
            precision=jax.lax.Precision.HIGHEST,
        ).astype(x.dtype)
    if sharded_k:
        out = _psum_tp(out, use_bf16_mxu=use_bf16_mxu)
    return out


def wo_a_tpu_inference_jax(
    o_flat: jax.Array,
    w_f8: jax.Array,
    w_scale_u8: jax.Array,
    use_bf16_mxu: bool = False,
) -> jax.Array:
    """Exact block-32x32 FP8 dequantization + float32 or BF16 MXU dot product for wo_a."""
    w_bf16 = dequant_fp8_block32_jax(w_f8, w_scale_u8)  # [512, 4096]
    if use_bf16_mxu:
        return jnp.dot(
            o_flat.astype(jnp.bfloat16),
            w_bf16.T,
            preferred_element_type=jnp.bfloat16,
            precision=jax.lax.Precision.DEFAULT,
        ).astype(o_flat.dtype)
    return jnp.dot(
        o_flat.astype(jnp.float32),
        w_bf16.astype(jnp.float32).T,
        precision=jax.lax.Precision.HIGHEST,
    ).astype(o_flat.dtype)


def engram_inject_jax(
    residual: jax.Array,
    local_engram_rows: jax.Array,
    engram_wkv: jax.Array,
    engram_wkv_scale: jax.Array,
    engram_q_weight: jax.Array,
    engram_k_weight: jax.Array,
    eps: float = 1e-20,
) -> jax.Array:
    """All-gather 4 hosts' 6-column Engram lookups and apply gated cross-attention onto `residual`."""
    t_len = residual.shape[0]
    gathered = jax.lax.all_gather(local_engram_rows, "tp", axis=0, tiled=True)  # [16, T, 6, 256]
    host_slices = gathered[0::4]  # [4, T, 6, 256]
    embed_flat = jnp.transpose(host_slices, (1, 0, 2, 3)).reshape(t_len, 6144)  # [T, 6144]

    kv = fp8_linear_tpu_inference_jax(
        embed_flat.astype(jnp.bfloat16), engram_wkv, engram_wkv_scale, sharded_k=False
    ).astype(jnp.float32)  # [T, 25600]

    key = kv[:, :20480].reshape(t_len, 4, 5120)
    value = kv[:, 20480:25600]  # [T, 5120]

    hidden = residual.astype(jnp.float32)  # [T, 4, 5120]
    hidden_rms = jax.lax.rsqrt(jnp.mean(hidden * hidden, axis=-1) + eps)
    key_rms = jax.lax.rsqrt(jnp.mean(key * key, axis=-1) + eps)

    qf = engram_q_weight.astype(jnp.float32)[None, :, :]
    kf = engram_k_weight.astype(jnp.float32)[None, :, :]
    dot = jnp.sum(hidden * qf * kf * key, axis=-1)
    dot = dot * hidden_rms * key_rms * (5120.0 ** -0.5)

    gate = jnp.sqrt(jnp.maximum(jnp.abs(dot), 1e-6))
    gate = jax.nn.sigmoid(jnp.where(dot < 0.0, -gate, gate))  # [T, 4]
    return (hidden + gate[:, :, None] * value[:, None, :]).astype(jnp.bfloat16)


def moe_sublayer_local_jax(
    x_norm: jax.Array,
    lw: LayerWeights,
    cfg: DSV41Config,
    actual_len: Optional[jax.Array] = None,
    use_bf16_mxu: bool = False,
) -> jax.Array:
    """Compute TP=16 sharded Shared Expert + Top-6 Routed Experts and all-reduce across 'tp'."""
    if use_bf16_mxu:
        logits = jnp.dot(
            x_norm.astype(jnp.bfloat16),
            lw.gate_weight.astype(jnp.bfloat16).T,
            preferred_element_type=jnp.bfloat16,
            precision=jax.lax.Precision.DEFAULT,
        ).astype(jnp.float32)
    else:
        logits = jnp.dot(
            x_norm.astype(jnp.float32),
            lw.gate_weight.astype(jnp.float32).T,
            precision=jax.lax.Precision.HIGHEST,
        )  # [T, 384]
    scores = jnp.sqrt(jax.nn.softplus(logits))
    biased = scores + lw.gate_bias[None, :]
    _, topk_idx = jax.lax.top_k(biased, cfg.num_experts_per_tok)  # [T, 6]
    topk_w = jnp.take_along_axis(scores, topk_idx, axis=-1)       # [T, 6]
    topk_w = (
        (topk_w / (jnp.sum(topk_w, axis=-1, keepdims=True) + 1e-20)) * cfg.routed_scaling_factor
    ).astype(jnp.bfloat16)

    g_sh = fp8_linear_tpu_inference_jax(
        x_norm, lw.shared_w1, lw.shared_w1_scale, sharded_k=False, use_bf16_mxu=use_bf16_mxu
    ).astype(jnp.float32)
    u_sh = fp8_linear_tpu_inference_jax(
        x_norm, lw.shared_w3, lw.shared_w3_scale, sharded_k=False, use_bf16_mxu=use_bf16_mxu
    ).astype(jnp.float32)
    g_sh = jnp.minimum(g_sh, cfg.swiglu_limit)
    u_sh = jnp.clip(u_sh, -cfg.swiglu_limit, cfg.swiglu_limit)
    h_sh = (jax.nn.silu(g_sh) * u_sh).astype(jnp.bfloat16)
    shared_out = fp8_linear_tpu_inference_jax(
        h_sh, lw.shared_w2, lw.shared_w2_scale, sharded_k=True, use_bf16_mxu=use_bf16_mxu
    )  # [T, 5120] bfloat16

    n_tok = x_norm.shape[0] if actual_len is None else actual_len[0]
    top_k = cfg.num_experts_per_tok
    routed_local = jnp.zeros((x_norm.shape[0], cfg.hidden_size), dtype=jnp.float32)

    def _tok_expert_step(step_idx: int, acc: jax.Array) -> jax.Array:
        t = step_idx // top_k
        k = step_idx % top_k
        e = topk_idx[t, k]
        w_tk = topk_w[t, k].astype(jnp.float32)
        if use_bf16_mxu:
            x_t = x_norm[t].astype(jnp.bfloat16)
            w1_bf16 = dequant_mxfp4_block32_jax(lw.routed_w1[e], lw.routed_w1_scale[e])
            w3_bf16 = dequant_mxfp4_block32_jax(lw.routed_w3[e], lw.routed_w3_scale[e])
            w2_bf16 = dequant_mxfp4_block32_jax(lw.routed_w2[e], lw.routed_w2_scale[e])
            g_rt = jnp.dot(w1_bf16, x_t, preferred_element_type=jnp.bfloat16, precision=jax.lax.Precision.DEFAULT).astype(jnp.float32)
            u_rt = jnp.dot(w3_bf16, x_t, preferred_element_type=jnp.bfloat16, precision=jax.lax.Precision.DEFAULT).astype(jnp.float32)
            g_rt = jnp.minimum(g_rt, cfg.swiglu_limit)
            u_rt = jnp.clip(u_rt, -cfg.swiglu_limit, cfg.swiglu_limit)
            h_rt = (jax.nn.silu(g_rt) * u_rt).astype(jnp.bfloat16)
            y_rt = jnp.dot(w2_bf16, h_rt, preferred_element_type=jnp.bfloat16, precision=jax.lax.Precision.DEFAULT).astype(jnp.bfloat16)
        else:
            x_t = x_norm[t].astype(jnp.float32)  # [5120]
            w1_bf16 = dequant_mxfp4_block32_jax(lw.routed_w1[e], lw.routed_w1_scale[e]).astype(jnp.float32)  # [160, 5120]
            w3_bf16 = dequant_mxfp4_block32_jax(lw.routed_w3[e], lw.routed_w3_scale[e]).astype(jnp.float32)  # [160, 5120]
            w2_bf16 = dequant_mxfp4_block32_jax(lw.routed_w2[e], lw.routed_w2_scale[e]).astype(jnp.float32)  # [5120, 160]
            g_rt = jnp.dot(w1_bf16, x_t, precision=jax.lax.Precision.HIGHEST)
            u_rt = jnp.dot(w3_bf16, x_t, precision=jax.lax.Precision.HIGHEST)
            g_rt = jnp.minimum(g_rt, cfg.swiglu_limit)
            u_rt = jnp.clip(u_rt, -cfg.swiglu_limit, cfg.swiglu_limit)
            h_rt = (jax.nn.silu(g_rt) * u_rt).astype(jnp.bfloat16).astype(jnp.float32)
            y_rt = jnp.dot(w2_bf16, h_rt, precision=jax.lax.Precision.HIGHEST).astype(jnp.bfloat16)
        return acc.at[t].add(y_rt.astype(jnp.float32) * w_tk)

    routed_local = jax.lax.fori_loop(0, n_tok * top_k, _tok_expert_step, routed_local)
    routed_total = _psum_tp(routed_local.astype(jnp.bfloat16), use_bf16_mxu=use_bf16_mxu)
    return (shared_out + routed_total).astype(jnp.bfloat16)


def compressor_prefill_jax(
    x_norm: jax.Array,
    lw: LayerWeights,
    cos_yarn: jax.Array,
    sin_yarn: jax.Array,
    actual_len: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run Compressor over padded prefill sequence `x_norm` [T, 5120] with dynamic `actual_len` [1]."""
    t_len = x_norm.shape[0]
    ratio = lw.compress_ratio
    act_t = actual_len[0]
    if ratio == 2:
        w_fused = jnp.concatenate(
            [lw.comp_wkv.astype(jnp.float32), lw.comp_wgate.astype(jnp.float32)], axis=0
        )  # [1024, 5120]
        kv_score = jnp.dot(
            x_norm.astype(jnp.float32), w_fused.T, precision=jax.lax.Precision.HIGHEST
        )  # [T, 1024]
        n_comp = t_len // 2
        tail_idx = jnp.clip(act_t - 1, 0, t_len - 1)
        odd_tail = jax.lax.dynamic_slice_in_dim(kv_score, tail_idx, 1, axis=0)
        tail_score = jnp.where(
            (act_t % 2) == 1,
            odd_tail,
            jnp.zeros((1, 1024), dtype=jnp.float32),
        )
        pairs = kv_score[: n_comp * 2].reshape(n_comp, 2, 1024)
        vals = pairs[:, :, :512]
        gates = jax.nn.softmax(pairs[:, :, 512:], axis=1)
        pooled = jnp.sum(vals * gates, axis=1)  # [N_comp, 512]
        latent_raw = rms_norm_f32(pooled, lw.comp_norm, 1e-20)
        comp_pos = jnp.arange(n_comp, dtype=jnp.int32) * 2
        n_valid_c = jnp.array([act_t // 2], dtype=jnp.int32)
    else:
        kv_score = jnp.dot(
            x_norm.astype(jnp.float32),
            lw.comp_wkv.astype(jnp.float32).T,
            precision=jax.lax.Precision.HIGHEST,
        )  # [T, 512]
        latent_raw = rms_norm_f32(kv_score, lw.comp_norm, 1e-20)
        tail_score = jnp.zeros((1, 1024), dtype=jnp.float32)
        comp_pos = jnp.arange(t_len, dtype=jnp.int32)
        n_valid_c = jnp.array([act_t], dtype=jnp.int32)

    rotated = apply_gptj_rope(latent_raw, comp_pos, cos_yarn, sin_yarn, inverse=False)
    nope_q = quantize_e8m0_roundtrip_bf16(rotated[:, :448], 64)
    comp_kv = jnp.concatenate([nope_q, rotated[:, 448:]], axis=-1)
    return latent_raw, comp_kv, tail_score, n_valid_c


def indexer_build_k_jax(
    latent_raw: jax.Array,
    lw: LayerWeights,
    comp_pos: jax.Array,
    cos_yarn: jax.Array,
    sin_yarn: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Project `latent_raw` [N_comp, 512] through indexer `wk` + `k_norm` + RoPE + E8M0 block-128 quant."""
    k_proj = jnp.dot(
        latent_raw,
        lw.idx_wk.T,
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    k_normed = rms_norm_f32(k_proj, lw.idx_k_norm, 1e-20)
    k_rot = apply_gptj_rope(k_normed, comp_pos, cos_yarn, sin_yarn, inverse=False)
    return quantize_e8m0_pair(k_rot, 128)


def indexer_select_topk_jax(
    qr: jax.Array,
    x_norm: jax.Array,
    lw: LayerWeights,
    positions: jax.Array,
    idx_k_f8: jax.Array,
    idx_k_scale: jax.Array,
    num_valid_comp: jax.Array,
    cos_yarn: jax.Array,
    sin_yarn: jax.Array,
    cfg: DSV41Config,
) -> jax.Array:
    """Compute top-512 compressed KV indices per query token [T, 512] int32 (-1 for invalid)."""
    t_len = qr.shape[0]
    ratio = lw.compress_ratio
    avail = (positions + 1) // ratio  # [T]
    max_comp = idx_k_f8.shape[0]
    comp_idx = jnp.arange(max_comp, dtype=jnp.int32)
    n_val = num_valid_comp[0]

    offsets = jnp.arange(cfg.index_topk, dtype=jnp.int32)[None, :]
    short_topk = jnp.where(offsets < avail[:, None], offsets, -1)

    q_idx = fp8_linear_tpu_inference_jax(
        qr, lw.idx_wq_b, lw.idx_wq_b_scale, sharded_k=False
    )
    q_idx = q_idx.reshape(t_len, cfg.index_n_heads, cfg.index_head_dim)
    q_idx = apply_gptj_rope(q_idx, positions, cos_yarn, sin_yarn, inverse=False)
    q_f8, q_scale = quantize_e8m0_pair(q_idx, 128)

    weights = jnp.dot(
        x_norm,
        lw.idx_weights_proj.T,
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16).astype(jnp.float32) * (cfg.index_n_heads ** -0.5)
    w_scaled = weights * q_scale[:, :, 0] * (cfg.index_head_dim ** -0.5)

    dots = jnp.einsum(
        "thd,cd->thc",
        q_f8.astype(jnp.float32),
        idx_k_f8.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    dots_scaled = jax.nn.relu(dots * idx_k_scale[None, None, :, 0])
    scores = jnp.einsum("thc,th->tc", dots_scaled, w_scaled, precision=jax.lax.Precision.HIGHEST)

    valid_mask = (comp_idx[None, :] < avail[:, None]) & (comp_idx[None, :] < n_val)
    scores_masked = jnp.where(valid_mask, scores, -jnp.inf)

    k_select = min(cfg.index_topk, max_comp)
    _, topk_i = jax.lax.top_k(scores_masked, k_select)
    topk_valid = topk_i < avail[:, None]
    topk_i = jnp.where(topk_valid, topk_i, -1)
    if k_select < cfg.index_topk:
        pad = jnp.full((t_len, cfg.index_topk - k_select), -1, dtype=jnp.int32)
        topk_i = jnp.concatenate([topk_i, pad], axis=-1)

    large_val = jnp.int32(100000000)
    sorted_topk = jnp.sort(jnp.where(topk_i < 0, large_val, topk_i), axis=-1)
    sorted_topk = jnp.where(sorted_topk == large_val, -1, sorted_topk)
    return jnp.where((avail <= cfg.index_topk)[:, None], short_topk, sorted_topk)


def attention_joint_swa_csa_jax(
    q: jax.Array,
    swa_kv: jax.Array,
    swa_positions: jax.Array,
    comp_kv: jax.Array,
    topk_indices: jax.Array,
    query_positions: jax.Array,
    attn_sink: jax.Array,
    compress_ratio: int,
    sliding_window: int = 128,
) -> jax.Array:
    """Exact joint SWA + CSA + Sink attention for 8 heads of group `g = c // 2` on one chip."""
    sm_scale = 512.0 ** -0.5
    qf = q.astype(jnp.float32)
    swa_kv_f = swa_kv.astype(jnp.float32)

    swa_logits = (
        jnp.einsum("thd,sd->ths", qf, swa_kv_f, precision=jax.lax.Precision.HIGHEST) * sm_scale
    )
    q_pos = query_positions[:, None]
    kv_pos = swa_positions[None, :]
    swa_mask = (kv_pos >= 0) & (kv_pos <= q_pos) & (kv_pos >= jnp.maximum(0, q_pos - (sliding_window - 1)))
    swa_logits = jnp.where(swa_mask[:, None, :], swa_logits, -jnp.inf)

    if compress_ratio == 0:
        m = jnp.maximum(jnp.max(swa_logits, axis=-1), attn_sink[None, :])
        w_swa = jnp.exp(swa_logits - m[:, :, None])
        denom = jnp.sum(w_swa, axis=-1) + jnp.exp(attn_sink[None, :] - m)
        num = jnp.einsum("ths,sd->thd", w_swa, swa_kv_f, precision=jax.lax.Precision.HIGHEST)
        return (num / denom[:, :, None]).astype(jnp.bfloat16)

    safe_idx = jnp.maximum(topk_indices, 0)
    c_kv_gathered = comp_kv[safe_idx].astype(jnp.float32)
    csa_logits = (
        jnp.einsum("thd,tkd->thk", qf, c_kv_gathered, precision=jax.lax.Precision.HIGHEST) * sm_scale
    )
    csa_valid = topk_indices >= 0
    csa_logits = jnp.where(csa_valid[:, None, :], csa_logits, -jnp.inf)

    m_swa = jnp.max(swa_logits, axis=-1)
    m_csa = jnp.max(csa_logits, axis=-1)
    m = jnp.maximum(jnp.maximum(m_swa, m_csa), attn_sink[None, :])

    w_swa = jnp.exp(swa_logits - m[:, :, None])
    w_csa = jnp.exp(csa_logits - m[:, :, None])
    denom = jnp.sum(w_swa, axis=-1) + jnp.sum(w_csa, axis=-1) + jnp.exp(attn_sink[None, :] - m)

    num_swa = jnp.einsum("ths,sd->thd", w_swa, swa_kv_f, precision=jax.lax.Precision.HIGHEST)
    num_csa = jnp.einsum("thk,tkd->thd", w_csa, c_kv_gathered, precision=jax.lax.Precision.HIGHEST)
    return ((num_swa + num_csa) / denom[:, :, None]).astype(jnp.bfloat16)


def o_proj_local_jax(
    o: jax.Array,
    positions: jax.Array,
    cos_table: jax.Array,
    sin_table: jax.Array,
    lw: LayerWeights,
    use_bf16_mxu: bool = False,
) -> jax.Array:
    """Apply inverse RoPE to `o` [T, 8, 512], project through local `wo_a` [512, 4096] & `wo_b` [5120, 512], and psum."""
    t_len = o.shape[0]
    o_unrot = apply_gptj_rope(o, positions, cos_table, sin_table, inverse=True)
    o_flat = o_unrot.reshape(t_len, 4096)

    z_local = wo_a_tpu_inference_jax(o_flat, lw.wo_a, lw.wo_a_scale, use_bf16_mxu=use_bf16_mxu)
    return fp8_linear_tpu_inference_jax(z_local, lw.wo_b, lw.wo_b_scale, sharded_k=True, use_bf16_mxu=use_bf16_mxu)


class DSV41JaxEngine:
    """Pure-JAX 16-chip reference engine for DeepSeek-V4.1-Flash (prefill + decode)."""

    def __init__(self, weights: ModelWeights, max_seq_len: int = 2048, use_bf16_mxu: bool = False) -> None:
        self.weights = weights
        self.use_bf16_mxu = use_bf16_mxu
        self.cfg = weights.config
        self.mesh = weights.mesh
        self.max_seq_len = max_seq_len

        cos_p, sin_p, cos_y, sin_y = build_rope_caches(self.cfg, max_seq_len=max_seq_len)
        local_devs = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))[
            jax.process_index() * 4 : (jax.process_index() + 1) * 4
        ]
        rep_sharding = NamedSharding(self.mesh, P())
        self.cos_plain = jax.make_array_from_single_device_arrays(
            cos_p.shape, rep_sharding, [jax.device_put(np.asarray(cos_p), d) for d in local_devs]
        )
        self.sin_plain = jax.make_array_from_single_device_arrays(
            sin_p.shape, rep_sharding, [jax.device_put(np.asarray(sin_p), d) for d in local_devs]
        )
        self.cos_yarn = jax.make_array_from_single_device_arrays(
            cos_y.shape, rep_sharding, [jax.device_put(np.asarray(cos_y), d) for d in local_devs]
        )
        self.sin_yarn = jax.make_array_from_single_device_arrays(
            sin_y.shape, rep_sharding, [jax.device_put(np.asarray(sin_y), d) for d in local_devs]
        )
        self._local_devs = local_devs
        self._rep_sharding = rep_sharding
        self._host_sharding = NamedSharding(self.mesh, P("tp", None, None, None))
        self._dummy_f32 = self.replicate_array(np.zeros((1,), dtype=np.float32))
        self._dummy_i32 = self.replicate_array(np.zeros((1,), dtype=np.int32))
        self._dummy_tail = self.replicate_array(np.zeros((1, 1024), dtype=np.float32))
        self._init_pm_cache: Dict[int, jax.Array] = {}
        self._compile_kernels()

    def replicate_array(self, arr: np.ndarray) -> jax.Array:
        return jax.make_array_from_single_device_arrays(
            arr.shape, self._rep_sharding, [jax.device_put(arr, d) for d in self._local_devs]
        )

    def _get_init_pre_mix(self, t_len: int) -> jax.Array:
        if t_len not in self._init_pm_cache:
            pm = np.zeros((t_len, self.cfg.hc_mult), dtype=np.float32)
            pm[:, 0] = 1.0
            self._init_pm_cache[t_len] = self.replicate_array(pm)
        return self._init_pm_cache[t_len]

    def put_host_engram_rows(self, local_rows: np.ndarray) -> jax.Array:
        t_len, cols, dim = local_rows.shape
        per_chip = [jax.device_put(local_rows[None, ...], d) for d in self._local_devs]
        return jax.make_array_from_single_device_arrays(
            (16, t_len, cols, dim), self._host_sharding, per_chip
        )

    def _compile_kernels(self) -> None:
        mesh = self.mesh
        cfg = self.cfg
        rep = NamedSharding(mesh, P())

        @functools.partial(
            jax.jit,
            in_shardings=(NamedSharding(mesh, P(None, "tp")), rep),
            out_shardings=rep,
        )
        def _embed_fn(embed_w: jax.Array, token_ids: jax.Array) -> jax.Array:
            def _local(w_part, tids):
                part = w_part[tids]
                emb = jax.lax.all_gather(part, "tp", axis=-1, tiled=True)
                return jnp.broadcast_to(
                    emb[:, None, :], (tids.shape[0], cfg.hc_mult, cfg.hidden_size)
                )

            return jax.shard_map(
                _local,
                mesh=mesh,
                in_specs=(P(None, "tp"), P()),
                out_specs=P(),
                check_vma=False,
            )(embed_w, token_ids)

        self._embed_fn = _embed_fn

        @functools.partial(
            jax.jit,
            in_shardings=(
                rep,
                NamedSharding(mesh, P("tp", None, None, None)),
                rep,
                rep,
                rep,
                rep,
            ),
            out_shardings=rep,
        )
        def _engram_fn(
            residual: jax.Array,
            engram_rows_in: jax.Array,
            engram_wkv: jax.Array,
            engram_wkv_scale: jax.Array,
            engram_q_weight: jax.Array,
            engram_k_weight: jax.Array,
        ) -> jax.Array:
            def _local(res, eng_rows, ewkv, ewkv_s, eqw, ekw):
                return engram_inject_jax(
                    res, eng_rows, ewkv, ewkv_s, eqw, ekw, cfg.rms_norm_eps
                )

            return jax.shard_map(
                _local,
                mesh=mesh,
                in_specs=(P(), P("tp", None, None, None), P(), P(), P(), P()),
                out_specs=P(),
                check_vma=False,
            )(
                residual,
                engram_rows_in,
                engram_wkv,
                engram_wkv_scale,
                engram_q_weight,
                engram_k_weight,
            )

        self._engram_fn = _engram_fn

        @functools.partial(
            jax.jit,
            in_shardings=(rep, rep, rep, NamedSharding(mesh, P("tp", None))),
            out_shardings=rep,
        )
        def _lm_head_fn(
            residual: jax.Array,
            pre_mix: jax.Array,
            norm_w: jax.Array,
            head_w: jax.Array,
        ) -> jax.Array:
            def _local(res, pm, nw, hw):
                h_coll = jnp.sum(pm[:, :, None] * res.astype(jnp.float32), axis=1).astype(jnp.bfloat16)
                h_norm = rms_norm_f32(h_coll, nw, cfg.rms_norm_eps)
                logits_part = jnp.dot(
                    h_norm.astype(jnp.float32),
                    hw.astype(jnp.float32).T,
                    precision=jax.lax.Precision.HIGHEST,
                )
                return jax.lax.all_gather(logits_part, "tp", axis=-1, tiled=True)

            return jax.shard_map(
                _local,
                mesh=mesh,
                in_specs=(P(), P(), P(), P("tp", None)),
                out_specs=P(),
                check_vma=False,
            )(residual, pre_mix, norm_w, head_w)

        self._lm_head_fn = _lm_head_fn

        @functools.partial(
            jax.jit,
            static_argnames=(
                "compress_ratio",
                "is_kv_source",
                "is_index_source",
                "is_decode",
            ),
            out_shardings=tuple(rep for _ in range(10)),
        )
        def _run_layer(
            compress_ratio: int,
            is_kv_source: bool,
            is_index_source: bool,
            is_decode: bool,
            residual: jax.Array,
            prev_pre_mix: jax.Array,
            positions: jax.Array,
            actual_len: jax.Array,
            cos_plain: jax.Array,
            sin_plain: jax.Array,
            cos_yarn: jax.Array,
            sin_yarn: jax.Array,
            swa_kv_in: jax.Array,
            swa_pos_in: jax.Array,
            comp_kv_in: jax.Array,
            idx_k_f8_in: jax.Array,
            idx_k_scale_in: jax.Array,
            num_valid_comp_in: jax.Array,
            topk_indices_in: jax.Array,
            comp_tail_in: jax.Array,
            attn_norm: jax.Array,
            ffn_norm: jax.Array,
            hc_attn_fn: jax.Array,
            hc_attn_scale: jax.Array,
            hc_attn_base: jax.Array,
            hc_ffn_fn: jax.Array,
            hc_ffn_scale: jax.Array,
            hc_ffn_base: jax.Array,
            wq_a: jax.Array,
            wq_a_scale: jax.Array,
            q_norm: jax.Array,
            wkv: jax.Array,
            wkv_scale: jax.Array,
            kv_norm: jax.Array,
            wq_b: jax.Array,
            wq_b_scale: jax.Array,
            attn_sink: jax.Array,
            wo_a: jax.Array,
            wo_a_scale: jax.Array,
            wo_b: jax.Array,
            wo_b_scale: jax.Array,
            gate_weight: jax.Array,
            gate_bias: jax.Array,
            shared_w1: jax.Array,
            shared_w1_scale: jax.Array,
            shared_w3: jax.Array,
            shared_w3_scale: jax.Array,
            shared_w2: jax.Array,
            shared_w2_scale: jax.Array,
            routed_w1: jax.Array,
            routed_w1_scale: jax.Array,
            routed_w3: jax.Array,
            routed_w3_scale: jax.Array,
            routed_w2: jax.Array,
            routed_w2_scale: jax.Array,
            comp_wkv: jax.Array,
            comp_wgate: jax.Array,
            comp_norm: jax.Array,
            idx_wq_b: jax.Array,
            idx_wq_b_scale: jax.Array,
            idx_weights_proj: jax.Array,
            idx_wk: jax.Array,
            idx_k_norm: jax.Array,
        ):
            in_specs = (
                P(), P(), P(), P(),
                P(), P(), P(), P(),
                P(), P(), P(), P(), P(), P(), P(), P(),
                P(), P(), P(), P(), P(), P(), P(), P(),
                P(), P(), P(), P(), P(), P(),
                P("tp", None), P("tp", None), P("tp"),
                P("tp", None), P("tp", None),
                P(None, "tp"), P(None, "tp"),
                P(), P(),
                P("tp", None), P("tp", None),
                P("tp", None), P("tp", None),
                P(None, "tp"), P(None, "tp"),
                P(None, "tp", None), P(None, "tp", None),
                P(None, "tp", None), P(None, "tp", None),
                P(None, None, "tp"), P(None, None, "tp"),
                P(), P(), P(),
                P(), P(), P(), P(), P(),
            )
            out_specs = (P(), P(), P(), P(), P(), P(), P(), P(), P(), P())

            def _body(
                res, prev_pm, pos, act_len, cp, sp, cy, sy,
                swa_kv, swa_pos, comp_kv, ik_f8, ik_sc, n_valid_c, topk_i, c_tail,
                a_norm, f_norm, hca_fn, hca_sc, hca_b, hcf_fn, hcf_sc, hcf_b,
                wqa, wqa_s, qn, wkv_w, wkv_s, kvn,
                wqb, wqb_s, a_sink, woa, woa_s, wob, wob_s,
                gw, gb, sw1, sw1_s, sw3, sw3_s, sw2, sw2_s,
                rw1, rw1_s, rw3, rw3_s, rw2, rw2_s,
                cwkv, cwgate, cnorm, iwqb, iwqb_s, iwproj, iwk, iknorm,
            ):
                lw_local = LayerWeights(
                    layer_id=0,
                    compress_ratio=compress_ratio,
                    is_kv_source=is_kv_source,
                    is_index_source=is_index_source,
                    has_engram=False,
                    attn_norm=a_norm, ffn_norm=f_norm,
                    hc_attn_fn=hca_fn, hc_attn_scale=hca_sc, hc_attn_base=hca_b,
                    hc_ffn_fn=hcf_fn, hc_ffn_scale=hcf_sc, hc_ffn_base=hcf_b,
                    wq_a=wqa, wq_a_scale=wqa_s, q_norm=qn,
                    wkv=wkv_w, wkv_scale=wkv_s, kv_norm=kvn,
                    wq_b=wqb, wq_b_scale=wqb_s, attn_sink=a_sink,
                    wo_a=woa, wo_a_scale=woa_s, wo_b=wob, wo_b_scale=wob_s,
                    gate_weight=gw, gate_bias=gb,
                    shared_w1=sw1, shared_w1_scale=sw1_s,
                    shared_w3=sw3, shared_w3_scale=sw3_s,
                    shared_w2=sw2, shared_w2_scale=sw2_s,
                    routed_w1=rw1, routed_w1_scale=rw1_s,
                    routed_w3=rw3, routed_w3_scale=rw3_s,
                    routed_w2=rw2, routed_w2_scale=rw2_s,
                    comp_wkv=cwkv, comp_wgate=cwgate, comp_norm=cnorm,
                    idx_wq_b=iwqb, idx_wq_b_scale=iwqb_s, idx_weights_proj=iwproj,
                    idx_wk=iwk, idx_k_norm=iknorm,
                )

                post_mix, res_mix, x_in, attn_pre = mhc_pre_delayed_jax(
                    res, hca_fn, hca_sc, hca_b, prev_pm,
                    rms_norm_eps=cfg.rms_norm_eps,
                    hc_eps=cfg.hc_eps,
                    sinkhorn_iters=cfg.hc_sinkhorn_iters,
                )
                bf16_mxu = self.use_bf16_mxu
                x_norm = rms_norm_f32(x_in, a_norm, cfg.rms_norm_eps)

                qr = rms_norm_f32(
                    fp8_linear_tpu_inference_jax(x_norm, wqa, wqa_s, sharded_k=False, use_bf16_mxu=bf16_mxu),
                    qn,
                    cfg.rms_norm_eps,
                )
                kv = rms_norm_f32(
                    fp8_linear_tpu_inference_jax(x_norm, wkv_w, wkv_s, sharded_k=False, use_bf16_mxu=bf16_mxu),
                    kvn,
                    cfg.rms_norm_eps,
                )

                cos_tbl = cp if compress_ratio == 0 else cy
                sin_tbl = sp if compress_ratio == 0 else sy

                q = fp8_linear_tpu_inference_jax(qr, wqb, wqb_s, sharded_k=False, use_bf16_mxu=bf16_mxu)
                q = q.reshape(x_norm.shape[0], 8, 512)
                q = apply_gptj_rope(q, pos, cos_tbl, sin_tbl, inverse=False)
                kv_rot = apply_gptj_rope(kv, pos, cos_tbl, sin_tbl, inverse=False)

                if not is_decode:
                    slot_idx = jnp.arange(cfg.sliding_window, dtype=jnp.int32)
                    base_pos = jnp.maximum(jnp.int32(0), act_len[0] - cfg.sliding_window)
                    p_slot = base_pos + ((slot_idx - base_pos) % cfg.sliding_window)
                    valid_slot = p_slot < act_len[0]
                    safe_p = jnp.clip(p_slot, 0, kv_rot.shape[0] - 1)
                    swa_kv_out = jnp.where(valid_slot[:, None], kv_rot[safe_p], jnp.bfloat16(0.0))
                    swa_pos_out = jnp.where(valid_slot, p_slot, jnp.int32(-1))

                    if is_kv_source:
                        latent_raw, c_kv_new, c_tail_out, n_valid_c_out = compressor_prefill_jax(
                            x_norm, lw_local, cos_tbl, sin_tbl, act_len
                        )
                        n_comp = c_kv_new.shape[0]
                        c_pos = jnp.arange(n_comp, dtype=jnp.int32) * compress_ratio
                        ik_f8_new, ik_sc_new = indexer_build_k_jax(
                            latent_raw, lw_local, c_pos, cos_tbl, sin_tbl
                        )
                        comp_kv_out = comp_kv.at[:n_comp].set(c_kv_new)
                        ik_f8_out = ik_f8.at[:n_comp].set(ik_f8_new)
                        ik_sc_out = ik_sc.at[:n_comp].set(ik_sc_new)
                    else:
                        comp_kv_out = comp_kv
                        ik_f8_out = ik_f8
                        ik_sc_out = ik_sc
                        n_valid_c_out = n_valid_c
                        c_tail_out = c_tail

                    if is_index_source:
                        topk_out = indexer_select_topk_jax(
                            qr, x_norm, lw_local, pos,
                            ik_f8_out, ik_sc_out, n_valid_c_out,
                            cos_tbl, sin_tbl, cfg,
                        )
                    else:
                        topk_out = topk_i

                    o = attention_joint_swa_csa_jax(
                        q, kv_rot, pos,
                        comp_kv_out, topk_out,
                        pos, a_sink, compress_ratio, cfg.sliding_window,
                    )
                else:
                    p_cur = pos[0]
                    slot = p_cur % cfg.sliding_window
                    swa_kv_out = swa_kv.at[slot].set(kv_rot[0])
                    swa_pos_out = swa_pos.at[slot].set(p_cur)

                    if is_kv_source:
                        if compress_ratio == 2:
                            w_fused = jnp.concatenate(
                                [cwkv.astype(jnp.float32), cwgate.astype(jnp.float32)], axis=0
                            )
                            cur_score = jnp.dot(
                                x_norm[:1].astype(jnp.float32), w_fused.T, precision=jax.lax.Precision.HIGHEST
                            )
                            is_odd = (p_cur % 2) == 1
                            pair_scores = jnp.concatenate([c_tail[:1], cur_score[:1]], axis=0)
                            gates = jax.nn.softmax(pair_scores[:, 512:], axis=0)
                            pooled = jnp.sum(pair_scores[:, :512] * gates, axis=0, keepdims=True)
                            latent_raw = rms_norm_f32(pooled, cnorm, 1e-20)
                            c_slot = p_cur // 2
                            c_pos = jnp.array([c_slot * 2], dtype=jnp.int32)
                            rot_lat = apply_gptj_rope(latent_raw, c_pos, cos_tbl, sin_tbl, inverse=False)
                            nope_q = quantize_e8m0_roundtrip_bf16(rot_lat[:, :448], 64)
                            new_comp = jnp.concatenate([nope_q, rot_lat[:, 448:]], axis=-1)[0]
                            new_ik_f8, new_ik_sc = indexer_build_k_jax(latent_raw, lw_local, c_pos, cos_tbl, sin_tbl)

                            comp_kv_out = jnp.where(is_odd, comp_kv.at[c_slot].set(new_comp), comp_kv)
                            ik_f8_out = jnp.where(is_odd, ik_f8.at[c_slot].set(new_ik_f8[0]), ik_f8)
                            ik_sc_out = jnp.where(is_odd, ik_sc.at[c_slot].set(new_ik_sc[0]), ik_sc)
                            n_valid_c_out = jnp.array([(p_cur + 1) // 2], dtype=jnp.int32)
                            c_tail_out = jnp.where(is_odd, c_tail[:1], cur_score[:1])
                        else:
                            cur_score = jnp.dot(
                                x_norm[:1].astype(jnp.float32),
                                cwkv.astype(jnp.float32).T,
                                precision=jax.lax.Precision.HIGHEST,
                            )
                            latent_raw = rms_norm_f32(cur_score[:1], cnorm, 1e-20)
                            c_slot = p_cur
                            c_pos = jnp.array([p_cur], dtype=jnp.int32)
                            rot_lat = apply_gptj_rope(latent_raw, c_pos, cos_tbl, sin_tbl, inverse=False)
                            nope_q = quantize_e8m0_roundtrip_bf16(rot_lat[:, :448], 64)
                            new_comp = jnp.concatenate([nope_q, rot_lat[:, 448:]], axis=-1)[0]
                            new_ik_f8, new_ik_sc = indexer_build_k_jax(latent_raw, lw_local, c_pos, cos_tbl, sin_tbl)

                            comp_kv_out = comp_kv.at[c_slot].set(new_comp)
                            ik_f8_out = ik_f8.at[c_slot].set(new_ik_f8[0])
                            ik_sc_out = ik_sc.at[c_slot].set(new_ik_sc[0])
                            n_valid_c_out = jnp.array([p_cur + 1], dtype=jnp.int32)
                            c_tail_out = c_tail
                    else:
                        comp_kv_out = comp_kv
                        ik_f8_out = ik_f8
                        ik_sc_out = ik_sc
                        n_valid_c_out = n_valid_c
                        c_tail_out = c_tail

                    if is_index_source:
                        topk_out = indexer_select_topk_jax(
                            qr, x_norm, lw_local, pos,
                            ik_f8_out, ik_sc_out, n_valid_c_out,
                            cos_tbl, sin_tbl, cfg,
                        )
                    else:
                        topk_out = topk_i

                    o = attention_joint_swa_csa_jax(
                        q, swa_kv_out, swa_pos_out,
                        comp_kv_out, topk_out,
                        pos, a_sink, compress_ratio, cfg.sliding_window,
                    )

                attn_out = o_proj_local_jax(o, pos, cos_tbl, sin_tbl, lw_local, use_bf16_mxu=bf16_mxu)

                res = mhc_post_jax(attn_out, res, post_mix, res_mix)
                ffn_post_mix, ffn_res_mix, ffn_in, ffn_pre = mhc_pre_delayed_jax(
                    res, hcf_fn, hcf_sc, hcf_b, attn_pre,
                    rms_norm_eps=cfg.rms_norm_eps,
                    hc_eps=cfg.hc_eps,
                    sinkhorn_iters=cfg.hc_sinkhorn_iters,
                )
                ffn_normed = rms_norm_f32(ffn_in, f_norm, cfg.rms_norm_eps)
                ffn_out = moe_sublayer_local_jax(
                    ffn_normed, lw_local, cfg, actual_len=None if is_decode else act_len, use_bf16_mxu=bf16_mxu
                )
                res_out = mhc_post_jax(ffn_out, res, ffn_post_mix, ffn_res_mix)

                return (
                    res_out, ffn_pre, swa_kv_out, swa_pos_out,
                    comp_kv_out, ik_f8_out, ik_sc_out, n_valid_c_out, topk_out, c_tail_out,
                )

            return jax.shard_map(
                _body,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            )(
                residual, prev_pre_mix, positions, actual_len,
                cos_plain, sin_plain, cos_yarn, sin_yarn,
                swa_kv_in, swa_pos_in, comp_kv_in, idx_k_f8_in, idx_k_scale_in,
                num_valid_comp_in, topk_indices_in, comp_tail_in,
                attn_norm, ffn_norm, hc_attn_fn, hc_attn_scale, hc_attn_base,
                hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
                wq_a, wq_a_scale, q_norm, wkv, wkv_scale, kv_norm,
                wq_b, wq_b_scale, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale,
                gate_weight, gate_bias,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale,
                routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale,
                comp_wkv, comp_wgate, comp_norm,
                idx_wq_b, idx_wq_b_scale, idx_weights_proj, idx_wk, idx_k_norm,
            )

        self._run_layer = _run_layer

    def _or_dummy(self, x: Optional[jax.Array]) -> jax.Array:
        return x if x is not None else self._dummy_f32

    def _build_engram_windows_prefill(self, token_ids_np: np.ndarray) -> np.ndarray:
        t_len = len(token_ids_np)
        windows = np.full((t_len, 4), -1, dtype=np.int64)
        for s in range(4):
            if t_len > s:
                windows[s:, s] = token_ids_np[: t_len - s]
        return windows

    def prefill(
        self,
        token_ids_np: np.ndarray,
        max_total_len: int = 512,
        return_all_logits: bool = True,
        pad_bucket: int = 64,
        enable_engram: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run exact 40-layer prefill on `token_ids_np` [T] (padded to `pad_bucket`) and return (logits_np, decode_state)."""
        actual_t_len = int(token_ids_np.shape[0])
        pad_len = ((actual_t_len + pad_bucket - 1) // pad_bucket) * pad_bucket
        padded_ids = np.zeros((pad_len,), dtype=np.int32)
        padded_ids[:actual_t_len] = token_ids_np.astype(np.int32)
        positions_np = np.arange(pad_len, dtype=np.int32)

        tids_jax = self.replicate_array(padded_ids)
        pos_jax = self.replicate_array(positions_np)
        act_len_jax = self.replicate_array(np.array([actual_t_len], dtype=np.int32))

        engram_rows_by_layer: Dict[int, jax.Array] = {}
        if enable_engram:
            windows = self._build_engram_windows_prefill(padded_ids)
            hashes = self.weights.engram_host.compute_hashes_for_windows(windows)
            for idx_e, lid in enumerate(self.cfg.engram_layer_ids):
                if idx_e in self.weights.engram_host.weight_mmaps:
                    local_r = self.weights.engram_host.gather_local_cols(hashes, idx_e)
                    engram_rows_by_layer[lid] = self.put_host_engram_rows(local_r)

        residual = self._embed_fn(self.weights.embed_weight, tids_jax)
        prev_pre_mix = self._get_init_pre_mix(pad_len)

        swa_kv_list: List[jax.Array] = []
        swa_pos_out: Optional[jax.Array] = None

        comp_kv_state: Dict[int, jax.Array] = {}
        idx_k_f8_state: Dict[int, jax.Array] = {}
        idx_k_sc_state: Dict[int, jax.Array] = {}
        n_valid_c_state: Dict[int, jax.Array] = {}
        comp_tail_state: Dict[int, jax.Array] = {}

        for lid in self.cfg.kv_source_layer_ids:
            ratio = self.weights.layers[lid].compress_ratio
            max_c = max_total_len // ratio
            comp_kv_state[lid] = self.replicate_array(np.zeros((max_c, 512), dtype=jnp.bfloat16))
            idx_k_f8_state[lid] = self.replicate_array(np.zeros((max_c, 128), dtype=jnp.float8_e4m3fn))
            idx_k_sc_state[lid] = self.replicate_array(np.ones((max_c, 1), dtype=np.float32))
            n_valid_c_state[lid] = self._dummy_i32
            comp_tail_state[lid] = self._dummy_tail

        cur_comp_kv: jax.Array = comp_kv_state[self.cfg.kv_source_layer_ids[0]]
        cur_ik_f8: jax.Array = idx_k_f8_state[self.cfg.kv_source_layer_ids[0]]
        cur_ik_sc: jax.Array = idx_k_sc_state[self.cfg.kv_source_layer_ids[0]]
        cur_n_valid_c: jax.Array = self._dummy_i32
        cur_topk: jax.Array = self.replicate_array(
            np.full((pad_len, self.cfg.index_topk), -1, dtype=np.int32)
        )

        for lw in self.weights.layers:
            lid = lw.layer_id
            if enable_engram and lw.has_engram and lid in engram_rows_by_layer:
                residual = self._engram_fn(
                    residual,
                    engram_rows_by_layer[lid],
                    lw.engram_wkv,
                    lw.engram_wkv_scale,
                    lw.engram_q_weight,
                    lw.engram_k_weight,
                )

            if lw.is_kv_source:
                cur_comp_kv = comp_kv_state[lid]
                cur_ik_f8 = idx_k_f8_state[lid]
                cur_ik_sc = idx_k_sc_state[lid]
                cur_n_valid_c = n_valid_c_state[lid]

            (
                residual,
                prev_pre_mix,
                swa_kv_ring,
                swa_pos_ring,
                comp_kv_out,
                ik_f8_out,
                ik_sc_out,
                n_valid_c_out,
                topk_out,
                c_tail_out,
            ) = self._run_layer(
                lw.compress_ratio,
                lw.is_kv_source,
                lw.is_index_source,
                False,
                residual,
                prev_pre_mix,
                pos_jax,
                act_len_jax,
                self.cos_plain,
                self.sin_plain,
                self.cos_yarn,
                self.sin_yarn,
                self._dummy_f32,
                self._dummy_i32,
                cur_comp_kv,
                cur_ik_f8,
                cur_ik_sc,
                cur_n_valid_c,
                cur_topk,
                self._dummy_tail,
                lw.attn_norm, lw.ffn_norm,
                lw.hc_attn_fn, lw.hc_attn_scale, lw.hc_attn_base,
                lw.hc_ffn_fn, lw.hc_ffn_scale, lw.hc_ffn_base,
                lw.wq_a, lw.wq_a_scale, lw.q_norm,
                lw.wkv, lw.wkv_scale, lw.kv_norm,
                lw.wq_b, lw.wq_b_scale, lw.attn_sink,
                lw.wo_a, lw.wo_a_scale, lw.wo_b, lw.wo_b_scale,
                lw.gate_weight, lw.gate_bias,
                lw.shared_w1, lw.shared_w1_scale,
                lw.shared_w3, lw.shared_w3_scale,
                lw.shared_w2, lw.shared_w2_scale,
                lw.routed_w1, lw.routed_w1_scale,
                lw.routed_w3, lw.routed_w3_scale,
                lw.routed_w2, lw.routed_w2_scale,
                self._or_dummy(lw.comp_wkv),
                self._or_dummy(lw.comp_wgate),
                self._or_dummy(lw.comp_norm),
                self._or_dummy(lw.idx_wq_b),
                self._or_dummy(lw.idx_wq_b_scale),
                self._or_dummy(lw.idx_weights_proj),
                self._or_dummy(lw.idx_wk),
                self._or_dummy(lw.idx_k_norm),
            )

            swa_kv_list.append(swa_kv_ring)
            swa_pos_out = swa_pos_ring

            if lw.is_kv_source:
                cur_comp_kv = comp_kv_out
                cur_ik_f8 = ik_f8_out
                cur_ik_sc = ik_sc_out
                cur_n_valid_c = n_valid_c_out
                comp_kv_state[lid] = comp_kv_out
                idx_k_f8_state[lid] = ik_f8_out
                idx_k_sc_state[lid] = ik_sc_out
                n_valid_c_state[lid] = n_valid_c_out
                comp_tail_state[lid] = c_tail_out

            if lw.is_index_source:
                cur_topk = topk_out

        logits_full = self._lm_head_fn(residual, prev_pre_mix, self.weights.norm_weight, self.weights.head_weight)
        logits_np_full = to_host_np(logits_full, dtype=np.float32)
        logits_np = (
            logits_np_full[:actual_t_len]
            if return_all_logits
            else logits_np_full[actual_t_len - 1 : actual_t_len]
        )

        decode_state = {
            "swa_kv": swa_kv_list,
            "swa_pos": swa_pos_out,
            "comp_kv": comp_kv_state,
            "idx_k_f8": idx_k_f8_state,
            "idx_k_sc": idx_k_sc_state,
            "n_valid_c": n_valid_c_state,
            "comp_tail": comp_tail_state,
            "token_history": list(int(x) for x in token_ids_np),
            "enable_engram": enable_engram,
        }
        return logits_np, decode_state

    def decode_step(
        self,
        next_token_id: int,
        decode_state: Dict[str, Any],
        enable_engram: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run one exact 40-layer decode step for `next_token_id` and update `decode_state`."""
        if enable_engram is None:
            enable_engram = bool(decode_state.get("enable_engram", True))
        hist = decode_state["token_history"]
        p_cur = len(hist)
        hist.append(int(next_token_id))

        engram_rows_by_layer: Dict[int, jax.Array] = {}
        if enable_engram:
            win = np.array(
                [[hist[p_cur - s] if (p_cur - s) >= 0 else -1 for s in range(4)]],
                dtype=np.int64,
            )
            hashes = self.weights.engram_host.compute_hashes_for_windows(win)
            for idx_e, lid in enumerate(self.cfg.engram_layer_ids):
                if idx_e in self.weights.engram_host.weight_mmaps:
                    local_r = self.weights.engram_host.gather_local_cols(hashes, idx_e)
                    engram_rows_by_layer[lid] = self.put_host_engram_rows(local_r)

        tids_jax = self.replicate_array(np.array([next_token_id], dtype=np.int32))
        pos_jax = self.replicate_array(np.array([p_cur], dtype=np.int32))

        residual = self._embed_fn(self.weights.embed_weight, tids_jax)
        prev_pre_mix = self._get_init_pre_mix(1)

        swa_pos_in = decode_state["swa_pos"]
        swa_pos_out = swa_pos_in

        first_kv_lid = self.cfg.kv_source_layer_ids[0]
        cur_comp_kv: jax.Array = decode_state["comp_kv"][first_kv_lid]
        cur_ik_f8: jax.Array = decode_state["idx_k_f8"][first_kv_lid]
        cur_ik_sc: jax.Array = decode_state["idx_k_sc"][first_kv_lid]
        cur_n_valid_c: jax.Array = decode_state["n_valid_c"][first_kv_lid]
        cur_topk: jax.Array = self.replicate_array(
            np.full((1, self.cfg.index_topk), -1, dtype=np.int32)
        )

        for lw in self.weights.layers:
            lid = lw.layer_id
            if enable_engram and lw.has_engram and lid in engram_rows_by_layer:
                residual = self._engram_fn(
                    residual,
                    engram_rows_by_layer[lid],
                    lw.engram_wkv,
                    lw.engram_wkv_scale,
                    lw.engram_q_weight,
                    lw.engram_k_weight,
                )

            if lw.is_kv_source:
                cur_comp_kv = decode_state["comp_kv"][lid]
                cur_ik_f8 = decode_state["idx_k_f8"][lid]
                cur_ik_sc = decode_state["idx_k_sc"][lid]
                cur_n_valid_c = decode_state["n_valid_c"][lid]
                c_tail_in = decode_state["comp_tail"][lid]
            else:
                c_tail_in = self._dummy_tail

            (
                residual,
                prev_pre_mix,
                swa_kv_updated,
                swa_pos_out,
                comp_kv_out,
                ik_f8_out,
                ik_sc_out,
                n_valid_c_out,
                topk_out,
                c_tail_out,
            ) = self._run_layer(
                lw.compress_ratio,
                lw.is_kv_source,
                lw.is_index_source,
                True,
                residual,
                prev_pre_mix,
                pos_jax,
                self._dummy_i32,
                self.cos_plain,
                self.sin_plain,
                self.cos_yarn,
                self.sin_yarn,
                decode_state["swa_kv"][lid],
                swa_pos_in,
                cur_comp_kv,
                cur_ik_f8,
                cur_ik_sc,
                cur_n_valid_c,
                cur_topk,
                c_tail_in,
                lw.attn_norm, lw.ffn_norm,
                lw.hc_attn_fn, lw.hc_attn_scale, lw.hc_attn_base,
                lw.hc_ffn_fn, lw.hc_ffn_scale, lw.hc_ffn_base,
                lw.wq_a, lw.wq_a_scale, lw.q_norm,
                lw.wkv, lw.wkv_scale, lw.kv_norm,
                lw.wq_b, lw.wq_b_scale, lw.attn_sink,
                lw.wo_a, lw.wo_a_scale, lw.wo_b, lw.wo_b_scale,
                lw.gate_weight, lw.gate_bias,
                lw.shared_w1, lw.shared_w1_scale,
                lw.shared_w3, lw.shared_w3_scale,
                lw.shared_w2, lw.shared_w2_scale,
                lw.routed_w1, lw.routed_w1_scale,
                lw.routed_w3, lw.routed_w3_scale,
                lw.routed_w2, lw.routed_w2_scale,
                self._or_dummy(lw.comp_wkv),
                self._or_dummy(lw.comp_wgate),
                self._or_dummy(lw.comp_norm),
                self._or_dummy(lw.idx_wq_b),
                self._or_dummy(lw.idx_wq_b_scale),
                self._or_dummy(lw.idx_weights_proj),
                self._or_dummy(lw.idx_wk),
                self._or_dummy(lw.idx_k_norm),
            )

            decode_state["swa_kv"][lid] = swa_kv_updated
            if lw.is_kv_source:
                decode_state["comp_kv"][lid] = comp_kv_out
                decode_state["idx_k_f8"][lid] = ik_f8_out
                decode_state["idx_k_sc"][lid] = ik_sc_out
                decode_state["n_valid_c"][lid] = n_valid_c_out
                decode_state["comp_tail"][lid] = c_tail_out
                cur_comp_kv = comp_kv_out
                cur_ik_f8 = ik_f8_out
                cur_ik_sc = ik_sc_out
                cur_n_valid_c = n_valid_c_out
            if lw.is_index_source:
                cur_topk = topk_out

        decode_state["swa_pos"] = swa_pos_out
        logits = self._lm_head_fn(residual, prev_pre_mix, self.weights.norm_weight, self.weights.head_weight)
        return to_host_np(logits, dtype=np.float32)[0], decode_state
