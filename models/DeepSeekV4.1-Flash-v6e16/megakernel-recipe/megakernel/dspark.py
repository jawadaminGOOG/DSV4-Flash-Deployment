"""Real `DSpark` (`mtp.0..2`) multi-token prediction draft head and lossless speculative decoding for DeepSeek-V4.1-Flash on TPU v6e-16.

Implements the exact `DSparkBlock`, `DSparkAttention`, `DSparkMarkovHead`, and `DSparkConfidenceHead`
architecture from `inference/model.py:1020-1156, 1274-1283` using the real weights in
`model-00044-of-00048.safetensors` (`mtp.0`), `model-00045-of-00048.safetensors` (`mtp.1`), and
`model-00046-of-00048.safetensors` (`mtp.2`):
  - `3` MTP layers (`mtp.0..2`), each with `compress_ratio = 0`, `128` MXFP4 routed experts (`top-3`,
    `routed_scaling_factor = 1.5`) + `1` FP8 shared expert, and `mHC` (`hc_mult = 4`, `20` Sinkhorn iters).
  - Target hidden fusion (`mtp.0.main_proj` `[5120, 15360]` FP8 + `mtp.0.main_norm` `[5120]` BF16) from
    backbone layers `dspark_target_layer_ids = [37, 38, 39]`.
  - Block draft input (`dspark_block_size = 5`, `dspark_noise_token_id = 128799`) with bidirectional
    intra-block + 128-window causal attention (`DSparkAttention`).
  - Sequential low-rank Markov transition head (`dspark_markov_rank = 256`: `mtp.2.markov_head.embed`
    `[129280, 256]` + `mtp.2.markov_head.head` `[129280, 256]`) and `mtp.2.confidence_head` (`[1, 5376]`).
"""

import functools
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .config import DSV41Config
from .engine_jax import (
    _psum_tp,
    apply_gptj_rope,
    build_rope_caches,
    dequant_fp8_block32_jax,
    dequant_mxfp4_block32_jax,
    fp8_linear_tpu_inference_jax,
    mhc_post_jax,
    mhc_pre_delayed_jax,
    rms_norm_f32,
    wo_a_tpu_inference_jax,
)
from .load import (
    BUCKET_DEFAULT,
    CACHE_DIR_DEFAULT,
    PREFIX_DEFAULT,
    _load_layer_cache,
    _make_replicated_array,
    _make_sharded_array,
    _save_layer_cache,
    _stream_safetensors_dict,
)


@dataclass
class DSparkStageWeights:
    """Sharded weights for one `mtp.{stage_id}` (`stage_id in {0, 1, 2}`) DSpark layer."""

    stage_id: int
    attn_norm: jax.Array          # [5120] bf16
    ffn_norm: jax.Array           # [5120] bf16
    hc_attn_fn: jax.Array         # [24, 20480] f32
    hc_attn_scale: jax.Array      # [3] f32
    hc_attn_base: jax.Array       # [24] f32
    hc_ffn_fn: jax.Array          # [24, 20480] f32
    hc_ffn_scale: jax.Array       # [3] f32
    hc_ffn_base: jax.Array        # [24] f32

    wq_a: jax.Array               # [1280, 5120] f8
    wq_a_scale: jax.Array         # [40, 160] u8
    q_norm: jax.Array             # [1280] bf16
    wkv: jax.Array                # [512, 5120] f8
    wkv_scale: jax.Array          # [16, 160] u8
    kv_norm: jax.Array            # [512] bf16

    wq_b: jax.Array               # P("tp", None): [65536, 1280] f8 (per-chip [4096, 1280])
    wq_b_scale: jax.Array         # P("tp", None): [2048, 40] u8 (per-chip [128, 40])
    attn_sink: jax.Array          # P("tp"): [128] f32 (per-chip [8])
    wo_a: jax.Array               # P("tp", None): [8192, 4096] f8 (per-chip [512, 4096])
    wo_a_scale: jax.Array         # P("tp", None): [256, 128] u8 (per-chip [16, 128])
    wo_b: jax.Array               # P(None, "tp"): [5120, 8192] f8 (per-chip [5120, 512])
    wo_b_scale: jax.Array         # P(None, "tp"): [160, 256] u8 (per-chip [160, 16])

    gate_weight: jax.Array        # [128, 5120] bf16
    gate_bias: jax.Array          # [128] f32
    shared_w1: jax.Array          # P("tp", None): [2560, 5120] f8
    shared_w1_scale: jax.Array    # P("tp", None): [80, 160] u8
    shared_w3: jax.Array          # P("tp", None): [2560, 5120] f8
    shared_w3_scale: jax.Array    # P("tp", None): [80, 160] u8
    shared_w2: jax.Array          # P(None, "tp"): [5120, 2560] f8
    shared_w2_scale: jax.Array    # P(None, "tp"): [160, 80] u8

    routed_w1: jax.Array          # P(None, "tp", None): [128, 2560, 2560] i8
    routed_w1_scale: jax.Array    # P(None, "tp", None): [128, 2560, 160] u8
    routed_w3: jax.Array          # P(None, "tp", None): [128, 2560, 2560] i8
    routed_w3_scale: jax.Array    # P(None, "tp", None): [128, 2560, 160] u8
    routed_w2: jax.Array          # P(None, None, "tp"): [128, 5120, 1280] i8
    routed_w2_scale: jax.Array    # P(None, None, "tp"): [128, 5120, 80] u8


@dataclass
class DSparkWeights:
    """Full 3-stage `mtp.0..2` sharded weight container for DeepSeek-V4.1-Flash DSpark."""

    cfg: DSV41Config
    mesh: Mesh
    embed_weight: jax.Array       # P(None, "tp"): [129280, 5120] bf16
    head_weight: jax.Array        # P("tp", None): [129280, 5120] bf16

    main_proj_weight: jax.Array   # P(): [5120, 15360] f8
    main_proj_scale: jax.Array    # P(): [160, 480] u8
    main_norm_weight: jax.Array   # P(): [5120] bf16

    stages: Tuple[DSparkStageWeights, ...]  # 3 stages: mtp.0, mtp.1, mtp.2

    mtp_norm_weight: jax.Array    # P(): [5120] bf16 (mtp.2.norm.weight)
    markov_embed: jax.Array       # P(None, "tp"): [129280, 256] bf16 (per-chip [129280, 16])
    markov_head: jax.Array        # P("tp", None): [129280, 256] bf16 (per-chip [8080, 256])
    confidence_proj: jax.Array    # P(): [1, 5376] bf16 (mtp.2.confidence_head.proj.weight)


def _pack_mtp_stage_for_host(
    raw: Dict[str, np.ndarray],
    stage_id: int,
    c_start: int,
    c_end: int,
) -> Dict[str, np.ndarray]:
    """Pack `mtp.{stage_id}` tensors for the 4 local chips `[c_start, c_end)` on this host."""
    p = f"mtp.{stage_id}"
    n_local = c_end - c_start
    packed: Dict[str, np.ndarray] = {}

    packed["attn_norm"] = raw[f"{p}.attn_norm.weight"]
    packed["ffn_norm"] = raw[f"{p}.ffn_norm.weight"]
    packed["hc_attn_fn"] = raw[f"{p}.hc_attn_fn"]
    packed["hc_attn_scale"] = raw[f"{p}.hc_attn_scale"]
    packed["hc_attn_base"] = raw[f"{p}.hc_attn_base"]
    packed["hc_ffn_fn"] = raw[f"{p}.hc_ffn_fn"]
    packed["hc_ffn_scale"] = raw[f"{p}.hc_ffn_scale"]
    packed["hc_ffn_base"] = raw[f"{p}.hc_ffn_base"]

    packed["wq_a"] = raw[f"{p}.attn.wq_a.weight"]
    packed["wq_a_scale"] = raw[f"{p}.attn.wq_a.scale"]
    packed["q_norm"] = raw[f"{p}.attn.q_norm.weight"]
    packed["wkv"] = raw[f"{p}.attn.wkv.weight"]
    packed["wkv_scale"] = raw[f"{p}.attn.wkv.scale"]
    packed["kv_norm"] = raw[f"{p}.attn.kv_norm.weight"]

    # wq_b in mtp.0..2 has compress_ratio == 0 -> [32768, 1280] (8 groups * 8 heads * 512)
    # Group-paired wq_b & attn_sink: chip c gets group g = c // 2 (matching load.py:622-632)
    wqb_raw = raw[f"{p}.attn.wq_b.weight"].reshape(8, 4096, 1280)
    wqb_s_raw = raw[f"{p}.attn.wq_b.scale"].reshape(8, 128, 40)
    sink_raw = raw[f"{p}.attn.attn_sink"].reshape(8, 8)

    wqb_host = np.stack([wqb_raw[c // 2] for c in range(c_start, c_end)], axis=0)      # [4, 4096, 1280]
    wqb_s_host = np.stack([wqb_s_raw[c // 2] for c in range(c_start, c_end)], axis=0)  # [4, 128, 40]
    sink_host = np.stack([sink_raw[c // 2] for c in range(c_start, c_end)], axis=0)    # [4, 8]

    woa_raw = raw[f"{p}.attn.wo_a.weight"]
    woa_s_raw = raw[f"{p}.attn.wo_a.scale"]
    wob_raw = raw[f"{p}.attn.wo_b.weight"]
    wob_s_raw = raw[f"{p}.attn.wo_b.scale"]

    woa_host = np.zeros((n_local, 512, 4096), dtype=ml_dtypes.float8_e4m3fn)
    woa_s_host = np.zeros((n_local, 16, 128), dtype=np.uint8)
    wob_host = np.zeros((n_local, 5120, 512), dtype=ml_dtypes.float8_e4m3fn)
    wob_s_host = np.zeros((n_local, 160, 16), dtype=np.uint8)

    for local_c, chip_id in enumerate(range(c_start, c_end)):
        woa_host[local_c] = woa_raw[chip_id * 512 : (chip_id + 1) * 512, :]
        woa_s_host[local_c] = woa_s_raw[chip_id * 16 : (chip_id + 1) * 16, :]
        wob_host[local_c] = wob_raw[:, chip_id * 512 : (chip_id + 1) * 512]
        wob_s_host[local_c] = wob_s_raw[:, chip_id * 16 : (chip_id + 1) * 16]

    packed["wq_b"] = wqb_host
    packed["wq_b_scale"] = wqb_s_host
    packed["attn_sink"] = sink_host
    packed["wo_a"] = woa_host
    packed["wo_a_scale"] = woa_s_host
    packed["wo_b"] = wob_host
    packed["wo_b_scale"] = wob_s_host

    # Gate (128 experts) + Shared Expert
    packed["gate_weight"] = raw[f"{p}.ffn.gate.weight"]
    packed["gate_bias"] = raw[f"{p}.ffn.gate.bias"]

    sw1_raw = raw[f"{p}.ffn.shared_experts.w1.weight"]
    sw1_s_raw = raw[f"{p}.ffn.shared_experts.w1.scale"]
    sw3_raw = raw[f"{p}.ffn.shared_experts.w3.weight"]
    sw3_s_raw = raw[f"{p}.ffn.shared_experts.w3.scale"]
    sw2_raw = raw[f"{p}.ffn.shared_experts.w2.weight"]
    sw2_s_raw = raw[f"{p}.ffn.shared_experts.w2.scale"]

    sw1_host = np.zeros((n_local, 160, 5120), dtype=ml_dtypes.float8_e4m3fn)
    sw1_s_host = np.ones((n_local, 5, 160), dtype=np.uint8) * 127
    sw3_host = np.zeros((n_local, 160, 5120), dtype=ml_dtypes.float8_e4m3fn)
    sw3_s_host = np.ones((n_local, 5, 160), dtype=np.uint8) * 127
    sw2_host = np.zeros((n_local, 5120, 160), dtype=ml_dtypes.float8_e4m3fn)
    sw2_s_host = np.ones((n_local, 160, 5), dtype=np.uint8) * 127

    for local_c, chip_id in enumerate(range(c_start, c_end)):
        r0 = chip_id * 160
        r1 = min(2304, (chip_id + 1) * 160)
        b0 = chip_id * 5
        b1 = min(72, (chip_id + 1) * 5)
        if r0 < 2304:
            n_r = r1 - r0
            n_b = b1 - b0
            sw1_host[local_c, :n_r, :] = sw1_raw[r0:r1, :]
            sw1_s_host[local_c, :n_b, :] = sw1_s_raw[b0:b1, :]
            sw3_host[local_c, :n_r, :] = sw3_raw[r0:r1, :]
            sw3_s_host[local_c, :n_b, :] = sw3_s_raw[b0:b1, :]
            sw2_host[local_c, :, :n_r] = sw2_raw[:, r0:r1]
            sw2_s_host[local_c, :, :n_b] = sw2_s_raw[:, b0:b1]

    packed["shared_w1"] = sw1_host
    packed["shared_w1_scale"] = sw1_s_host
    packed["shared_w3"] = sw3_host
    packed["shared_w3_scale"] = sw3_s_host
    packed["shared_w2"] = sw2_host
    packed["shared_w2_scale"] = sw2_s_host

    # 128 Routed Experts (MXFP4)
    n_exp = 128
    rw1_host = np.zeros((n_local, n_exp, 160, 2560), dtype=np.int8)
    rw1_s_host = np.ones((n_local, n_exp, 160, 160), dtype=np.uint8) * 127
    rw3_host = np.zeros((n_local, n_exp, 160, 2560), dtype=np.int8)
    rw3_s_host = np.ones((n_local, n_exp, 160, 160), dtype=np.uint8) * 127
    rw2_host = np.zeros((n_local, n_exp, 5120, 80), dtype=np.int8)
    rw2_s_host = np.ones((n_local, n_exp, 5120, 5), dtype=np.uint8) * 127

    for e in range(n_exp):
        ep = f"{p}.ffn.experts.{e}"
        w1_e = raw[f"{ep}.w1.weight"]
        s1_e = raw[f"{ep}.w1.scale"]
        w3_e = raw[f"{ep}.w3.weight"]
        s3_e = raw[f"{ep}.w3.scale"]
        w2_e = raw[f"{ep}.w2.weight"]
        s2_e = raw[f"{ep}.w2.scale"]

        for local_c, chip_id in enumerate(range(c_start, c_end)):
            r0 = chip_id * 160
            r1 = min(2304, (chip_id + 1) * 160)
            if r0 < 2304:
                n_r = r1 - r0
                rw1_host[local_c, e, :n_r, :] = w1_e[r0:r1, :]
                rw1_s_host[local_c, e, :n_r, :] = s1_e[r0:r1, :]
                rw3_host[local_c, e, :n_r, :] = w3_e[r0:r1, :]
                rw3_s_host[local_c, e, :n_r, :] = s3_e[r0:r1, :]

            k0 = chip_id * 80
            k1 = min(1152, (chip_id + 1) * 80)
            b0 = chip_id * 5
            b1 = min(72, (chip_id + 1) * 5)
            if k0 < 1152:
                rw2_host[local_c, e, :, : (k1 - k0)] = w2_e[:, k0:k1]
                rw2_s_host[local_c, e, :, : (b1 - b0)] = s2_e[:, b0:b1]

    packed["routed_w1"] = rw1_host
    packed["routed_w1_scale"] = rw1_s_host
    packed["routed_w3"] = rw3_host
    packed["routed_w3_scale"] = rw3_s_host
    packed["routed_w2"] = rw2_host
    packed["routed_w2_scale"] = rw2_s_host

    if stage_id == 0:
        packed["main_proj_weight"] = raw["mtp.0.main_proj.weight"]
        packed["main_proj_scale"] = raw["mtp.0.main_proj.scale"]
        packed["main_norm_weight"] = raw["mtp.0.main_norm.weight"]
    if stage_id == 2:
        packed["mtp_norm_weight"] = raw["mtp.2.norm.weight"]
        packed["confidence_proj"] = raw["mtp.2.confidence_head.proj.weight"]
        me_raw = raw["mtp.2.markov_head.embed.weight"]  # [129280, 256] bf16
        mh_raw = raw["mtp.2.markov_head.head.weight"]   # [129280, 256] bf16
        packed["markov_embed"] = np.stack(
            [me_raw[:, c * 16 : (c + 1) * 16] for c in range(c_start, c_end)], axis=0
        )
        packed["markov_head"] = np.stack(
            [mh_raw[c * 8080 : (c + 1) * 8080, :] for c in range(c_start, c_end)], axis=0
        )

    return packed


def load_dspark_weights(
    cfg: DSV41Config,
    mesh: Mesh,
    embed_weight: jax.Array,
    head_weight: jax.Array,
    model_uri: str = f"gs://{BUCKET_DEFAULT}/{PREFIX_DEFAULT}",
    cache_dir: str = CACHE_DIR_DEFAULT,
) -> DSparkWeights:
    """Stream `mtp.0..2` (`model-00044..00046-of-00048.safetensors`) and shard across 16 TPU v6e devices."""
    os.makedirs(cache_dir, exist_ok=True)
    host_idx = jax.process_index()
    devices_by_process = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
    local_devs = devices_by_process[host_idx * 4 : (host_idx + 1) * 4]
    c_start = host_idx * 4
    c_end = c_start + 4

    stages: List[DSparkStageWeights] = []
    main_proj_w = None
    main_proj_s = None
    main_norm_w = None
    mtp_norm_w = None
    markov_embed = None
    markov_head = None
    conf_proj = None

    _deq_fp8 = jax.jit(dequant_fp8_block32_jax)
    _deq_fp4 = jax.jit(jax.vmap(dequant_mxfp4_block32_jax))

    for stage_id in range(3):
        shard_idx = 44 + stage_id
        cache_file = os.path.join(cache_dir, f"mtp_{stage_id}_host{host_idx}.npz")
        if os.path.exists(cache_file):
            packed = _load_layer_cache(cache_file)
        else:
            shard_uri = f"{model_uri.rstrip('/')}/model-000{shard_idx:02d}-of-00048.safetensors"
            raw = _stream_safetensors_dict(shard_uri)
            packed = _pack_mtp_stage_for_host(raw, stage_id, c_start, c_end)
            _save_layer_cache(cache_file, packed)
            del raw

        def rep(key: str) -> jax.Array:
            return _make_replicated_array(mesh, local_devs, packed[key])

        def sharded(key: str, global_shape: Tuple[int, ...], spec: P) -> jax.Array:
            return _make_sharded_array(
                mesh, local_devs, global_shape, spec, [packed[key][i] for i in range(4)]
            )

        def rep_fp8_bf16(w_key: str, s_key: str) -> jax.Array:
            w_np = np.ascontiguousarray(packed[w_key])
            s_np = np.ascontiguousarray(packed[s_key])
            dev_arrs = [
                _deq_fp8(jax.device_put(w_np, d), jax.device_put(s_np, d))
                for d in local_devs
            ]
            return jax.make_array_from_single_device_arrays(
                w_np.shape, NamedSharding(mesh, P()), dev_arrs
            )

        def sharded_fp8_bf16(
            w_key: str, s_key: str, global_shape: Tuple[int, ...], spec: P
        ) -> jax.Array:
            dev_arrs = [
                _deq_fp8(
                    jax.device_put(np.ascontiguousarray(packed[w_key][i]), local_devs[i]),
                    jax.device_put(np.ascontiguousarray(packed[s_key][i]), local_devs[i]),
                )
                for i in range(4)
            ]
            return jax.make_array_from_single_device_arrays(
                global_shape, NamedSharding(mesh, spec), dev_arrs
            )

        def sharded_fp4_bf16(
            w_key: str, s_key: str, global_shape: Tuple[int, ...], spec: P
        ) -> jax.Array:
            dev_arrs = [
                _deq_fp4(
                    jax.device_put(np.ascontiguousarray(packed[w_key][i]), local_devs[i]),
                    jax.device_put(np.ascontiguousarray(packed[s_key][i]), local_devs[i]),
                )
                for i in range(4)
            ]
            return jax.make_array_from_single_device_arrays(
                global_shape, NamedSharding(mesh, spec), dev_arrs
            )

        stages.append(
            DSparkStageWeights(
                stage_id=stage_id,
                attn_norm=rep("attn_norm"),
                ffn_norm=rep("ffn_norm"),
                hc_attn_fn=rep("hc_attn_fn"),
                hc_attn_scale=rep("hc_attn_scale"),
                hc_attn_base=rep("hc_attn_base"),
                hc_ffn_fn=rep("hc_ffn_fn"),
                hc_ffn_scale=rep("hc_ffn_scale"),
                hc_ffn_base=rep("hc_ffn_base"),
                wq_a=rep_fp8_bf16("wq_a", "wq_a_scale"),
                wq_a_scale=rep("wq_a_scale"),
                q_norm=rep("q_norm"),
                wkv=rep_fp8_bf16("wkv", "wkv_scale"),
                wkv_scale=rep("wkv_scale"),
                kv_norm=rep("kv_norm"),
                wq_b=sharded_fp8_bf16("wq_b", "wq_b_scale", (65536, 1280), P("tp", None)),
                wq_b_scale=sharded("wq_b_scale", (2048, 40), P("tp", None)),
                attn_sink=sharded("attn_sink", (128,), P("tp")),
                wo_a=sharded_fp8_bf16("wo_a", "wo_a_scale", (8192, 4096), P("tp", None)),
                wo_a_scale=sharded("wo_a_scale", (256, 128), P("tp", None)),
                wo_b=sharded_fp8_bf16("wo_b", "wo_b_scale", (5120, 8192), P(None, "tp")),
                wo_b_scale=sharded("wo_b_scale", (160, 256), P(None, "tp")),
                gate_weight=rep("gate_weight"),
                gate_bias=rep("gate_bias"),
                shared_w1=sharded_fp8_bf16("shared_w1", "shared_w1_scale", (2560, 5120), P("tp", None)),
                shared_w1_scale=sharded("shared_w1_scale", (80, 160), P("tp", None)),
                shared_w3=sharded_fp8_bf16("shared_w3", "shared_w3_scale", (2560, 5120), P("tp", None)),
                shared_w3_scale=sharded("shared_w3_scale", (80, 160), P("tp", None)),
                shared_w2=sharded_fp8_bf16("shared_w2", "shared_w2_scale", (5120, 2560), P(None, "tp")),
                shared_w2_scale=sharded("shared_w2_scale", (160, 80), P(None, "tp")),
                routed_w1=sharded_fp4_bf16("routed_w1", "routed_w1_scale", (128, 2560, 5120), P(None, "tp", None)),
                routed_w1_scale=sharded("routed_w1_scale", (128, 2560, 160), P(None, "tp", None)),
                routed_w3=sharded_fp4_bf16("routed_w3", "routed_w3_scale", (128, 2560, 5120), P(None, "tp", None)),
                routed_w3_scale=sharded("routed_w3_scale", (128, 2560, 160), P(None, "tp", None)),
                routed_w2=sharded_fp4_bf16("routed_w2", "routed_w2_scale", (128, 5120, 2560), P(None, None, "tp")),
                routed_w2_scale=sharded("routed_w2_scale", (128, 5120, 80), P(None, None, "tp")),
            )
        )
        if stage_id == 0:
            mpw_full = packed["main_proj_weight"]  # [5120, 15360] f8
            mps_full = packed["main_proj_scale"]   # [160, 480] u8
            mpw_devs = [
                _deq_fp8(
                    jax.device_put(
                        np.ascontiguousarray(mpw_full[:, (c_start + i) * 960 : (c_start + i + 1) * 960]),
                        local_devs[i],
                    ),
                    jax.device_put(
                        np.ascontiguousarray(mps_full[:, (c_start + i) * 30 : (c_start + i + 1) * 30]),
                        local_devs[i],
                    ),
                )
                for i in range(4)
            ]
            main_proj_w = jax.make_array_from_single_device_arrays(
                (5120, 15360), NamedSharding(mesh, P(None, "tp")), mpw_devs
            )
            mps_devs = [
                jax.device_put(
                    np.ascontiguousarray(mps_full[:, (c_start + i) * 30 : (c_start + i + 1) * 30]),
                    local_devs[i],
                )
                for i in range(4)
            ]
            main_proj_s = jax.make_array_from_single_device_arrays(
                (160, 480), NamedSharding(mesh, P(None, "tp")), mps_devs
            )
            main_norm_w = rep("main_norm_weight")
        if stage_id == 2:
            mtp_norm_w = rep("mtp_norm_weight")
            conf_proj = rep("confidence_proj")
            markov_embed = sharded("markov_embed", (129280, 256), P(None, "tp"))
            markov_head = sharded("markov_head", (129280, 256), P("tp", None))

    return DSparkWeights(
        cfg=cfg,
        mesh=mesh,
        embed_weight=embed_weight,
        head_weight=head_weight,
        main_proj_weight=main_proj_w,
        main_proj_scale=main_proj_s,
        main_norm_weight=main_norm_w,
        stages=tuple(stages),
        mtp_norm_weight=mtp_norm_w,
        markov_embed=markov_embed,
        markov_head=markov_head,
        confidence_proj=conf_proj,
    )


def _linear_bf16_or_fp8(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    sharded_k: bool = False,
) -> jax.Array:
    """1-pass BF16 MXU linear using pre-dequantized `bfloat16` weight (or FP8 fallback)."""
    w_bf16 = w if w.dtype == jnp.bfloat16 else dequant_fp8_block32_jax(w, w_scale)
    out = jnp.dot(
        x.astype(jnp.bfloat16),
        w_bf16.T,
        preferred_element_type=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
    ).astype(x.dtype)
    if sharded_k:
        out = _psum_tp(out, use_bf16_mxu=True)
    return out


def _dspark_attn_local_jax(
    x_norm: jax.Array,
    draft_pos: jax.Array,
    main_x: jax.Array,
    start_pos: jax.Array,
    swa_kv: jax.Array,
    swa_pos: jax.Array,
    cos_plain: jax.Array,
    sin_plain: jax.Array,
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
    eps: float = 1e-20,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Exact `DSparkAttention.forward` (`model.py:1032-1074`) for one MTP stage inside `shard_map`."""
    # 1. Project `main_x` ([1, 5120] or [5, 5120]) and `x_norm` ([5, 5120]) through `wkv` in one batched matmul
    m_rows = main_x.shape[0]
    both_x = jnp.concatenate([main_x, x_norm], axis=0)
    both_kv = rms_norm_f32(
        _linear_bf16_or_fp8(both_x, wkv, wkv_scale, sharded_k=False),
        kv_norm,
        eps,
    )
    main_kv = both_kv[:m_rows]
    kv_draft = both_kv[m_rows:]

    if start_pos.shape[0] == 1:
        main_kv_rot = apply_gptj_rope(
            main_kv, start_pos, cos_plain, sin_plain, inverse=False
        )
        p0 = start_pos[0]
        slot0 = p0 % 128
        swa_kv_out = swa_kv.at[slot0].set(main_kv_rot[0])
        swa_pos_out = swa_pos.at[slot0].set(p0)
    else:
        n_acc_val = start_pos[0]
        p_base = start_pos[1]
        p_5 = p_base + jnp.arange(5, dtype=jnp.int32)
        main_kv_rot = apply_gptj_rope(
            main_kv, p_5, cos_plain, sin_plain, inverse=False
        )
        swa_kv_out = swa_kv
        swa_pos_out = swa_pos
        for r in range(5):
            active_r = r <= n_acc_val
            p_r = p_base + r
            slot_r = p_r % 128
            swa_kv_out = swa_kv_out.at[slot_r].set(
                jnp.where(active_r, main_kv_rot[r], swa_kv_out[slot_r])
            )
            swa_pos_out = swa_pos_out.at[slot_r].set(
                jnp.where(active_r, p_r, swa_pos_out[slot_r])
            )
        p0 = p_base + n_acc_val

    # 2. Project draft block `x_norm` [5, 5120] at `draft_pos` [5] across all 8 group heads
    qr = rms_norm_f32(
        _linear_bf16_or_fp8(x_norm, wq_a, wq_a_scale, sharded_k=False),
        q_norm,
        eps,
    )
    q_full = _linear_bf16_or_fp8(qr, wq_b, wq_b_scale, sharded_k=False)  # [5, 4096]
    q_swa = q_full.reshape(5, 8, 512)
    q_rot = apply_gptj_rope(q_swa, draft_pos, cos_plain, sin_plain, inverse=False)
    kv_draft_rot = apply_gptj_rope(
        kv_draft, draft_pos, cos_plain, sin_plain, inverse=False
    )  # [5, 512]

    # 3. Bidirectional intra-block + causal 128-window attention (`model.py:1021-1029, 1064-1067`)
    sm_scale = 512.0 ** -0.5
    win_valid = (swa_pos_out >= 0) & (swa_pos_out <= p0) & ((p0 - swa_pos_out) < 128)  # [128]
    win_logits = (
        jnp.einsum(
            "bhd,sd->bhs",
            q_rot.astype(jnp.bfloat16),
            swa_kv_out.astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.DEFAULT,
        )
        * sm_scale
    )
    win_logits = jnp.where(win_valid[None, None, :], win_logits, -jnp.inf)

    blk_logits = (
        jnp.einsum(
            "bhd,td->bht",
            q_rot.astype(jnp.bfloat16),
            kv_draft_rot.astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.DEFAULT,
        )
        * sm_scale
    )  # [5, 8, 5] — all 5 draft tokens attend bidirectionally to all 5 block tokens

    sink_swa = attn_sink  # [8]
    m_win = jnp.max(win_logits, axis=-1)
    m_blk = jnp.max(blk_logits, axis=-1)
    m = jnp.maximum(jnp.maximum(m_win, m_blk), sink_swa[None, :])

    w_win = jnp.exp(win_logits - m[:, :, None])
    w_blk = jnp.exp(blk_logits - m[:, :, None])
    denom = (
        jnp.sum(w_win, axis=-1)
        + jnp.sum(w_blk, axis=-1)
        + jnp.exp(sink_swa[None, :] - m)
    )
    num = jnp.einsum(
        "bhs,sd->bhd",
        w_win.astype(jnp.bfloat16),
        swa_kv_out.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT,
    ) + jnp.einsum(
        "bht,td->bhd",
        w_blk.astype(jnp.bfloat16),
        kv_draft_rot.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT,
    )
    o_swa = (num / denom[:, :, None]).astype(jnp.bfloat16)  # [5, 8, 512]
    o_unrot = apply_gptj_rope(o_swa, draft_pos, cos_plain, sin_plain, inverse=True)

    o_flat = o_unrot.reshape(5, 4096)
    z_local = _linear_bf16_or_fp8(o_flat, wo_a, wo_a_scale, sharded_k=False)
    attn_out = _linear_bf16_or_fp8(z_local, wo_b, wo_b_scale, sharded_k=True)
    return attn_out, swa_kv_out, swa_pos_out


def _dspark_moe_local_jax(
    x_norm: jax.Array,
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
    routed_scaling_factor: float = 1.5,
    swiglu_limit: float = 10.0,
) -> jax.Array:
    """Exact 128-expert `top-3` MoE + 1 shared expert (`model.py:800-905`) for one MTP stage."""
    logits = jnp.dot(
        x_norm.astype(jnp.bfloat16),
        gate_weight.astype(jnp.bfloat16).T,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT,
    )  # [5, 128]
    scores = jnp.sqrt(jax.nn.softplus(logits))
    biased = scores + gate_bias[None, :]
    _, topk_idx = jax.lax.top_k(biased, 3)  # [5, 3]
    topk_w = jnp.take_along_axis(scores, topk_idx, axis=-1)
    topk_w = (
        (topk_w / (jnp.sum(topk_w, axis=-1, keepdims=True) + 1e-20))
        * routed_scaling_factor
    ).astype(jnp.bfloat16)

    g_sh = _linear_bf16_or_fp8(
        x_norm, shared_w1, shared_w1_scale, sharded_k=False
    ).astype(jnp.float32)
    u_sh = _linear_bf16_or_fp8(
        x_norm, shared_w3, shared_w3_scale, sharded_k=False
    ).astype(jnp.float32)
    g_sh = jnp.minimum(g_sh, swiglu_limit)
    u_sh = jnp.clip(u_sh, -swiglu_limit, swiglu_limit)
    h_sh = (jax.nn.silu(g_sh) * u_sh).astype(jnp.bfloat16)
    shared_out = _linear_bf16_or_fp8(
        h_sh, shared_w2, shared_w2_scale, sharded_k=True
    )

    n_tok = x_norm.shape[0]
    flat_idx = topk_idx.reshape(-1)  # [n_tok * 3]
    if routed_w1.dtype == jnp.bfloat16:
        w1_bf16 = routed_w1[flat_idx].reshape(n_tok, 3, 160, 5120)
        w3_bf16 = routed_w3[flat_idx].reshape(n_tok, 3, 160, 5120)
        w2_bf16 = routed_w2[flat_idx].reshape(n_tok, 3, 5120, 160)
    else:
        w1_bf16 = jax.vmap(dequant_mxfp4_block32_jax)(
            routed_w1[flat_idx], routed_w1_scale[flat_idx]
        ).reshape(n_tok, 3, 160, 5120)
        w3_bf16 = jax.vmap(dequant_mxfp4_block32_jax)(
            routed_w3[flat_idx], routed_w3_scale[flat_idx]
        ).reshape(n_tok, 3, 160, 5120)
        w2_bf16 = jax.vmap(dequant_mxfp4_block32_jax)(
            routed_w2[flat_idx], routed_w2_scale[flat_idx]
        ).reshape(n_tok, 3, 5120, 160)

    x_bf16 = x_norm.astype(jnp.bfloat16)
    g_rt = jnp.einsum(
        "tknd,td->tkn",
        w1_bf16,
        x_bf16,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT,
    )
    u_rt = jnp.einsum(
        "tknd,td->tkn",
        w3_bf16,
        x_bf16,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT,
    )
    g_rt = jnp.minimum(g_rt, swiglu_limit)
    u_rt = jnp.clip(u_rt, -swiglu_limit, swiglu_limit)
    h_rt = (jax.nn.silu(g_rt) * u_rt).astype(jnp.bfloat16)
    y_rt = jnp.einsum(
        "tkdn,tkn->tkd",
        w2_bf16,
        h_rt,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT,
    ).astype(jnp.bfloat16)
    routed_local = jnp.sum(
        y_rt.astype(jnp.float32) * topk_w.astype(jnp.float32)[:, :, None], axis=1
    )
    routed_total = _psum_tp(routed_local.astype(jnp.bfloat16), use_bf16_mxu=True)
    return (shared_out + routed_total).astype(jnp.bfloat16)


class DSparkDraftEngine:
    """16-chip compiled DSpark (`mtp.0..2`) draft head on TPU v6e-16."""

    def __init__(self, weights: DSparkWeights, max_seq_len: int = 4096) -> None:
        self.weights = weights
        self.cfg = weights.cfg
        self.mesh = weights.mesh
        self.rep_s = NamedSharding(self.mesh, P())
        host_idx = jax.process_index()
        self.local_devs = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))[
            host_idx * 4 : (host_idx + 1) * 4
        ]
        cos_p, sin_p, _, _ = build_rope_caches(self.cfg, max_seq_len=max_seq_len)
        self.cos_plain = self.replicate(np.asarray(cos_p))
        self.sin_plain = self.replicate(np.asarray(sin_p))
        self._compile()

    def replicate(self, arr: np.ndarray) -> jax.Array:
        a = np.ascontiguousarray(arr)
        return jax.make_array_from_single_device_arrays(
            a.shape, self.rep_s, [jax.device_put(a, d) for d in self.local_devs]
        )

    def init_kv_state(self) -> Tuple[jax.Array, jax.Array]:
        swa_kv = self.replicate(np.zeros((3, 128, 512), dtype=ml_dtypes.bfloat16))
        swa_pos = self.replicate(np.full((128,), -1, dtype=np.int32))
        return swa_kv, swa_pos

    def _compile(self) -> None:
        mesh = self.mesh
        cfg = self.cfg

        # 1. Seed / update MTP window KV cache for 1 token position (`start_pos == 0` / accepted draft tokens)
        @functools.partial(jax.jit, out_shardings=(self.rep_s, self.rep_s))
        def _update_kv_only(
            main_hidden_1x15360: jax.Array,
            pos_1d: jax.Array,
            swa_kv_in: jax.Array,
            swa_pos_in: jax.Array,
            cos_p: jax.Array,
            sin_p: jax.Array,
            main_proj_w: jax.Array,
            main_proj_s: jax.Array,
            main_norm_w: jax.Array,
            wkv0: jax.Array, wkv0_s: jax.Array, kvn0: jax.Array,
            wkv1: jax.Array, wkv1_s: jax.Array, kvn1: jax.Array,
            wkv2: jax.Array, wkv2_s: jax.Array, kvn2: jax.Array,
        ):
            def _local(
                mh, p_arr, skv, spos, cp, sp, mpw, mps, mnw,
                w0, s0, kn0, w1, s1, kn1, w2, s2, kn2,
            ):
                chip_idx = jax.lax.axis_index("tp")
                mh_part = jax.lax.dynamic_slice_in_dim(mh, chip_idx * 960, 960, axis=-1)
                main_x = rms_norm_f32(
                    _linear_bf16_or_fp8(mh_part, mpw, mps, sharded_k=True),
                    mnw,
                    cfg.rms_norm_eps,
                )  # [T, 5120]
                t_len = main_x.shape[0]
                kv0 = apply_gptj_rope(
                    rms_norm_f32(_linear_bf16_or_fp8(main_x, w0, s0, sharded_k=False), kn0, cfg.rms_norm_eps),
                    p_arr, cp, sp, inverse=False,
                )
                kv1 = apply_gptj_rope(
                    rms_norm_f32(_linear_bf16_or_fp8(main_x, w1, s1, sharded_k=False), kn1, cfg.rms_norm_eps),
                    p_arr, cp, sp, inverse=False,
                )
                kv2 = apply_gptj_rope(
                    rms_norm_f32(_linear_bf16_or_fp8(main_x, w2, s2, sharded_k=False), kn2, cfg.rms_norm_eps),
                    p_arr, cp, sp, inverse=False,
                )

                def _write_step(i: int, carry):
                    cur_kv, cur_pos = carry
                    p_i = p_arr[i]
                    slot = p_i % 128
                    cur_kv = cur_kv.at[0, slot].set(kv0[i])
                    cur_kv = cur_kv.at[1, slot].set(kv1[i])
                    cur_kv = cur_kv.at[2, slot].set(kv2[i])
                    cur_pos = cur_pos.at[slot].set(p_i)
                    return cur_kv, cur_pos

                return jax.lax.fori_loop(0, t_len, _write_step, (skv, spos))

            return jax.shard_map(
                _local,
                mesh=mesh,
                in_specs=(
                    P(), P(), P(), P(), P(), P(),
                    P(None, "tp"), P(None, "tp"), P(),
                    P(), P(), P(), P(), P(), P(), P(), P(), P(),
                ),
                out_specs=(P(), P()),
                check_vma=False,
            )(
                main_hidden_1x15360, pos_1d, swa_kv_in, swa_pos_in, cos_p, sin_p,
                main_proj_w, main_proj_s, main_norm_w,
                wkv0, wkv0_s, kvn0, wkv1, wkv1_s, kvn1, wkv2, wkv2_s, kvn2,
            )

        self._update_kv_only = _update_kv_only

        # 2. Full 3-stage `DSpark` draft step (`forward_embed` + `mtp.0..2` + `markov_head` + `confidence_head`)
        stage_in_specs = (
            P(), P(), P(), P(), P(), P(), P(), P(),
            P(), P(), P(), P(), P(), P(),
            P("tp", None), P("tp", None), P("tp"),
            P("tp", None), P("tp", None), P(None, "tp"), P(None, "tp"),
            P(), P(),
            P("tp", None), P("tp", None), P("tp", None), P("tp", None), P(None, "tp"), P(None, "tp"),
            P(None, "tp", None), P(None, "tp", None),
            P(None, "tp", None), P(None, "tp", None),
            P(None, None, "tp"), P(None, None, "tp"),
        )
        all_in_specs = (
            P(), P(), P(), P(), P(), P(), P(),
            P(None, "tp"), P("tp", None),
            P(None, "tp"), P(None, "tp"), P(),
            *stage_in_specs,
            *stage_in_specs,
            *stage_in_specs,
            P(), P(None, "tp"), P("tp", None), P(),
        )

        @functools.partial(
            jax.jit,
            out_shardings=(self.rep_s, self.rep_s, self.rep_s, self.rep_s),
        )
        def _draft_step_fn(*args):
            def _local(*largs):
                (
                    root_tok_1d,
                    main_hidden_1x15360,
                    start_pos_1d,
                    swa_kv_in,
                    swa_pos_in,
                    cos_p,
                    sin_p,
                    embed_w_part,
                    head_w_part,
                    main_proj_w,
                    main_proj_s,
                    main_norm_w,
                ) = largs[:12]
                s0_args = largs[12 : 12 + 35]
                s1_args = largs[12 + 35 : 12 + 70]
                s2_args = largs[12 + 70 : 12 + 105]
                mtp_norm_w, markov_emb_part, markov_head_part, conf_proj = largs[12 + 105 :]

                chip_idx = jax.lax.axis_index("tp")
                # forward_embed (`model.py:1128-1135`)
                if main_hidden_1x15360.ndim == 3:
                    dspark_h = main_hidden_1x15360
                    n_acc_val = start_pos_1d[0]
                    p_base = start_pos_1d[1]
                    mh_5 = jnp.concatenate(
                        [dspark_h[0, :5], dspark_h[1, :5], dspark_h[2, :5]], axis=-1
                    )  # [5, 15360]
                    mh_part = jax.lax.dynamic_slice_in_dim(mh_5, chip_idx * 960, 960, axis=-1)
                    main_x = rms_norm_f32(
                        _linear_bf16_or_fp8(mh_part, main_proj_w, main_proj_s, sharded_k=True),
                        main_norm_w,
                        cfg.rms_norm_eps,
                    )  # [5, 5120]
                    p0 = p_base + n_acc_val
                else:
                    mh_part = jax.lax.dynamic_slice_in_dim(
                        main_hidden_1x15360, chip_idx * 960, 960, axis=-1
                    )
                    main_x = rms_norm_f32(
                        _linear_bf16_or_fp8(mh_part, main_proj_w, main_proj_s, sharded_k=True),
                        main_norm_w,
                        cfg.rms_norm_eps,
                    )  # [1, 5120]
                    p0 = start_pos_1d[0]

                noise_ids = jnp.full((5,), cfg.dspark_noise_token_id, dtype=jnp.int32)
                draft_in_ids = noise_ids.at[0].set(root_tok_1d[0])  # [5]
                emb_part = embed_w_part[draft_in_ids]  # [5, 320]
                emb_full = jax.lax.all_gather(emb_part, "tp", axis=-1, tiled=True)  # [5, 5120]
                res = jnp.broadcast_to(emb_full[:, None, :], (5, 4, 5120))
                pre_mix = jnp.zeros((5, 4), dtype=jnp.float32).at[:, 0].set(1.0)

                draft_pos = p0 + 1 + jnp.arange(5, dtype=jnp.int32)  # [5]

                swa_kv_list = []
                swa_pos_out = swa_pos_in
                for s_idx, s_a in enumerate([s0_args, s1_args, s2_args]):
                    (
                        a_norm, f_norm, hca_fn, hca_sc, hca_b, hcf_fn, hcf_sc, hcf_b,
                        wqa, wqa_s, qn, wkv, wkv_s, kvn,
                        wqb, wqb_s, a_sink, woa, woa_s, wob, wob_s,
                        gw, gb, sw1, sw1_s, sw3, sw3_s, sw2, sw2_s,
                        rw1, rw1_s, rw3, rw3_s, rw2, rw2_s,
                    ) = s_a
                    post_mix, res_mix, x_in, attn_pre = mhc_pre_delayed_jax(
                        res, hca_fn, hca_sc, hca_b, pre_mix,
                        rms_norm_eps=cfg.rms_norm_eps,
                        hc_eps=cfg.hc_eps,
                        sinkhorn_iters=cfg.hc_sinkhorn_iters,
                    )
                    x_norm = rms_norm_f32(x_in, a_norm, cfg.rms_norm_eps)
                    attn_out, skv_s, swa_pos_out = _dspark_attn_local_jax(
                        x_norm, draft_pos, main_x, start_pos_1d,
                        swa_kv_in[s_idx], swa_pos_in, cos_p, sin_p,
                        wqa, wqa_s, qn, wkv, wkv_s, kvn,
                        wqb, wqb_s, a_sink, woa, woa_s, wob, wob_s,
                        cfg.rms_norm_eps,
                    )
                    swa_kv_list.append(skv_s)
                    res = mhc_post_jax(attn_out, res, post_mix, res_mix)

                    ffn_post, ffn_res, ffn_in, pre_mix = mhc_pre_delayed_jax(
                        res, hcf_fn, hcf_sc, hcf_b, attn_pre,
                        rms_norm_eps=cfg.rms_norm_eps,
                        hc_eps=cfg.hc_eps,
                        sinkhorn_iters=cfg.hc_sinkhorn_iters,
                    )
                    ffn_normed = rms_norm_f32(ffn_in, f_norm, cfg.rms_norm_eps)
                    ffn_out = _dspark_moe_local_jax(
                        ffn_normed, gw, gb,
                        sw1, sw1_s, sw3, sw3_s, sw2, sw2_s,
                        rw1, rw1_s, rw3, rw3_s, rw2, rw2_s,
                        routed_scaling_factor=cfg.routed_scaling_factor,
                        swiglu_limit=cfg.swiglu_limit,
                    )
                    res = mhc_post_jax(ffn_out, res, ffn_post, ffn_res)

                swa_kv_out = jnp.stack(swa_kv_list, axis=0)

                # forward_head (`model.py:1137-1156`)
                x_coll = jnp.sum(
                    pre_mix[:, :, None] * res.astype(jnp.float32), axis=1
                ).astype(jnp.bfloat16)  # [5, 5120]
                x_norm_final = rms_norm_f32(x_coll, mtp_norm_w, cfg.rms_norm_eps)  # [5, 5120]
                base_logits_local = jnp.dot(
                    x_norm_final.astype(jnp.bfloat16),
                    head_w_part.astype(jnp.bfloat16).T,
                    preferred_element_type=jnp.float32,
                    precision=jax.lax.Precision.DEFAULT,
                )  # [5, 8080]

                vocab_offset = chip_idx * 8080
                out_ids = jnp.zeros((6,), dtype=jnp.int32)
                out_ids = out_ids.at[0].set(root_tok_1d[0])
                conf_scores = jnp.zeros((5,), dtype=jnp.float32)

                for i in range(5):
                    prev_id = out_ids[i]
                    m_part = markov_emb_part[prev_id]  # [16] bf16
                    m_emb = jax.lax.all_gather(m_part, "tp", axis=-1, tiled=True)  # [256] bf16
                    m_bias_local = jnp.dot(
                        markov_head_part.astype(jnp.bfloat16),
                        m_emb.astype(jnp.bfloat16),
                        preferred_element_type=jnp.float32,
                        precision=jax.lax.Precision.DEFAULT,
                    )  # [8080]
                    tot_local = base_logits_local[i] + m_bias_local  # [8080]
                    loc_max = jnp.max(tot_local)
                    loc_idx = jnp.argmax(tot_local).astype(jnp.int32) + vocab_offset
                    all_max = jax.lax.all_gather(loc_max[None], "tp", axis=0, tiled=True)  # [16]
                    all_idx = jax.lax.all_gather(loc_idx[None], "tp", axis=0, tiled=True)  # [16]
                    win_chip = jnp.argmax(all_max)
                    next_id = all_idx[win_chip]
                    out_ids = out_ids.at[i + 1].set(next_id)

                    cat_feat = jnp.concatenate(
                        [x_coll[i].astype(jnp.float32), m_emb.astype(jnp.float32)], axis=-1
                    )  # [5376]
                    c_val = jnp.sum(cat_feat * conf_proj[0].astype(jnp.float32))
                    conf_scores = conf_scores.at[i].set(c_val)

                return out_ids[1:6], conf_scores, swa_kv_out, swa_pos_out

            return jax.shard_map(
                _local,
                mesh=mesh,
                in_specs=all_in_specs,
                out_specs=(P(), P(), P(), P()),
                check_vma=False,
            )(*args)

        self._draft_step_fn = _draft_step_fn

    def _stage_args(self, st: DSparkStageWeights) -> Tuple[jax.Array, ...]:
        return (
            st.attn_norm, st.ffn_norm,
            st.hc_attn_fn, st.hc_attn_scale, st.hc_attn_base,
            st.hc_ffn_fn, st.hc_ffn_scale, st.hc_ffn_base,
            st.wq_a, st.wq_a_scale, st.q_norm,
            st.wkv, st.wkv_scale, st.kv_norm,
            st.wq_b, st.wq_b_scale, st.attn_sink,
            st.wo_a, st.wo_a_scale, st.wo_b, st.wo_b_scale,
            st.gate_weight, st.gate_bias,
            st.shared_w1, st.shared_w1_scale,
            st.shared_w3, st.shared_w3_scale,
            st.shared_w2, st.shared_w2_scale,
            st.routed_w1, st.routed_w1_scale,
            st.routed_w3, st.routed_w3_scale,
            st.routed_w2, st.routed_w2_scale,
        )

    def update_kv_cache(
        self,
        main_hidden_np: np.ndarray,
        positions_np: np.ndarray,
        swa_kv: jax.Array,
        swa_pos: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Seed or advance `mtp.0..2` sliding-window KV cache for `main_hidden_np` [T, 15360] at `positions_np` [T]."""
        w = self.weights
        s0, s1, s2 = w.stages
        return self._update_kv_only(
            self.replicate(main_hidden_np.astype(ml_dtypes.bfloat16)),
            self.replicate(positions_np.astype(np.int32)),
            swa_kv,
            swa_pos,
            self.cos_plain,
            self.sin_plain,
            w.main_proj_weight,
            w.main_proj_scale,
            w.main_norm_weight,
            s0.wkv, s0.wkv_scale, s0.kv_norm,
            s1.wkv, s1.wkv_scale, s1.kv_norm,
            s2.wkv, s2.wkv_scale, s2.kv_norm,
        )

    def draft_block(
        self,
        root_token_id: int,
        main_hidden_1x15360: np.ndarray,
        start_pos: int,
        swa_kv: jax.Array,
        swa_pos: jax.Array,
    ) -> Tuple[np.ndarray, np.ndarray, jax.Array, jax.Array]:
        """Draft `5` tokens starting from `root_token_id` at `start_pos + 1 .. start_pos + 5`."""
        w = self.weights
        root_jax = self.replicate(np.array([root_token_id], dtype=np.int32))
        mh_jax = self.replicate(main_hidden_1x15360.astype(ml_dtypes.bfloat16))
        pos_jax = self.replicate(np.array([start_pos], dtype=np.int32))

        draft_ids_jax, conf_jax, swa_kv_out, swa_pos_out = self._draft_step_fn(
            root_jax,
            mh_jax,
            pos_jax,
            swa_kv,
            swa_pos,
            self.cos_plain,
            self.sin_plain,
            w.embed_weight,
            w.head_weight,
            w.main_proj_weight,
            w.main_proj_scale,
            w.main_norm_weight,
            *self._stage_args(w.stages[0]),
            *self._stage_args(w.stages[1]),
            *self._stage_args(w.stages[2]),
            w.mtp_norm_weight,
            w.markov_embed,
            w.markov_head,
            w.confidence_proj,
        )
        draft_ids_np = np.asarray(draft_ids_jax.addressable_shards[0].data, dtype=np.int32)
        conf_np = np.asarray(conf_jax.addressable_shards[0].data, dtype=np.float32)
        return draft_ids_np, conf_np, swa_kv_out, swa_pos_out

    def draft_block_on_device(
        self,
        root_token_id: int,
        dspark_h: jax.Array,
        prev_n_acc: int,
        prev_p0: int,
        swa_kv: jax.Array,
        swa_pos: jax.Array,
    ) -> Tuple[np.ndarray, jax.Array, jax.Array]:
        """Fused on-device KV update for `0 .. prev_n_acc` + 5-token draft from `dspark_h[:, prev_n_acc]`."""
        w = self.weights
        root_jax = self.replicate(np.array([root_token_id], dtype=np.int32))
        pos_jax = self.replicate(np.array([prev_n_acc, prev_p0], dtype=np.int32))

        draft_ids_jax, _, swa_kv_out, swa_pos_out = self._draft_step_fn(
            root_jax,
            dspark_h,
            pos_jax,
            swa_kv,
            swa_pos,
            self.cos_plain,
            self.sin_plain,
            w.embed_weight,
            w.head_weight,
            w.main_proj_weight,
            w.main_proj_scale,
            w.main_norm_weight,
            *self._stage_args(w.stages[0]),
            *self._stage_args(w.stages[1]),
            *self._stage_args(w.stages[2]),
            w.mtp_norm_weight,
            w.markov_embed,
            w.markov_head,
            w.confidence_proj,
        )
        draft_ids_np = np.asarray(
            draft_ids_jax.addressable_shards[0].data, dtype=np.int32
        )
        return draft_ids_np, swa_kv_out, swa_pos_out
