"""Real-checkpoint streaming loader and SHA-256 weight verifier for DeepSeek-V4.1-Flash on TPU v6e-16."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import ml_dtypes
import numpy as np

from .config import DSV41Config

BUCKET_DEFAULT = "dsv4-flash-jawadamin-asia-ne1"
PREFIX_DEFAULT = "deepseek-v4.1-flash"
CACHE_DIR_DEFAULT = "/dev/shm/dsv41_megakernel_cache_v1"

SAMPLED_TENSORS: Tuple[Tuple[str, str], ...] = (
    ("model-00002-of-00048.safetensors", "embed.weight"),
    ("model-00003-of-00048.safetensors", "layers.0.attn.wq_a.weight"),
    ("model-00003-of-00048.safetensors", "layers.0.ffn.experts.0.w1.weight"),
    ("model-00017-of-00048.safetensors", "layers.14.attn.compressor.wkv.weight"),
    ("model-00042-of-00048.safetensors", "layers.39.ffn.gate.weight"),
    ("model-00043-of-00048.safetensors", "norm.weight"),
)

FP4_E2M1_TABLE_NP = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def compute_and_verify_gcs_hashes(
    model_uri: str = f"gs://{BUCKET_DEFAULT}/{PREFIX_DEFAULT}",
    reference_hashes_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute SHA-256 of sampled tensors directly from GCS safetensors bytes and verify against reference."""
    from google.cloud import storage

    assert model_uri.startswith("gs://"), f"Expected gs:// URI, got {model_uri}"
    without_scheme = model_uri[len("gs://") :].rstrip("/")
    bucket_name, prefix = without_scheme.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    computed: Dict[str, Dict[str, Any]] = {}

    for shard, tname in SAMPLED_TENSORS:
        blob = bucket.blob(f"{prefix}/{shard}")
        hdr_len = struct.unpack("<Q", blob.download_as_bytes(start=0, end=7))[0]
        hdr = json.loads(blob.download_as_bytes(start=8, end=8 + hdr_len - 1).decode("utf-8"))
        info = hdr[tname]
        s, e = info["data_offsets"]
        data_start = 8 + hdr_len + s
        nbytes = e - s
        if nbytes <= 8 * 1024 * 1024:
            raw = blob.download_as_bytes(start=data_start, end=data_start + nbytes - 1)
            sha = hashlib.sha256(raw).hexdigest()
            mode = "full"
        else:
            head = blob.download_as_bytes(start=data_start, end=data_start + 4 * 1024 * 1024 - 1)
            tail = blob.download_as_bytes(start=data_start + nbytes - 1024 * 1024, end=data_start + nbytes - 1)
            sha = hashlib.sha256(head + tail).hexdigest()
            mode = "head4M_tail1M"
        computed[tname] = {
            "shard": shard,
            "dtype": info["dtype"],
            "shape": info["shape"],
            "nbytes": nbytes,
            "hash_mode": mode,
            "sha256": sha,
        }

    if reference_hashes_path and os.path.exists(reference_hashes_path):
        with open(reference_hashes_path, "r") as f:
            ref = json.load(f).get("tensors", {})
        for tname, meta in computed.items():
            if tname in ref:
                expected = ref[tname]["sha256"]
                assert meta["sha256"] == expected, (
                    f"SHA-256 mismatch on {tname}: got {meta['sha256']} != expected {expected}"
                )
    return computed


def ue8m0_to_f32_scale(scale_u8: jax.Array) -> jax.Array:
    """Convert UE8M0 uint8 exponent byte to exact IEEE-754 float32 power-of-two scale 2^(e - 127)."""
    return (scale_u8.astype(jnp.uint32) << jnp.uint32(23)).view(jnp.float32)


def dequant_fp8_block32_jax(weight_f8: jax.Array, scale_u8: jax.Array) -> jax.Array:
    """Dequantize F8_E4M3 [M, K] with block-(32, 32) UE8M0 scale [M//32, K//32] to BF16 [M, K]."""
    m, k = weight_f8.shape
    bm, bk = scale_u8.shape
    assert m == bm * 32 and k == bk * 32, f"Shape mismatch: weight={weight_f8.shape}, scale={scale_u8.shape}"
    w_f32 = weight_f8.astype(jnp.float32).reshape(bm, 32, bk, 32)
    s_f32 = ue8m0_to_f32_scale(scale_u8)[:, None, :, None]
    return (w_f32 * s_f32).reshape(m, k).astype(jnp.bfloat16)


def dequant_mxfp4_block32_jax(packed_i8: jax.Array, scale_u8: jax.Array) -> jax.Array:
    """Dequantize packed MXFP4 int8 [..., M, K//2] + UE8M0 uint8 [..., M, K//32] to BF16 [..., M, K]."""
    u8 = packed_i8.view(jnp.uint8)
    low = u8 & jnp.uint8(0x0F)
    high = (u8 >> jnp.uint8(4)) & jnp.uint8(0x0F)
    wide = (
        jnp.stack([low, high], axis=-1)
        .reshape(*packed_i8.shape[:-1], packed_i8.shape[-1] * 2)
        .astype(jnp.int32)
    )
    index = jax.lax.bitwise_and(wide, 7)
    sign_bit = jax.lax.bitwise_and(wide, 8) << 28
    f32_bits = (
        jnp.where(
            index <= 1,
            (-index) & 0x3F000000,
            0x3F000000 + (index << 22),
        )
        | sign_bit
    )
    vals = jax.lax.bitcast_convert_type(f32_bits, jnp.float32)
    *lead, m, k = vals.shape
    num_blocks = k // 32
    vals_blk = vals.reshape(*lead, m, num_blocks, 32)
    s_f32 = ue8m0_to_f32_scale(scale_u8)[..., None]
    return (vals_blk * s_f32).reshape(*lead, m, k).astype(jnp.bfloat16)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _find_next_prime(start: int, seen: set) -> int:
    c = start + 1
    while not _is_prime(c) or c in seen:
        c += 1
    return c


def build_compressed_token_map(tokenizer_dir: str) -> Tuple[np.ndarray, int]:
    """Build the 129280 -> 99092 normalized token map matching vllm/models/deepseek_v4_1/common/engram.py."""
    from transformers import AutoTokenizer
    from tokenizers import Regex, normalizers

    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tok.backend_tokenizer
    key_to_new: Dict[str, int] = {}
    lookup = np.zeros(len(tok), dtype=np.int64)
    for token_id in range(len(tok)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


class EngramHostTables:
    """Host-sharded Engram hash tables (6 hash columns per host across 4 hosts = 24 columns)."""

    def __init__(
        self,
        cfg: DSV41Config,
        token_map: np.ndarray,
        host_index: int,
        num_hosts: int = 4,
    ) -> None:
        self.cfg = cfg
        self.token_map = token_map.astype(np.int64)
        self.pad_id = int(self.token_map[cfg.engram_pad_token_id])
        self.host_index = host_index
        self.num_hosts = num_hosts

        # Multipliers: [2, 4]
        max_long = np.iinfo(np.int64).max
        multiplier_bound = max(1, (max_long // cfg.engram_compressed_vocab_size) // 2)
        mult_rows = []
        for lid in cfg.engram_layer_ids:
            gen = np.random.default_rng(10007 * lid)
            vals = gen.integers(low=0, high=multiplier_bound, size=(cfg.engram_max_ngram_size,), dtype=np.int64)
            mult_rows.append(vals * 2 + 1)
        self.multipliers = np.stack(mult_rows, axis=0)  # [2, 4]

        # Primes & offsets: [2, 24]
        seen: set = set()
        primes_list = []
        for _ in cfg.engram_layer_ids:
            layer_p = []
            for _ in range(cfg.engram_max_ngram_size - 1):
                cur = cfg.engram_vocab_size - 1
                for _ in range(cfg.engram_n_heads):
                    cur = _find_next_prime(cur, seen)
                    seen.add(cur)
                    layer_p.append(cur)
            primes_list.append(layer_p)
        self.flat_primes = np.array(primes_list, dtype=np.int64)  # [2, 24]
        self.flat_offsets = np.zeros_like(self.flat_primes)
        for i in range(len(cfg.engram_layer_ids)):
            self.flat_offsets[i, 1:] = np.cumsum(self.flat_primes[i, :-1])

        self.n_hash_cols = self.flat_primes.shape[1]  # 24
        assert self.n_hash_cols % num_hosts == 0
        self.cols_per_host = self.n_hash_cols // num_hosts  # 6
        self.col_start = host_index * self.cols_per_host
        self.col_end = self.col_start + self.cols_per_host

        self.vocab_start: List[int] = []
        self.vocab_end: List[int] = []
        for i in range(len(cfg.engram_layer_ids)):
            s = int(self.flat_offsets[i, self.col_start])
            e = int(
                self.flat_offsets[i, self.col_end - 1] + self.flat_primes[i, self.col_end - 1]
            )
            self.vocab_start.append(s)
            self.vocab_end.append(e)

        self.weight_mmaps: Dict[int, np.ndarray] = {}
        self.scale_mmaps: Dict[int, np.ndarray] = {}
        self._fp8_lut = (
            np.arange(256, dtype=np.uint8)
            .view(ml_dtypes.float8_e4m3fn)
            .astype(np.float32)
        )
        self._local_primes = self.flat_primes[:, self.col_start : self.col_end].copy()
        self._local_offsets = (
            self.flat_offsets[:, self.col_start : self.col_end]
            - np.array(self.vocab_start, dtype=np.int64)[:, None]
        ).copy()

    def compute_hashes_for_windows(self, windows_4: np.ndarray) -> np.ndarray:
        """Compute [T, 2, 24] int64 hash IDs from [T, 4] token windows [tok_t, tok_{t-1}, tok_{t-2}, tok_{t-3}].

        Entries < 0 in windows_4 indicate positions before sequence start (lookback < 0).
        """
        t_count = windows_4.shape[0]
        num_layers = len(self.cfg.engram_layer_ids)
        num_heads = self.cfg.engram_n_heads
        out = np.empty((t_count, num_layers, self.n_hash_cols), dtype=np.int64)
        blocked = np.zeros(t_count, dtype=bool)
        rolling = np.zeros((t_count, num_layers), dtype=np.int64)

        for shift in range(self.cfg.engram_max_ngram_size):
            tok = windows_4[:, shift]
            blocked = blocked | (tok < 0)
            safe_tok = np.clip(tok, 0, len(self.token_map) - 1)
            mapped = np.where(blocked, self.pad_id, self.token_map[safe_tok])
            rolling = rolling ^ (mapped[:, None] * self.multipliers[None, :, shift])
            if shift > 0:
                c0 = (shift - 1) * num_heads
                c1 = shift * num_heads
                hashed = (rolling[:, :, None] % self.flat_primes[None, :, c0:c1]) + self.flat_offsets[None, :, c0:c1]
                out[:, :, c0:c1] = hashed
        return out

    def gather_local_cols(self, hash_ids: np.ndarray, layer_hash_index: int) -> np.ndarray:
        """Gather & dequantize this host's 6 columns for `layer_hash_index` (0 -> L1, 1 -> L14).

        Args:
            hash_ids: [T, 24] or [T, 2, 24] int64 hash indices.
        Returns:
            local_rows: [T, 6, 256] bfloat16 numpy array.
        """
        if hash_ids.ndim == 3:
            h_layer = hash_ids[:, layer_hash_index, :]
        else:
            h_layer = hash_ids
        cols = h_layer[:, self.col_start : self.col_end] - self.vocab_start[layer_hash_index]
        w_u8 = self.weight_mmaps[layer_hash_index][cols]  # [T, 6, 256] uint8
        s_u8 = self.scale_mmaps[layer_hash_index][cols]   # [T, 6, 8] uint8
        w_f8 = self._fp8_lut[w_u8]
        s_f32 = (s_u8.astype(np.uint32) << np.uint32(23)).view(np.float32)
        vals = (w_f8.reshape(-1, self.cols_per_host, 8, 32) * s_f32[..., None]).reshape(
            -1, self.cols_per_host, 256
        )
        u32 = vals.view(np.uint32)
        return ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16).view(ml_dtypes.bfloat16)

    def fast_gather_both_chips(self, windows_4: np.ndarray, b_tile: int = 8) -> np.ndarray:
        """Vectorized local 6-column hash + mmap gather + FP8->BF16 dequant for both L1 and L14 (~170 us).

        Args:
            windows_4: [T, 4] int64 token windows (`T <= b_tile`).
            b_tile: Pallas megakernel batch tile (`8`).
        Returns:
            chips_bf16: [4, 1, 2, b_tile, 384] bfloat16 numpy array ready for per-chip `device_put`.
        """
        t_count = windows_4.shape[0]
        blocked = np.maximum.accumulate(windows_4 < 0, axis=1)
        safe_tok = np.clip(windows_4, 0, len(self.token_map) - 1)
        mapped = np.where(blocked, self.pad_id, self.token_map[safe_tok])
        terms = mapped[:, None, :] * self.multipliers[None, :, :]
        r0 = terms[:, :, 0]
        r1 = r0 ^ terms[:, :, 1]
        r2 = r1 ^ terms[:, :, 2]
        r3 = r2 ^ terms[:, :, 3]
        rolls = np.stack([r1, r2, r3], axis=2)
        roll_6 = np.repeat(rolls, 8, axis=2)[:, :, self.col_start : self.col_end]
        cols = (roll_6 % self._local_primes[None, :, :]) + self._local_offsets[None, :, :]

        out = np.zeros((4, 1, 2, b_tile, 384), dtype=ml_dtypes.bfloat16)
        for idx_e in range(2):
            if idx_e in self.weight_mmaps:
                c_e = cols[:, idx_e, :]
                wf = self._fp8_lut[self.weight_mmaps[idx_e][c_e]]
                sf = (self.scale_mmaps[idx_e][c_e].astype(np.uint32) << np.uint32(23)).view(np.float32)
                v32 = (wf.reshape(t_count, 6, 8, 32) * sf[..., None]).reshape(t_count, 4, 384)
                u32 = v32.view(np.uint32)
                v_bf16 = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16).view(ml_dtypes.bfloat16)
                out[:, 0, idx_e, :t_count, :] = np.transpose(v_bf16, (1, 0, 2))
        return out


@dataclass
class LayerWeights:
    """Per-layer weights in 16-chip TP=16 sharded HBM layout."""

    layer_id: int
    compress_ratio: int
    is_kv_source: bool
    is_index_source: bool
    has_engram: bool

    # Replicated norms & mHC parameters
    attn_norm: jax.Array          # BF16 [5120]
    ffn_norm: jax.Array           # BF16 [5120]
    hc_attn_fn: jax.Array         # F32 [24, 20480]
    hc_attn_scale: jax.Array      # F32 [3]
    hc_attn_base: jax.Array       # F32 [24]
    hc_ffn_fn: jax.Array          # F32 [24, 20480]
    hc_ffn_scale: jax.Array       # F32 [3]
    hc_ffn_base: jax.Array        # F32 [24]

    # Attention input projections (replicated for zero-collective q_norm/kv_norm)
    wq_a: jax.Array               # F8_E4M3 [1280, 5120]
    wq_a_scale: jax.Array         # U8 [40, 160]
    q_norm: jax.Array             # BF16 [1280]
    wkv: jax.Array                # F8_E4M3 [512, 5120]
    wkv_scale: jax.Array          # U8 [16, 160]
    kv_norm: jax.Array            # BF16 [512]

    # Attention head & output projections (TP=16 group-paired layout: g = c // 2)
    wq_b: jax.Array               # F8_E4M3 [65536, 1280] P('tp', None) -> [4096, 1280]/chip (8 heads of group g)
    wq_b_scale: jax.Array         # U8 [2048, 40] P('tp', None) -> [128, 40]/chip
    attn_sink: jax.Array          # F32 [128] P('tp') -> [8]/chip (8 heads of group g)
    wo_a: jax.Array               # F8_E4M3 [8192, 4096] P('tp', None) -> [512, 4096]/chip
    wo_a_scale: jax.Array         # U8 [256, 128] P('tp', None) -> [16, 128]/chip
    wo_b: jax.Array               # F8_E4M3 [5120, 8192] P(None, 'tp') -> [5120, 512]/chip
    wo_b_scale: jax.Array         # U8 [160, 256] P(None, 'tp') -> [160, 16]/chip

    # Router + Shared Expert + 384 Routed Experts (TP=16 zero-skew sharded: 160 intermediate/chip)
    gate_weight: jax.Array        # BF16 [384, 5120] (replicated)
    gate_bias: jax.Array          # F32 [384] (replicated)
    shared_w1: jax.Array          # F8_E4M3 [2560, 5120] P('tp', None) -> [160, 5120]/chip
    shared_w1_scale: jax.Array    # U8 [80, 160] P('tp', None) -> [5, 160]/chip
    shared_w3: jax.Array          # F8_E4M3 [2560, 5120] P('tp', None) -> [160, 5120]/chip
    shared_w3_scale: jax.Array    # U8 [80, 160] P('tp', None) -> [5, 160]/chip
    shared_w2: jax.Array          # F8_E4M3 [5120, 2560] P(None, 'tp') -> [5120, 160]/chip
    shared_w2_scale: jax.Array    # U8 [160, 80] P(None, 'tp') -> [160, 5]/chip

    routed_w1: jax.Array          # I8 [384, 2560, 2560] P(None, 'tp', None) -> [384, 160, 2560]/chip
    routed_w1_scale: jax.Array    # U8 [384, 2560, 160] P(None, 'tp', None) -> [384, 160, 160]/chip
    routed_w3: jax.Array          # I8 [384, 2560, 2560] P(None, 'tp', None) -> [384, 160, 2560]/chip
    routed_w3_scale: jax.Array    # U8 [384, 2560, 160] P(None, 'tp', None) -> [384, 160, 160]/chip
    routed_w2: jax.Array          # I8 [384, 5120, 1280] P(None, None, 'tp') -> [384, 5120, 80]/chip
    routed_w2_scale: jax.Array    # U8 [384, 5120, 80] P(None, None, 'tp') -> [384, 5120, 5]/chip

    # Optional Compressor (kv_source layers: [2, 8, 14, 20])
    comp_wkv: Optional[jax.Array] = None     # BF16 [512, 5120]
    comp_wgate: Optional[jax.Array] = None   # BF16 [512, 5120] (layers 2, 8, 14 only)
    comp_norm: Optional[jax.Array] = None    # BF16 [512]

    # Optional Indexer (index_source layers: [2, 8, 14, 20, 24, 28, 32, 36])
    idx_wq_b: Optional[jax.Array] = None          # F8_E4M3 [4096, 1280]
    idx_wq_b_scale: Optional[jax.Array] = None    # U8 [128, 40]
    idx_weights_proj: Optional[jax.Array] = None  # BF16 [32, 5120]
    idx_wk: Optional[jax.Array] = None            # BF16 [128, 512] (kv_source layers only)
    idx_k_norm: Optional[jax.Array] = None        # BF16 [128] (kv_source layers only)

    # Optional Engram TPU projections (layers 1 & 14; large table stays in host DRAM)
    engram_wkv: Optional[jax.Array] = None        # F8_E4M3 [25600, 6144]
    engram_wkv_scale: Optional[jax.Array] = None  # U8 [800, 192]
    engram_q_weight: Optional[jax.Array] = None   # BF16 [4, 5120]
    engram_k_weight: Optional[jax.Array] = None   # BF16 [4, 5120]


@dataclass
class ModelWeights:
    """Full 40-layer DeepSeek-V4.1-Flash model weights on 16 TPU v6e chips + host-sharded Engram tables."""

    config: DSV41Config
    mesh: Mesh
    embed_weight: jax.Array       # BF16 [129280, 5120] P(None, 'tp') -> [129280, 320]/chip
    norm_weight: jax.Array        # BF16 [5120] P()
    head_weight: jax.Array        # BF16 [129280, 5120] P('tp', None) -> [8080, 5120]/chip
    layers: Tuple[LayerWeights, ...]
    engram_host: EngramHostTables
    gcs_hashes: Dict[str, Dict[str, Any]]


def _torch_to_numpy(t: Any) -> np.ndarray:
    import torch

    if t.dtype == torch.float32:
        return t.numpy().copy()
    if t.dtype == torch.int8:
        return t.numpy().copy()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint8).numpy().copy().view(ml_dtypes.bfloat16).reshape(tuple(t.shape))
    if t.dtype == torch.float8_e4m3fn:
        return t.view(torch.uint8).numpy().copy().view(ml_dtypes.float8_e4m3fn).reshape(tuple(t.shape))
    if t.dtype == torch.float8_e8m0fnu:
        return t.view(torch.uint8).numpy().copy().reshape(tuple(t.shape))
    raise ValueError(f"Unsupported torch dtype: {t.dtype}")


def _stream_safetensors_dict(shard_uri: str) -> Dict[str, np.ndarray]:
    from runai_model_streamer import SafetensorsStreamer

    out: Dict[str, np.ndarray] = {}
    with SafetensorsStreamer() as streamer:
        streamer.stream_files([shard_uri])
        for name, t in streamer.get_tensors():
            out[name] = _torch_to_numpy(t)
    return out


def _make_sharded_array(
    mesh: Mesh,
    local_devices: List[jax.Device],
    global_shape: Tuple[int, ...],
    spec: P,
    per_chip_arrays: List[np.ndarray],
) -> jax.Array:
    sharding = NamedSharding(mesh, spec)
    device_buffers = [jax.device_put(arr, dev) for arr, dev in zip(per_chip_arrays, local_devices)]
    return jax.make_array_from_single_device_arrays(global_shape, sharding, device_buffers)


def _make_replicated_array(
    mesh: Mesh,
    local_devices: List[jax.Device],
    arr: np.ndarray,
) -> jax.Array:
    sharding = NamedSharding(mesh, P())
    device_buffers = [jax.device_put(arr, dev) for dev in local_devices]
    return jax.make_array_from_single_device_arrays(arr.shape, sharding, device_buffers)


def _ensure_tokenizer_and_config(model_uri: str, cache_dir: str) -> str:
    from google.cloud import storage

    tok_dir = os.path.join(cache_dir, "tokenizer")
    sentinel = os.path.join(tok_dir, ".complete")
    if os.path.exists(sentinel):
        return tok_dir

    os.makedirs(tok_dir, exist_ok=True)
    without_scheme = model_uri[len("gs://") :].rstrip("/")
    bucket_name, prefix = without_scheme.split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for blob in bucket.list_blobs(prefix=f"{prefix}/"):
        fname = blob.name.split("/")[-1]
        if fname and not fname.endswith(".safetensors"):
            dst = os.path.join(tok_dir, fname)
            if not os.path.exists(dst):
                blob.download_to_filename(dst)
    with open(sentinel, "w") as f:
        f.write("ok\n")
    return tok_dir


def _load_or_stream_engram_shard(
    model_uri: str,
    shard_name: str,
    layer_id: int,
    layer_hash_index: int,
    engram_host: EngramHostTables,
    cache_dir: str,
) -> Dict[str, np.ndarray]:
    """Stream small TPU Engram tensors + this host's 6-column slice of the 98.3 GB Engram table."""
    from google.cloud import storage
    from runai_model_streamer.safetensors_streamer.safetensors_streamer import (
        DistributedStreamer,
        FileChunks,
    )

    h = engram_host.host_index
    w_bin = os.path.join(cache_dir, f"engram_L{layer_id}_host{h}_weight.bin")
    s_bin = os.path.join(cache_dir, f"engram_L{layer_id}_host{h}_scale.bin")
    tpu_npz = os.path.join(cache_dir, f"engram_L{layer_id}_tpu.npz")

    v_start = engram_host.vocab_start[layer_hash_index]
    v_end = engram_host.vocab_end[layer_hash_index]
    part_rows = v_end - v_start

    if not (os.path.exists(w_bin) and os.path.exists(s_bin) and os.path.exists(tpu_npz)):
        without_scheme = model_uri[len("gs://") :].rstrip("/")
        bucket_name, prefix = without_scheme.split("/", 1)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(f"{prefix}/{shard_name}")
        hdr_len = struct.unpack("<Q", blob.download_as_bytes(start=0, end=7))[0]
        hdr = json.loads(blob.download_as_bytes(start=8, end=8 + hdr_len - 1).decode("utf-8"))
        base_off = 8 + hdr_len
        p_prefix = f"layers.{layer_id}.engram"
        shard_uri = f"{model_uri.rstrip('/')}/{shard_name}"

        if not os.path.exists(tpu_npz):
            small_names = [
                f"{p_prefix}.wkv.weight",
                f"{p_prefix}.wkv.scale",
                f"{p_prefix}.q_weight",
                f"{p_prefix}.k_weight",
            ]
            tpu_dict: Dict[str, np.ndarray] = {}
            with DistributedStreamer() as ds:
                chunks = []
                for idx_t, tname in enumerate(small_names):
                    info = hdr[tname]
                    s, e = info["data_offsets"]
                    chunks.append(FileChunks(idx_t, shard_uri, base_off + s, [e - s]))
                ds.stream_files(chunks, credentials=None, device="cpu", is_distributed=False)
                for fi, ci, buf in ds.get_chunks():
                    tname = small_names[fi]
                    info = hdr[tname]
                    raw_u8 = buf.numpy().reshape(-1)
                    if info["dtype"] == "F8_E4M3":
                        arr = raw_u8.copy().view(ml_dtypes.float8_e4m3fn).reshape(info["shape"])
                    elif info["dtype"] == "F8_E8M0":
                        arr = raw_u8.copy().reshape(info["shape"])
                    elif info["dtype"] == "BF16":
                        arr = raw_u8.copy().view(ml_dtypes.bfloat16).reshape(info["shape"])
                    else:
                        raise ValueError(f"Unexpected dtype {info['dtype']}")
                    tpu_dict[tname] = arr
            _save_layer_cache(tpu_npz, tpu_dict)

        if not (os.path.exists(w_bin) and os.path.exists(s_bin)):
            w_info = hdr[f"{p_prefix}.embed.weight"]
            s_info = hdr[f"{p_prefix}.embed.scale"]
            w_start = base_off + w_info["data_offsets"][0] + v_start * 256
            w_nbytes = part_rows * 256
            s_start = base_off + s_info["data_offsets"][0] + v_start * 8
            s_nbytes = part_rows * 8

            chunk_bytes = 2 * 1024 * 1024 * 1024
            w_sizes = []
            rem = w_nbytes
            while rem > 0:
                c_sz = min(rem, chunk_bytes)
                w_sizes.append(c_sz)
                rem -= c_sz

            w_mmap_out = np.memmap(w_bin + ".tmp", dtype=np.uint8, mode="w+", shape=(part_rows, 256))
            s_mmap_out = np.memmap(s_bin + ".tmp", dtype=np.uint8, mode="w+", shape=(part_rows, 8))
            w_offsets = [0] + list(np.cumsum(w_sizes[:-1]))

            with DistributedStreamer() as ds:
                req = [
                    FileChunks(0, shard_uri, w_start, w_sizes),
                    FileChunks(1, shard_uri, s_start, [s_nbytes]),
                ]
                ds.stream_files(req, credentials=None, device="cpu", is_distributed=False)
                for fi, ci, buf in ds.get_chunks():
                    if fi == 0:
                        byte_off = w_offsets[ci]
                        row_off = byte_off // 256
                        n_rows = w_sizes[ci] // 256
                        w_mmap_out[row_off : row_off + n_rows] = buf.numpy().reshape(n_rows, 256)
                    else:
                        s_mmap_out[:] = buf.numpy().reshape(part_rows, 8)

            w_mmap_out.flush()
            s_mmap_out.flush()
            del w_mmap_out
            del s_mmap_out
            os.replace(w_bin + ".tmp", w_bin)
            os.replace(s_bin + ".tmp", s_bin)

    engram_host.weight_mmaps[layer_hash_index] = np.memmap(
        w_bin, dtype=np.uint8, mode="r", shape=(part_rows, 256)
    )
    engram_host.scale_mmaps[layer_hash_index] = np.memmap(
        s_bin, dtype=np.uint8, mode="r", shape=(part_rows, 8)
    )
    return _load_layer_cache(tpu_npz)


def _pack_layer_host_slice(
    layer_id: int,
    cfg: DSV41Config,
    raw: Dict[str, np.ndarray],
    host_index: int,
) -> Dict[str, np.ndarray]:
    """Slice a single layer's raw safetensors dict into the 4 local TP ranks [4*h : 4*h + 4] for host h."""
    p = f"layers.{layer_id}"
    c_start = host_index * 4
    c_end = c_start + 4
    packed: Dict[str, np.ndarray] = {}

    # Replicated tensors
    for k_short, k_full in [
        ("attn_norm", f"{p}.attn_norm.weight"),
        ("ffn_norm", f"{p}.ffn_norm.weight"),
        ("hc_attn_fn", f"{p}.hc_attn_fn"),
        ("hc_attn_scale", f"{p}.hc_attn_scale"),
        ("hc_attn_base", f"{p}.hc_attn_base"),
        ("hc_ffn_fn", f"{p}.hc_ffn_fn"),
        ("hc_ffn_scale", f"{p}.hc_ffn_scale"),
        ("hc_ffn_base", f"{p}.hc_ffn_base"),
        ("wq_a", f"{p}.attn.wq_a.weight"),
        ("wq_a_scale", f"{p}.attn.wq_a.scale"),
        ("q_norm", f"{p}.attn.q_norm.weight"),
        ("wkv", f"{p}.attn.wkv.weight"),
        ("wkv_scale", f"{p}.attn.wkv.scale"),
        ("kv_norm", f"{p}.attn.kv_norm.weight"),
        ("gate_weight", f"{p}.ffn.gate.weight"),
        ("gate_bias", f"{p}.ffn.gate.bias"),
    ]:
        packed[k_short] = raw[k_full]

    # Group-paired wq_b & attn_sink: chip c gets group g = c // 2
    wq_b_full = raw[f"{p}.attn.wq_b.weight"].reshape(8, 4096, 1280)
    wq_b_s_full = raw[f"{p}.attn.wq_b.scale"].reshape(8, 128, 40)
    sink_full = raw[f"{p}.attn.attn_sink"].reshape(8, 8)

    wq_b_host = np.stack([wq_b_full[c // 2] for c in range(c_start, c_end)], axis=0)      # [4, 4096, 1280]
    wq_b_s_host = np.stack([wq_b_s_full[c // 2] for c in range(c_start, c_end)], axis=0)  # [4, 128, 40]
    sink_host = np.stack([sink_full[c // 2] for c in range(c_start, c_end)], axis=0)      # [4, 8]
    packed["wq_b"] = wq_b_host
    packed["wq_b_scale"] = wq_b_s_host
    packed["attn_sink"] = sink_host

    # Sharded wo_a [8192, 4096] -> [4, 512, 4096] and wo_b [5120, 8192] -> [4, 5120, 512]
    wo_a_full = raw[f"{p}.attn.wo_a.weight"].reshape(16, 512, 4096)
    wo_a_s_full = raw[f"{p}.attn.wo_a.scale"].reshape(16, 16, 128)
    wo_b_full = raw[f"{p}.attn.wo_b.weight"].reshape(5120, 16, 512)
    wo_b_s_full = raw[f"{p}.attn.wo_b.scale"].reshape(160, 16, 16)

    packed["wo_a"] = wo_a_full[c_start:c_end]
    packed["wo_a_scale"] = wo_a_s_full[c_start:c_end]
    packed["wo_b"] = np.transpose(wo_b_full[:, c_start:c_end, :], (1, 0, 2)).copy()
    packed["wo_b_scale"] = np.transpose(wo_b_s_full[:, c_start:c_end, :], (1, 0, 2)).copy()

    # Zero-pad shared expert from 2304 -> 2560 and slice [c_start:c_end] (160 rows/cols per chip)
    def pad_shared_w13(w: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w_pad = np.zeros((2560, 5120), dtype=w.dtype)
        s_pad = np.zeros((80, 160), dtype=s.dtype)
        w_pad[:2304] = w
        s_pad[:72] = s
        return w_pad.reshape(16, 160, 5120)[c_start:c_end].copy(), s_pad.reshape(16, 5, 160)[c_start:c_end].copy()

    packed["shared_w1"], packed["shared_w1_scale"] = pad_shared_w13(
        raw[f"{p}.ffn.shared_experts.w1.weight"], raw[f"{p}.ffn.shared_experts.w1.scale"]
    )
    packed["shared_w3"], packed["shared_w3_scale"] = pad_shared_w13(
        raw[f"{p}.ffn.shared_experts.w3.weight"], raw[f"{p}.ffn.shared_experts.w3.scale"]
    )
    sw2 = np.zeros((5120, 2560), dtype=raw[f"{p}.ffn.shared_experts.w2.weight"].dtype)
    sw2_s = np.zeros((160, 80), dtype=raw[f"{p}.ffn.shared_experts.w2.scale"].dtype)
    sw2[:, :2304] = raw[f"{p}.ffn.shared_experts.w2.weight"]
    sw2_s[:, :72] = raw[f"{p}.ffn.shared_experts.w2.scale"]
    packed["shared_w2"] = np.transpose(sw2.reshape(5120, 16, 160)[:, c_start:c_end, :], (1, 0, 2)).copy()
    packed["shared_w2_scale"] = np.transpose(sw2_s.reshape(160, 16, 5)[:, c_start:c_end, :], (1, 0, 2)).copy()

    # 384 Routed Experts: pad 2304 -> 2560 and slice ONLY this host's [c_start*160 : c_end*160]
    row_lo = c_start * 160
    row_hi = c_end * 160
    rw1_host = np.zeros((4, 384, 160, 2560), dtype=np.int8)
    rw1_s_host = np.zeros((4, 384, 160, 160), dtype=np.uint8)
    rw3_host = np.zeros((4, 384, 160, 2560), dtype=np.int8)
    rw3_s_host = np.zeros((4, 384, 160, 160), dtype=np.uint8)
    rw2_host = np.zeros((4, 384, 5120, 80), dtype=np.int8)
    rw2_s_host = np.zeros((4, 384, 5120, 5), dtype=np.uint8)

    for e in range(cfg.n_routed_experts):
        ep = f"{p}.ffn.experts.{e}"
        w1_e = raw[f"{ep}.w1.weight"]
        s1_e = raw[f"{ep}.w1.scale"]
        w3_e = raw[f"{ep}.w3.weight"]
        s3_e = raw[f"{ep}.w3.scale"]
        w2_e = raw[f"{ep}.w2.weight"]
        s2_e = raw[f"{ep}.w2.scale"]

        for local_c in range(4):
            chip_id = c_start + local_c
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

    # Optional Compressor
    if layer_id in cfg.kv_source_layer_ids:
        packed["comp_wkv"] = raw[f"{p}.attn.compressor.wkv.weight"]
        packed["comp_norm"] = raw[f"{p}.attn.compressor.norm.weight"]
        if cfg.compress_ratios[layer_id] == 2:
            packed["comp_wgate"] = raw[f"{p}.attn.compressor.wgate.weight"]

    # Optional Indexer
    if layer_id in cfg.index_source_layer_ids:
        packed["idx_wq_b"] = raw[f"{p}.attn.indexer.wq_b.weight"]
        packed["idx_wq_b_scale"] = raw[f"{p}.attn.indexer.wq_b.scale"]
        packed["idx_weights_proj"] = raw[f"{p}.attn.indexer.weights_proj.weight"]
        if layer_id in cfg.kv_source_layer_ids:
            packed["idx_wk"] = raw[f"{p}.attn.indexer.wk.weight"]
            packed["idx_k_norm"] = raw[f"{p}.attn.indexer.k_norm.weight"]

    return packed


def _save_layer_cache(cache_file: str, packed: Dict[str, np.ndarray]) -> None:
    # Store bfloat16/float8 arrays as raw uint8 with metadata so np.load is instant and lossless
    save_dict: Dict[str, Any] = {}
    dtypes: Dict[str, str] = {}
    for k, arr in packed.items():
        if arr.dtype == ml_dtypes.bfloat16:
            save_dict[k] = arr.view(np.uint8)
            dtypes[k] = "bf16"
        elif arr.dtype == ml_dtypes.float8_e4m3fn:
            save_dict[k] = arr.view(np.uint8)
            dtypes[k] = "f8_e4m3"
        else:
            save_dict[k] = arr
            dtypes[k] = str(arr.dtype)
    save_dict["__dtypes__"] = np.array(json.dumps(dtypes))
    tmp_file = cache_file + ".tmp.npz"
    np.savez(tmp_file, **save_dict)
    os.replace(tmp_file, cache_file)


def _load_layer_cache(cache_file: str) -> Dict[str, np.ndarray]:
    npz = np.load(cache_file)
    dtypes = json.loads(str(npz["__dtypes__"]))
    out: Dict[str, np.ndarray] = {}
    for k, dt in dtypes.items():
        arr = npz[k]
        if dt == "bf16":
            out[k] = arr.view(ml_dtypes.bfloat16)
        elif dt == "f8_e4m3":
            out[k] = arr.view(ml_dtypes.float8_e4m3fn)
        else:
            out[k] = arr
    return out


def load_dsv41_weights(
    model_uri: str = f"gs://{BUCKET_DEFAULT}/{PREFIX_DEFAULT}",
    cache_dir: str = CACHE_DIR_DEFAULT,
    reference_hashes_path: Optional[str] = None,
    load_engram_tables: bool = True,
) -> ModelWeights:
    """Stream and shard the full DeepSeek-V4.1-Flash checkpoint onto 16 TPU v6e devices."""
    t0 = time.time()
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Verify sampled tensor SHA-256 hashes directly against GCS
    gcs_hashes = compute_and_verify_gcs_hashes(model_uri, reference_hashes_path)

    # 2. Download tokenizer & config.json and build EngramHostTables
    tok_dir = _ensure_tokenizer_and_config(model_uri, cache_dir)
    cfg = DSV41Config.from_json(os.path.join(tok_dir, "config.json"))
    tmap_cache = os.path.join(cache_dir, "compressed_token_map.npy")
    if os.path.exists(tmap_cache):
        token_map = np.load(tmap_cache)
    else:
        token_map, csize = build_compressed_token_map(tok_dir)
        assert csize == cfg.engram_compressed_vocab_size, (
            f"Compressed vocab size mismatch: {csize} != {cfg.engram_compressed_vocab_size}"
        )
        np.save(tmap_cache, token_map)

    host_index = jax.process_index()
    num_hosts = jax.process_count()
    devices_by_process = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
    mesh = Mesh(np.array(devices_by_process), ("tp",))
    local_devices = devices_by_process[host_index * 4 : (host_index + 1) * 4]
    c_start = host_index * 4
    c_end = c_start + 4

    engram_host = EngramHostTables(cfg, token_map, host_index=host_index, num_hosts=num_hosts)

    # 3. Load Engram shards 47 (layer 1) and 48 (layer 14)
    engram_tpu_tensors: Dict[int, Dict[str, np.ndarray]] = {}
    if load_engram_tables:
        for idx_e, (lid, s_num) in enumerate([(1, 47), (14, 48)]):
            shard_name = f"model-{s_num:05d}-of-00048.safetensors"
            engram_tpu_tensors[lid] = _load_or_stream_engram_shard(
                model_uri, shard_name, lid, idx_e, engram_host, cache_dir
            )

    # 4. Load embed.weight (shard 2) and norm.weight + head.weight (shard 43)
    top_cache = os.path.join(cache_dir, f"top_host{host_index}.npz")
    if os.path.exists(top_cache):
        top_packed = _load_layer_cache(top_cache)
    else:
        s2 = _stream_safetensors_dict(f"{model_uri.rstrip('/')}/model-00002-of-00048.safetensors")
        s43 = _stream_safetensors_dict(f"{model_uri.rstrip('/')}/model-00043-of-00048.safetensors")
        emb_full = s2["embed.weight"].reshape(129280, 16, 320)
        head_full = s43["head.weight"].reshape(16, 8080, 5120)
        top_packed = {
            "embed_weight": np.transpose(emb_full[:, c_start:c_end, :], (1, 0, 2)).copy(),  # [4, 129280, 320]
            "norm_weight": s43["norm.weight"],
            "head_weight": head_full[c_start:c_end].copy(),  # [4, 8080, 5120]
        }
        _save_layer_cache(top_cache, top_packed)

    embed_weight = _make_sharded_array(
        mesh, local_devices, (129280, 5120), P(None, "tp"), [top_packed["embed_weight"][i] for i in range(4)]
    )
    norm_weight = _make_replicated_array(mesh, local_devices, top_packed["norm_weight"])
    head_weight = _make_sharded_array(
        mesh, local_devices, (129280, 5120), P("tp", None), [top_packed["head_weight"][i] for i in range(4)]
    )

    # 5. Stream or load from /dev/shm all 40 backbone layers (shards 3..42)
    layers_list: List[LayerWeights] = []
    for layer_id in range(cfg.num_hidden_layers):
        l_cache = os.path.join(cache_dir, f"layer_{layer_id:02d}_host{host_index}.npz")
        if os.path.exists(l_cache):
            packed = _load_layer_cache(l_cache)
        else:
            shard_uri = f"{model_uri.rstrip('/')}/model-{layer_id + 3:05d}-of-00048.safetensors"
            raw = _stream_safetensors_dict(shard_uri)
            packed = _pack_layer_host_slice(layer_id, cfg, raw, host_index)
            _save_layer_cache(l_cache, packed)
            del raw

        def rep(key: str) -> jax.Array:
            return _make_replicated_array(mesh, local_devices, packed[key])

        def sharded(key: str, global_shape: Tuple[int, ...], spec: P) -> jax.Array:
            return _make_sharded_array(mesh, local_devices, global_shape, spec, [packed[key][i] for i in range(4)])

        is_kv_src = layer_id in cfg.kv_source_layer_ids
        is_idx_src = layer_id in cfg.index_source_layer_ids
        has_eng = layer_id in cfg.engram_layer_ids
        c_ratio = cfg.compress_ratios[layer_id]

        comp_wkv = rep("comp_wkv") if is_kv_src else None
        comp_norm = rep("comp_norm") if is_kv_src else None
        comp_wgate = rep("comp_wgate") if (is_kv_src and c_ratio == 2) else None

        idx_wq_b = rep("idx_wq_b") if is_idx_src else None
        idx_wq_b_scale = rep("idx_wq_b_scale") if is_idx_src else None
        idx_weights_proj = rep("idx_weights_proj") if is_idx_src else None
        idx_wk = rep("idx_wk") if (is_idx_src and is_kv_src) else None
        idx_k_norm = rep("idx_k_norm") if (is_idx_src and is_kv_src) else None

        engram_wkv = None
        engram_wkv_scale = None
        engram_q_weight = None
        engram_k_weight = None
        if has_eng and layer_id in engram_tpu_tensors:
            et = engram_tpu_tensors[layer_id]
            ep = f"layers.{layer_id}.engram"
            engram_wkv = _make_replicated_array(mesh, local_devices, et[f"{ep}.wkv.weight"])
            engram_wkv_scale = _make_replicated_array(mesh, local_devices, et[f"{ep}.wkv.scale"])
            engram_q_weight = _make_replicated_array(mesh, local_devices, et[f"{ep}.q_weight"])
            engram_k_weight = _make_replicated_array(mesh, local_devices, et[f"{ep}.k_weight"])

        lw = LayerWeights(
            layer_id=layer_id,
            compress_ratio=c_ratio,
            is_kv_source=is_kv_src,
            is_index_source=is_idx_src,
            has_engram=has_eng,
            attn_norm=rep("attn_norm"),
            ffn_norm=rep("ffn_norm"),
            hc_attn_fn=rep("hc_attn_fn"),
            hc_attn_scale=rep("hc_attn_scale"),
            hc_attn_base=rep("hc_attn_base"),
            hc_ffn_fn=rep("hc_ffn_fn"),
            hc_ffn_scale=rep("hc_ffn_scale"),
            hc_ffn_base=rep("hc_ffn_base"),
            wq_a=rep("wq_a"),
            wq_a_scale=rep("wq_a_scale"),
            q_norm=rep("q_norm"),
            wkv=rep("wkv"),
            wkv_scale=rep("wkv_scale"),
            kv_norm=rep("kv_norm"),
            wq_b=sharded("wq_b", (65536, 1280), P("tp", None)),
            wq_b_scale=sharded("wq_b_scale", (2048, 40), P("tp", None)),
            attn_sink=sharded("attn_sink", (128,), P("tp")),
            wo_a=sharded("wo_a", (8192, 4096), P("tp", None)),
            wo_a_scale=sharded("wo_a_scale", (256, 128), P("tp", None)),
            wo_b=sharded("wo_b", (5120, 8192), P(None, "tp")),
            wo_b_scale=sharded("wo_b_scale", (160, 256), P(None, "tp")),
            gate_weight=rep("gate_weight"),
            gate_bias=rep("gate_bias"),
            shared_w1=sharded("shared_w1", (2560, 5120), P("tp", None)),
            shared_w1_scale=sharded("shared_w1_scale", (80, 160), P("tp", None)),
            shared_w3=sharded("shared_w3", (2560, 5120), P("tp", None)),
            shared_w3_scale=sharded("shared_w3_scale", (80, 160), P("tp", None)),
            shared_w2=sharded("shared_w2", (5120, 2560), P(None, "tp")),
            shared_w2_scale=sharded("shared_w2_scale", (160, 80), P(None, "tp")),
            routed_w1=sharded("routed_w1", (384, 2560, 2560), P(None, "tp", None)),
            routed_w1_scale=sharded("routed_w1_scale", (384, 2560, 160), P(None, "tp", None)),
            routed_w3=sharded("routed_w3", (384, 2560, 2560), P(None, "tp", None)),
            routed_w3_scale=sharded("routed_w3_scale", (384, 2560, 160), P(None, "tp", None)),
            routed_w2=sharded("routed_w2", (384, 5120, 1280), P(None, None, "tp")),
            routed_w2_scale=sharded("routed_w2_scale", (384, 5120, 80), P(None, None, "tp")),
            comp_wkv=comp_wkv,
            comp_wgate=comp_wgate,
            comp_norm=comp_norm,
            idx_wq_b=idx_wq_b,
            idx_wq_b_scale=idx_wq_b_scale,
            idx_weights_proj=idx_weights_proj,
            idx_wk=idx_wk,
            idx_k_norm=idx_k_norm,
            engram_wkv=engram_wkv,
            engram_wkv_scale=engram_wkv_scale,
            engram_q_weight=engram_q_weight,
            engram_k_weight=engram_k_weight,
        )
        layers_list.append(lw)
        if host_index == 0 and ((layer_id + 1) % 8 == 0 or layer_id == 0):
            print(f"[load_dsv41_weights] Loaded layer {layer_id + 1}/40 ({time.time() - t0:.1f}s)")

    return ModelWeights(
        config=cfg,
        mesh=mesh,
        embed_weight=embed_weight,
        norm_weight=norm_weight,
        head_weight=head_weight,
        layers=tuple(layers_list),
        engram_host=engram_host,
        gcs_hashes=gcs_hashes,
    )
