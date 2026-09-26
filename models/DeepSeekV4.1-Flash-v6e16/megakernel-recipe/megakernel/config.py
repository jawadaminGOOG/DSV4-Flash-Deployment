"""Authoritative configuration for DeepSeek-V4.1-Flash on TPU v6e-16 (4x4 mesh)."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Tuple


@dataclass(frozen=True)
class DSV41Config:
    """Exact DeepSeek-V4.1-Flash configuration loaded from config.json."""

    model_type: str = "deepseek_v41_text"
    vocab_size: int = 129280
    hidden_size: int = 5120
    moe_intermediate_size: int = 2304
    padded_moe_intermediate_size: int = 2560  # 2560 / 16 = 160 (5 * 32 MXFP4 blocks, 20 * 8 sublanes)
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    swiglu_limit: float = 10.0
    rms_norm_eps: float = 1e-20
    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_original_max_pos: int = 65536
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    sliding_window: int = 128
    compress_ratios: Tuple[int, ...] = (
        0, 0,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0,
    )
    compress_rope_theta: float = 160000.0
    kv_source_layer_ids: Tuple[int, ...] = (2, 8, 14, 20)
    index_source_layer_ids: Tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer_id: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: Tuple[int, ...] = (1, 14)
    engram_num_embeddings: Tuple[int, ...] = (384006168, 384016682)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 99092
    num_nextn_predict_layers: int = 3
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: Tuple[int, ...] = (37, 38, 39)
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3
    tp_size: int = 16
    vmem_limit_bytes: int = 128 * 1024 * 1024

    @property
    def local_num_heads(self) -> int:
        assert self.num_attention_heads % self.tp_size == 0
        return self.num_attention_heads // self.tp_size  # 4 heads/chip

    @property
    def local_moe_intermediate_size(self) -> int:
        assert self.padded_moe_intermediate_size % self.tp_size == 0
        return self.padded_moe_intermediate_size // self.tp_size  # 160/chip

    @classmethod
    def from_json(cls, config_path: str) -> "DSV41Config":
        with open(config_path, "r") as f:
            raw = json.load(f)
        tc = raw.get("text_config", raw)
        rs = tc.get("rope_scaling", {})
        cfg = cls(
            model_type=tc["model_type"],
            vocab_size=tc["vocab_size"],
            hidden_size=tc["hidden_size"],
            moe_intermediate_size=tc["moe_intermediate_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc["head_dim"],
            qk_rope_head_dim=tc["qk_rope_head_dim"],
            q_lora_rank=tc["q_lora_rank"],
            o_lora_rank=tc["o_lora_rank"],
            o_groups=tc["o_groups"],
            swiglu_limit=float(tc["swiglu_limit"]),
            rms_norm_eps=float(tc["rms_norm_eps"]),
            max_position_embeddings=tc["max_position_embeddings"],
            rope_theta=float(tc["rope_theta"]),
            rope_factor=float(rs.get("factor", 16.0)),
            rope_beta_fast=int(rs.get("beta_fast", 32)),
            rope_beta_slow=int(rs.get("beta_slow", 1)),
            rope_original_max_pos=int(rs.get("original_max_position_embeddings", 65536)),
            n_routed_experts=tc["n_routed_experts"],
            n_shared_experts=tc["n_shared_experts"],
            num_experts_per_tok=tc["num_experts_per_tok"],
            scoring_func=tc["scoring_func"],
            topk_method=tc["topk_method"],
            norm_topk_prob=bool(tc["norm_topk_prob"]),
            routed_scaling_factor=float(tc["routed_scaling_factor"]),
            sliding_window=tc["sliding_window"],
            compress_ratios=tuple(tc["compress_ratios"]),
            compress_rope_theta=float(tc["compress_rope_theta"]),
            kv_source_layer_ids=tuple(tc["kv_source_layer_ids"]),
            index_source_layer_ids=tuple(tc["index_source_layer_ids"]),
            index_n_heads=tc["index_n_heads"],
            index_head_dim=tc["index_head_dim"],
            index_topk=tc["index_topk"],
            candidate_source_layer_id=tc["candidate_source_layer_id"],
            candidate_topk_blocks=tc["candidate_topk_blocks"],
            candidate_block_size=tc["candidate_block_size"],
            hc_mult=tc["hc_mult"],
            hc_sinkhorn_iters=tc["hc_sinkhorn_iters"],
            hc_eps=float(tc["hc_eps"]),
            engram_layer_ids=tuple(tc["engram_layer_ids"]),
            engram_num_embeddings=tuple(tc["engram_num_embeddings"]),
            engram_max_ngram_size=tc["engram_max_ngram_size"],
            engram_vocab_size=tc["engram_vocab_size"],
            engram_n_heads=tc["engram_n_heads"],
            engram_head_dim=tc["engram_head_dim"],
            engram_pad_token_id=tc["engram_pad_token_id"],
            engram_compressed_vocab_size=tc["engram_compressed_vocab_size"],
            num_nextn_predict_layers=tc["num_nextn_predict_layers"],
            dspark_block_size=tc["dspark_block_size"],
            dspark_noise_token_id=tc["dspark_noise_token_id"],
            dspark_target_layer_ids=tuple(tc["dspark_target_layer_ids"]),
            dspark_markov_rank=tc["dspark_markov_rank"],
            dspark_n_routed_experts=tc["dspark_n_routed_experts"],
            dspark_num_experts_per_tok=tc["dspark_num_experts_per_tok"],
        )
        return cfg
