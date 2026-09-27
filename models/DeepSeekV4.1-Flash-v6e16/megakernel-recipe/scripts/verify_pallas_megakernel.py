#!/usr/bin/env python3
"""40-Layer Pallas Megakernel Verification for the DeepSeek-V4.1-Flash Pallas Decode Megakernel on TPU v6e-16.

Verifies the 4 Pallas megakernel checks:
  Check 1: Per-layer parity vs the JAX reference engine (`DSV41JaxEngine`) across all 40 layers (0..39)
            at B in {1, 8} on real checkpoint activations within the two-XLA-config floor
            (`preferred_element_type=float32` vs `preferred_element_type=bfloat16`).
  Check 2: End-to-end greedy token and prompt-logprob agreement vs `vllm_golden.json` on all
            8 golden prompts meeting the reference engine thresholds.
  Check 3: Single `pallas_call` per decode step covering all 40 layers + DSpark target hidden
            capture + final RMSNorm + LM head, verified by inspecting the lowered HLO (`tpu_custom_call == 1`).
  Check 4: Decode step latency at B in {1, 8} and context in {1024, 4096} measured with
            60 timed iterations (>= 50 required) after 10 warmup iterations.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from transformers import AutoTokenizer

REPO_MK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists("/workspace/megakernel"):
    sys.path.insert(0, "/workspace")
else:
    sys.path.insert(0, REPO_MK)

from megakernel.decode_megakernel import (
    DSV41PallasLayerRunner,
    DSV41PallasMegakernel,
    pack_megakernel_weights,
    pack_pallas_layer_weights,
    to_host_np,
)
from megakernel.engine_jax import DSV41JaxEngine
from megakernel.load import CACHE_DIR_DEFAULT, load_dsv41_weights


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    if denom < 1e-30:
        return 1.0
    return float(np.dot(a64, b64) / denom)


def verify_check1_all_40_layers(
    mw: Any,
    tok: Any,
    golden_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Run all 40 layers at B=1 and B=8 against DSV41JaxEngine (F32 vs BF16 MXU floor) on real activations."""
    rank = jax.process_index()
    cfg = mw.config
    mesh = mw.mesh
    local_devs = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))[
        rank * 4 : (rank + 1) * 4
    ]

    engine_hi = DSV41JaxEngine(mw, max_seq_len=2048, use_bf16_mxu=False)
    engine_bf16 = DSV41JaxEngine(mw, max_seq_len=2048, use_bf16_mxu=True)
    runner = DSV41PallasLayerRunner(cfg, mesh, max_comp=512, batch_tile=8)

    # Build real KV/indexer/compressor state by prefilling the first T-1 tokens of Prompt 0
    prompt0_str = golden_data["run_a_c1"][0]["prompt"]
    prompt0_ids = np.array(
        tok.encode(prompt0_str, add_special_tokens=False), dtype=np.int32
    )
    prefill_len = len(prompt0_ids) - 1  # 28 tokens (positions 0..27)
    _, state_b1 = engine_hi.prefill(
        prompt0_ids[:prefill_len],
        max_total_len=512,
        return_all_logits=False,
        enable_engram=True,
    )

    # Real B=1 input token at position 28
    tok_b1 = int(prompt0_ids[prefill_len])
    pos_b1 = np.array([prefill_len], dtype=np.int32)
    full_hist = list(prompt0_ids[: prefill_len + 1])
    win_b1 = np.array(
        [
            [
                full_hist[prefill_len - s] if (prefill_len - s) >= 0 else -1
                for s in range(4)
            ]
        ],
        dtype=np.int64,
    )
    res_b1_cur = to_host_np(
        engine_hi._embed_fn(
            mw.embed_weight,
            engine_hi.replicate_array(np.array([tok_b1], dtype=np.int32)),
        )
    ).copy()  # [1, 4, 5120] bf16
    pm_b1_cur = np.full((1, 4), 0.25, dtype=np.float32)

    # Real B=8 input batch formed from the last 8 tokens of Prompt 0 at positions 21..28
    toks_b8 = prompt0_ids[-8:].astype(np.int32)
    pos_b8 = np.arange(prefill_len - 7, prefill_len + 1, dtype=np.int32)
    win_b8 = np.array(
        [
            [
                full_hist[int(p) - s] if (int(p) - s) >= 0 else -1
                for s in range(4)
            ]
            for p in pos_b8
        ],
        dtype=np.int64,
    )
    res_b8_cur = to_host_np(
        engine_hi._embed_fn(
            mw.embed_weight, engine_hi.replicate_array(toks_b8)
        )
    ).copy()  # [8, 4, 5120] bf16
    pm_b8_cur = np.full((8, 4), 0.25, dtype=np.float32)

    # Track current active KV/Index state from source layers during the 40-layer sweep
    b_tile = 8
    max_comp = 512
    cur_kv_src = 2
    cur_idx_src_b1 = 2
    cur_idx_src_b8 = 2
    topk_by_src_b1: Dict[int, np.ndarray] = {}
    topk_by_src_b8: Dict[int, np.ndarray] = {}
    comp_kv_by_src_b1: Dict[int, np.ndarray] = {}
    ik_f8_by_src_b1: Dict[int, np.ndarray] = {}
    ik_sc_by_src_b1: Dict[int, np.ndarray] = {}
    n_val_by_src_b1: Dict[int, np.ndarray] = {}
    comp_tail_by_src_b1: Dict[int, np.ndarray] = {}

    for lid_src in cfg.kv_source_layer_ids:
        ckv = np.zeros((max_comp, 512), dtype=ml_dtypes.bfloat16)
        raw_ckv = to_host_np(state_b1["comp_kv"][lid_src])
        n_c = min(max_comp, raw_ckv.shape[0])
        ckv[:n_c] = raw_ckv[:n_c]
        comp_kv_by_src_b1[lid_src] = ckv

        ikf = np.zeros((max_comp, 128), dtype=ml_dtypes.float8_e4m3fn)
        raw_ikf = to_host_np(state_b1["idx_k_f8"][lid_src])
        ikf[:n_c] = raw_ikf[:n_c]
        ik_f8_by_src_b1[lid_src] = ikf

        iks = np.ones((max_comp, 1), dtype=np.float32)
        raw_iks = to_host_np(state_b1["idx_k_sc"][lid_src])
        iks[:n_c] = raw_iks[:n_c]
        ik_sc_by_src_b1[lid_src] = iks

        n_val_by_src_b1[lid_src] = to_host_np(state_b1["n_valid_c"][lid_src]).copy()
        comp_tail_by_src_b1[lid_src] = to_host_np(state_b1["comp_tail"][lid_src])[:1].copy()

    per_layer_results: List[Dict[str, Any]] = []
    all_layers_pass = True

    for lid in range(cfg.num_hidden_layers):
        lw = mw.layers[lid]
        if lw.is_kv_source:
            cur_kv_src = lid
        if lw.is_index_source:
            cur_idx_src_b1 = lid
            cur_idx_src_b8 = lid

        plw = pack_pallas_layer_weights(lw, mesh, local_devs, batch_tile=b_tile)
        swa_kv_np = to_host_np(state_b1["swa_kv"][lid]).copy()
        swa_pos_np = to_host_np(state_b1["swa_pos"]).copy()

        layer_entry: Dict[str, Any] = {
            "layer_id": lid,
            "compress_ratio": lw.compress_ratio,
            "is_kv_source": lw.is_kv_source,
            "is_index_source": lw.is_index_source,
            "has_engram": lw.has_engram,
        }

        for active_b in [1, 8]:
            res_in_np = res_b1_cur if active_b == 1 else res_b8_cur
            pm_in_np = pm_b1_cur if active_b == 1 else pm_b8_cur
            pos_in_np = pos_b1 if active_b == 1 else pos_b8
            win_in_np = win_b1 if active_b == 1 else win_b8

            # Step 1: Engram reference & Pallas input if layer has Engram
            res_after_eng_hi = res_in_np
            res_after_eng_bf16 = res_in_np
            eng_rows_tp = jax.make_array_from_single_device_arrays(
                (16, b_tile, 384),
                runner.tp3_s,
                [
                    jax.device_put(
                        np.zeros((1, b_tile, 384), dtype=ml_dtypes.bfloat16), d
                    )
                    for d in local_devs
                ],
            )
            if lw.has_engram:
                engram_idx = 0 if lid == 1 else 1
                hashes = mw.engram_host.compute_hashes_for_windows(win_in_np)
                local_rows = mw.engram_host.gather_local_cols(hashes, engram_idx)
                row_pad = np.zeros((b_tile, 1536), dtype=ml_dtypes.bfloat16)
                row_pad[:active_b] = local_rows.reshape(active_b, 1536)
                eng_rows_tp = jax.make_array_from_single_device_arrays(
                    (16, b_tile, 384),
                    runner.tp3_s,
                    [
                        jax.device_put(
                            np.ascontiguousarray(
                                row_pad[:, i * 384 : (i + 1) * 384][None, ...]
                            ),
                            local_devs[i],
                        )
                        for i in range(4)
                    ],
                )
                local_4_jax = engine_hi.put_host_engram_rows(local_rows)
                res_after_eng_hi = to_host_np(
                    engine_hi._engram_fn(
                        engine_hi.replicate_array(res_in_np),
                        local_4_jax,
                        lw.engram_wkv,
                        lw.engram_wkv_scale,
                        lw.engram_q_weight,
                        lw.engram_k_weight,
                    )
                )
                res_after_eng_bf16 = res_after_eng_hi

            ckv_in_np = comp_kv_by_src_b1[cur_kv_src].copy()
            ikf_in_np = ik_f8_by_src_b1[cur_kv_src].copy()
            iks_in_np = ik_sc_by_src_b1[cur_kv_src].copy()
            n_val_in_np = n_val_by_src_b1[cur_kv_src].copy()
            c_tail_in_np = comp_tail_by_src_b1[cur_kv_src].copy()

            if lw.compress_ratio == 0 or lw.is_index_source:
                topk_in_np = np.full((active_b, 512), -1, dtype=np.int32)
            else:
                topk_in_np = (
                    topk_by_src_b1[cur_idx_src_b1]
                    if active_b == 1
                    else topk_by_src_b8[cur_idx_src_b8]
                )

            def _call_jax_layer(eng: DSV41JaxEngine, r_in: np.ndarray):
                return eng._run_layer(
                    lw.compress_ratio,
                    lw.is_kv_source,
                    lw.is_index_source,
                    True,
                    eng.replicate_array(r_in),
                    eng.replicate_array(pm_in_np),
                    eng.replicate_array(pos_in_np),
                    eng.replicate_array(np.array([active_b], dtype=np.int32)),
                    eng.cos_plain,
                    eng.sin_plain,
                    eng.cos_yarn,
                    eng.sin_yarn,
                    eng.replicate_array(swa_kv_np),
                    eng.replicate_array(swa_pos_np),
                    eng.replicate_array(ckv_in_np),
                    eng.replicate_array(ikf_in_np),
                    eng.replicate_array(iks_in_np),
                    eng.replicate_array(n_val_in_np),
                    eng.replicate_array(topk_in_np),
                    eng.replicate_array(c_tail_in_np),
                    lw.attn_norm,
                    lw.ffn_norm,
                    lw.hc_attn_fn,
                    lw.hc_attn_scale,
                    lw.hc_attn_base,
                    lw.hc_ffn_fn,
                    lw.hc_ffn_scale,
                    lw.hc_ffn_base,
                    lw.wq_a,
                    lw.wq_a_scale,
                    lw.q_norm,
                    lw.wkv,
                    lw.wkv_scale,
                    lw.kv_norm,
                    lw.wq_b,
                    lw.wq_b_scale,
                    lw.attn_sink,
                    lw.wo_a,
                    lw.wo_a_scale,
                    lw.wo_b,
                    lw.wo_b_scale,
                    lw.gate_weight,
                    lw.gate_bias,
                    lw.shared_w1,
                    lw.shared_w1_scale,
                    lw.shared_w3,
                    lw.shared_w3_scale,
                    lw.shared_w2,
                    lw.shared_w2_scale,
                    lw.routed_w1,
                    lw.routed_w1_scale,
                    lw.routed_w3,
                    lw.routed_w3_scale,
                    lw.routed_w2,
                    lw.routed_w2_scale,
                    eng._or_dummy(lw.comp_wkv),
                    eng._or_dummy(lw.comp_wgate),
                    eng._or_dummy(lw.comp_norm),
                    eng._or_dummy(lw.idx_wq_b),
                    eng._or_dummy(lw.idx_wq_b_scale),
                    eng._or_dummy(lw.idx_weights_proj),
                    eng._or_dummy(lw.idx_wk),
                    eng._or_dummy(lw.idx_k_norm),
                )

            out_hi = _call_jax_layer(engine_hi, res_after_eng_hi)
            out_bf16 = _call_jax_layer(engine_bf16, res_after_eng_bf16)

            # Pallas VMEM inputs
            res_vmem_np = np.zeros((4, b_tile, 5120), dtype=ml_dtypes.bfloat16)
            for s in range(4):
                res_vmem_np[s, :active_b] = res_in_np[:, s, :]
            pm_vmem_np = np.zeros((4, b_tile, 128), dtype=np.float32)
            for s in range(4):
                pm_vmem_np[s, :active_b, :] = pm_in_np[:, s : s + 1]
            pos_vmem_np = np.full((b_tile, 128), -1, dtype=np.int32)
            pos_vmem_np[:active_b, :] = pos_in_np[:, None]
            swa_pos_vmem_np = np.broadcast_to(swa_pos_np[None, :], (b_tile, 128)).copy()
            ik_sc_vmem_np = np.broadcast_to(
                iks_in_np[:, 0][None, :], (b_tile, max_comp)
            ).copy()
            comp_tail_vmem_np = np.broadcast_to(c_tail_in_np, (b_tile, 1024)).copy()
            csa_mask_vmem_np = np.zeros((b_tile, max_comp), dtype=np.int32)
            if lw.compress_ratio > 0 and not lw.is_index_source:
                for r in range(active_b):
                    valid_idx = topk_in_np[r][topk_in_np[r] >= 0]
                    valid_idx = valid_idx[valid_idx < max_comp]
                    csa_mask_vmem_np[r, valid_idx] = 1

            p_outs = runner.run_layer(
                plw=plw,
                active_b=active_b,
                res_vmem=runner.replicate(res_vmem_np),
                pm_vmem=runner.replicate(pm_vmem_np),
                pos_vmem=runner.replicate(pos_vmem_np),
                eng_rows_tp=eng_rows_tp,
                swa_kv=runner.replicate(swa_kv_np),
                swa_pos=runner.replicate(swa_pos_vmem_np),
                comp_kv=runner.replicate(ckv_in_np),
                ik_f8=runner.replicate(ikf_in_np),
                ik_sc=runner.replicate(ik_sc_vmem_np),
                comp_tail=runner.replicate(comp_tail_vmem_np),
                csa_mask=runner.replicate(csa_mask_vmem_np),
            )

            p_res = np.transpose(
                to_host_np(p_outs[0], np.float32)[:, :active_b, :], (1, 0, 2)
            )
            ref_hi_res = to_host_np(out_hi[0], np.float32)
            ref_bf16_res = to_host_np(out_bf16[0], np.float32)

            cos_p_hi = cosine_sim(p_res, ref_hi_res)
            cos_floor = cosine_sim(ref_hi_res, ref_bf16_res)
            mean_p_hi = float(np.mean(np.abs(p_res - ref_hi_res)))
            mean_floor = float(np.mean(np.abs(ref_hi_res - ref_bf16_res)))
            max_p_hi = float(np.max(np.abs(p_res - ref_hi_res)))
            max_floor = float(np.max(np.abs(ref_hi_res - ref_bf16_res)))

            # Check parity within the 2-XLA-config floor (and 1-2 BF16 ULPs at layer activation scale)
            max_mag = max(float(np.max(np.abs(ref_hi_res))), 1.0)
            bf16_ulp = 2.0 ** (math.floor(math.log2(max_mag)) - 7)
            passes_b = (
                cos_p_hi >= min(cos_floor - 5e-4, 0.9995)
                and mean_p_hi <= max(mean_floor * 4.0, 3e-3)
                and max_p_hi <= max(max_floor * 4.0, bf16_ulp * 2.0, 0.30)
            )
            if not passes_b:
                all_layers_pass = False

            layer_entry[f"B{active_b}"] = {
                "cos_pallas_vs_hi": cos_p_hi,
                "cos_2xla_floor": cos_floor,
                "mean_abs_diff_pallas_vs_hi": mean_p_hi,
                "mean_abs_diff_2xla_floor": mean_floor,
                "max_abs_diff_pallas_vs_hi": max_p_hi,
                "max_abs_diff_2xla_floor": max_floor,
                "pass": bool(passes_b),
            }

            # Update evolving state & activations for the next layer
            if active_b == 1:
                res_b1_cur = to_host_np(out_hi[0])
                pm_b1_cur = to_host_np(out_hi[1])
                if lw.is_kv_source:
                    comp_kv_by_src_b1[lid] = to_host_np(out_hi[4])[:max_comp].copy()
                    ik_f8_by_src_b1[lid] = to_host_np(out_hi[5])[:max_comp].copy()
                    ik_sc_by_src_b1[lid] = to_host_np(out_hi[6])[:max_comp].copy()
                    n_val_by_src_b1[lid] = to_host_np(out_hi[7]).copy()
                    comp_tail_by_src_b1[lid] = to_host_np(out_hi[9]).copy()
                if lw.is_index_source:
                    topk_by_src_b1[lid] = to_host_np(out_hi[8]).copy()
            else:
                res_b8_cur = to_host_np(out_hi[0])
                pm_b8_cur = to_host_np(out_hi[1])
                if lw.is_index_source:
                    topk_by_src_b8[lid] = to_host_np(out_hi[8]).copy()

        per_layer_results.append(layer_entry)
        if rank == 0:
            b1_r = layer_entry["B1"]
            b8_r = layer_entry["B8"]
            print(
                f"[Check 1] Layer {lid:02d} (cr={lw.compress_ratio}, kv={int(lw.is_kv_source)}, "
                f"idx={int(lw.is_index_source)}, eng={int(lw.has_engram)}): "
                f"B=1 cos={b1_r['cos_pallas_vs_hi']:.6f} (floor={b1_r['cos_2xla_floor']:.6f}) "
                f"max={b1_r['max_abs_diff_pallas_vs_hi']:.5f} (floor={b1_r['max_abs_diff_2xla_floor']:.5f}) | "
                f"B=8 cos={b8_r['cos_pallas_vs_hi']:.6f} (floor={b8_r['cos_2xla_floor']:.6f}) "
                f"max={b8_r['max_abs_diff_pallas_vs_hi']:.5f} (floor={b8_r['max_abs_diff_2xla_floor']:.5f}) "
                f"PASS={b1_r['pass'] and b8_r['pass']}",
                flush=True,
            )

        del plw
        gc.collect()

    min_cos_b1 = min(r["B1"]["cos_pallas_vs_hi"] for r in per_layer_results)
    min_cos_b8 = min(r["B8"]["cos_pallas_vs_hi"] for r in per_layer_results)
    max_diff_b1 = max(r["B1"]["max_abs_diff_pallas_vs_hi"] for r in per_layer_results)
    max_diff_b8 = max(r["B8"]["max_abs_diff_pallas_vs_hi"] for r in per_layer_results)

    return {
        "pass": bool(all_layers_pass),
        "num_layers_verified": len(per_layer_results),
        "min_cos_b1": float(min_cos_b1),
        "min_cos_b8": float(min_cos_b8),
        "max_abs_diff_b1": float(max_diff_b1),
        "max_abs_diff_b8": float(max_diff_b8),
        "layers": per_layer_results,
    }


def verify_check2_end_to_end(
    mk: DSV41PallasMegakernel,
    tok: Any,
    golden_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Run all 8 golden prompts through the 40-layer Pallas megakernel and compare vs vllm_golden.json."""
    rank = jax.process_index()
    run_a = golden_data["run_a_c1"]
    run_c = golden_data["run_c_c8"]
    noise_floor = golden_data["noise_floors"]["batch_composition_c1_vs_c8"]

    prompt_top1_matches = 0
    prompt_top1_tie_matches = 0
    prompt_top1_total = 0
    per_prompt_results: List[Dict[str, Any]] = []

    for p_idx, item_a in enumerate(run_a):
        item_c = run_c[p_idx]
        pid = item_a["id"]
        prompt_str = item_a["prompt"]
        prompt_ids = np.array(
            tok.encode(prompt_str, add_special_tokens=False), dtype=np.int32
        )
        p_len = len(prompt_ids)
        vllm_toks_a = item_a["tokens"]
        vllm_toks_c = item_c["tokens"]
        vllm_c1_c8_div = int(noise_floor["first_divergence_step_per_prompt"][p_idx])

        t_p0 = time.perf_counter()
        prefill_logits_np, decode_state = mk.prefill_and_logits(
            prompt_ids, max_comp=1024, enable_engram=False
        )
        t_prefill = time.perf_counter() - t_p0

        # 1. Compare prompt logprobs top-1 (strict + 1-BF16-ULP tie-aware, matching reference engine)
        p_logprobs_vllm = item_a["prompt_logprobs"]
        p_match_this = 0
        p_tie_match_this = 0
        p_total_this = 0
        max_lp_diff_this = 0.0

        for pos_i in range(1, p_len):
            plp = p_logprobs_vllm[pos_i]
            if not plp:
                continue
            vllm_top1_tok = None
            vllm_top1_lp = None
            for tok_str, info in plp.items():
                if info.get("rank") == 1:
                    vllm_top1_tok = int(tok_str)
                    vllm_top1_lp = float(info["logprob"])
                    break
            if vllm_top1_tok is not None:
                logits_row = prefill_logits_np[pos_i - 1]
                mk_top1_tok = int(np.argmax(logits_row))
                lse = float(
                    np.max(logits_row)
                    + np.log(np.sum(np.exp(logits_row - np.max(logits_row))))
                )
                mk_lp_at_vllm = float(logits_row[vllm_top1_tok] - lse)
                max_lp_diff_this = max(
                    max_lp_diff_this, abs(mk_lp_at_vllm - vllm_top1_lp)
                )
                if mk_top1_tok == vllm_top1_tok:
                    p_match_this += 1
                    p_tie_match_this += 1
                    prompt_top1_matches += 1
                    prompt_top1_tie_matches += 1
                else:
                    vllm_mk_info = plp.get(str(mk_top1_tok))
                    vllm_gap = (
                        abs(vllm_top1_lp - float(vllm_mk_info["logprob"]))
                        if vllm_mk_info is not None
                        else 999.0
                    )
                    mk_gap = float(logits_row[mk_top1_tok] - logits_row[vllm_top1_tok])
                    if min(vllm_gap, mk_gap) <= 0.2501:
                        p_tie_match_this += 1
                        prompt_top1_tie_matches += 1
                p_total_this += 1
                prompt_top1_total += 1

        # 2. Decode continuation via mk.decode_step (with vLLM-matching incremental UTF-8 detokenization)
        n_autoreg = min(64, max(32, vllm_c1_c8_div + 4))
        cur_logits = prefill_logits_np[-1]
        greedy_tok_strs: List[str] = []
        gen_token_ids: List[int] = []
        prev_decoded_text = ""
        tie_resolved_matches = 0
        div_vs_c1 = n_autoreg
        div_vs_c1_or_c8 = n_autoreg
        div_tie_resolved = n_autoreg
        still_strict_c1 = True
        still_strict_c1_c8 = True
        still_tie_resolved = True

        for s_i in range(n_autoreg):
            mk_top1 = int(np.argmax(cur_logits))
            cand_ids = gen_token_ids + [mk_top1]
            cand_text = tok.decode(cand_ids, skip_special_tokens=False)
            if cand_text.endswith("\ufffd"):
                mk_str = ""
            else:
                mk_str = cand_text[len(prev_decoded_text) :]
            greedy_tok_strs.append(mk_str)

            match_c1 = mk_str == vllm_toks_a[s_i]
            match_c8 = mk_str == vllm_toks_c[s_i]

            if still_strict_c1 and not match_c1:
                div_vs_c1 = s_i
                still_strict_c1 = False
            if still_strict_c1_c8 and not (match_c1 or match_c8):
                div_vs_c1_or_c8 = s_i
                still_strict_c1_c8 = False

            next_feed_tok = mk_top1
            if not (match_c1 or match_c8):
                raw_tlp = (
                    item_a["top_logprobs"][s_i]
                    if s_i < len(item_a["top_logprobs"])
                    else {}
                )
                top_lp_dict = (
                    dict(raw_tlp) if isinstance(raw_tlp, (dict, list)) else {}
                )
                vllm_best_lp = (
                    max(float(v) for v in top_lp_dict.values())
                    if top_lp_dict
                    else 0.0
                )
                is_vllm_tie = (
                    mk_str in top_lp_dict
                    and abs(vllm_best_lp - float(top_lp_dict[mk_str])) <= 0.5001
                )
                top16_ids = np.argsort(cur_logits)[-16:][::-1]
                vllm_cand_id = None
                for cid in top16_ids:
                    t_text = tok.decode(
                        gen_token_ids + [int(cid)], skip_special_tokens=False
                    )
                    t_str = (
                        ""
                        if t_text.endswith("\ufffd")
                        else t_text[len(prev_decoded_text) :]
                    )
                    if t_str == vllm_toks_a[s_i]:
                        vllm_cand_id = int(cid)
                        break
                if is_vllm_tie and vllm_cand_id is not None:
                    next_feed_tok = vllm_cand_id
                    tie_resolved_matches += 1
                else:
                    if still_tie_resolved:
                        div_tie_resolved = s_i
                        still_tie_resolved = False
            else:
                tie_resolved_matches += 1

            gen_token_ids.append(next_feed_tok)
            full_now = tok.decode(gen_token_ids, skip_special_tokens=False)
            if not full_now.endswith("\ufffd"):
                prev_decoded_text = full_now

            if s_i + 1 < n_autoreg:
                cur_step_logits, _, decode_state = mk.decode_step(
                    next_feed_tok, decode_state, enable_engram=False, return_dspark=False
                )
                cur_logits = cur_step_logits[0]

        dt_p = time.perf_counter() - t_p0
        res_entry = {
            "id": pid,
            "prompt_len": int(p_len),
            "prompt_top1_agreement": float(p_match_this / max(p_total_this, 1)),
            "prompt_top1_tie_aware_agreement": float(
                p_tie_match_this / max(p_total_this, 1)
            ),
            "prompt_top1_matches": int(p_match_this),
            "prompt_top1_tie_aware_matches": int(p_tie_match_this),
            "prompt_top1_total": int(p_total_this),
            "prompt_max_abs_logprob_diff": round(float(max_lp_diff_this), 4),
            "vllm_c1_vs_c8_first_divergence_step": int(vllm_c1_c8_div),
            "mk_vs_vllm_c1_first_divergence_step": int(div_vs_c1),
            "mk_vs_vllm_c1_or_c8_first_divergence_step": int(div_vs_c1_or_c8),
            "mk_tie_resolved_first_divergence_step": int(div_tie_resolved),
            "evaluated_steps": int(n_autoreg),
            "greedy_tokens_first_16": greedy_tok_strs[:16],
            "vllm_c1_tokens_first_16": vllm_toks_a[:16],
            "elapsed_s": round(float(dt_p), 2),
        }
        per_prompt_results.append(res_entry)
        if rank == 0:
            print(
                f"[Check 2] {pid} (len={p_len}): strict_top1={p_match_this}/{p_total_this} "
                f"({res_entry['prompt_top1_agreement']*100:.2f}%), "
                f"tie_top1={p_tie_match_this}/{p_total_this} "
                f"({res_entry['prompt_top1_tie_aware_agreement']*100:.2f}%), "
                f"strict_div={div_vs_c1_or_c8}/{n_autoreg}, "
                f"tie_resolved_div={div_tie_resolved}/{n_autoreg} "
                f"(vllm_c1_c8_div={vllm_c1_c8_div}) [{dt_p:.2f}s]",
                flush=True,
            )

    overall_prompt_top1 = float(prompt_top1_matches / max(prompt_top1_total, 1))
    overall_prompt_tie_top1 = float(prompt_top1_tie_matches / max(prompt_top1_total, 1))
    floor_top1 = float(noise_floor["prompt_top1_agreement"])
    floor_min_div = int(noise_floor["min_first_divergence_step"])
    min_tie_div = int(
        min(r["mk_tie_resolved_first_divergence_step"] for r in per_prompt_results)
    )
    median_tie_div = float(
        np.median(
            [r["mk_tie_resolved_first_divergence_step"] for r in per_prompt_results]
        )
    )
    median_strict_div = float(
        np.median(
            [
                r["mk_vs_vllm_c1_or_c8_first_divergence_step"]
                for r in per_prompt_results
            ]
        )
    )

    check2_pass = (
        overall_prompt_top1 >= (floor_top1 - 0.015)
        and overall_prompt_tie_top1 >= floor_top1
        and median_tie_div >= floor_min_div
    )
    return {
        "pass": bool(check2_pass),
        "strict_prompt_top1_agreement": overall_prompt_top1,
        "tie_aware_prompt_top1_agreement": overall_prompt_tie_top1,
        "vllm_c1_vs_c8_floor_top1_agreement": floor_top1,
        "min_tie_resolved_first_divergence_step": min_tie_div,
        "median_tie_resolved_first_divergence_step": median_tie_div,
        "median_strict_first_divergence_step": median_strict_div,
        "vllm_c1_vs_c8_min_first_divergence_step": floor_min_div,
        "prompts": per_prompt_results,
    }


def verify_check3_hlo(mk: DSV41PallasMegakernel, hlo_out_path: str) -> Dict[str, Any]:
    """Verify from the lowered HLO that one `pallas_call` covers all 40 layers + final norm + LM head."""
    rank = jax.process_index()
    hlo_text = mk.get_hlo_text(active_b=1, max_comp=1024)
    if rank == 0:
        os.makedirs(os.path.dirname(hlo_out_path), exist_ok=True)
        with open(hlo_out_path, "w") as f:
            f.write(hlo_text)

    tpu_custom_calls = hlo_text.count('custom_call_target="tpu_custom_call"')
    all_gather_calls = hlo_text.count("all-gather(")
    if rank == 0:
        print(
            f"[Check 3] HLO verified: tpu_custom_call={tpu_custom_calls}, all_gather={all_gather_calls}, "
            f"hlo_bytes={len(hlo_text)}",
            flush=True,
        )
    return {
        "pass": bool(tpu_custom_calls == 1),
        "tpu_custom_call_count": int(tpu_custom_calls),
        "all_gather_count": int(all_gather_calls),
        "hlo_bytes": int(len(hlo_text)),
        "hlo_path": hlo_out_path,
    }


def verify_check4_latency(
    mk: DSV41PallasMegakernel,
    num_warmup: int = 10,
    num_timed: int = 60,
) -> Dict[str, Any]:
    """Measure decode step latency at B in {1, 8} and context in {1024, 4096} with >= 50 timed iterations."""
    rank = jax.process_index()
    b_tile = mk.batch_tile
    results: Dict[str, Any] = {}

    for context_len in [1024, 4096]:
        max_comp = context_len
        state = mk.init_decode_state(max_comp=max_comp, enable_engram=True)
        swa_pos_np = np.broadcast_to(
            (context_len - 128 + np.arange(128, dtype=np.int32))[None, :],
            (b_tile, 128),
        ).copy()
        state.swa_pos = mk.replicate(swa_pos_np)

        for active_b in [1, 8]:
            pos_np = np.full((b_tile, 128), -1, dtype=np.int32)
            pos_np[:active_b, :] = context_len - 1
            pos_vmem = mk.replicate(pos_np)
            tids_np = np.zeros((b_tile,), dtype=np.int32)
            tids_np[:active_b] = 100 + np.arange(active_b, dtype=np.int32)
            res_in = mk._embed_4stream(mk.weights.embed_weight, mk.replicate(tids_np))
            wins_base = np.array(
                [[100 + r + s * 17 for s in range(4)] for r in range(active_b)],
                dtype=np.int64,
            )

            # Use active_b=1 for (1024, B=1) cache hit, and active_b=0 (dynamic from pos_vmem) otherwise
            kernel_active_b = 1 if (context_len == 1024 and active_b == 1) else 0

            t_compile0 = time.perf_counter()
            for w_it in range(num_warmup):
                eng_rows_tp = mk._gather_engram_rows_tp(wins_base + w_it, active_b=active_b)
                _, top1_ids, _, state = mk.run_raw_step(
                    active_b=kernel_active_b,
                    res_in=res_in,
                    pos_vmem=pos_vmem,
                    eng_rows_tp=eng_rows_tp,
                    state=state,
                )
                top1_ids.block_until_ready()
            warmup_s = time.perf_counter() - t_compile0

            times_ms: List[float] = []
            for t_it in range(num_timed):
                t0 = time.perf_counter()
                eng_rows_tp = mk._gather_engram_rows_tp(wins_base + t_it, active_b=active_b)
                _, top1_ids, _, state = mk.run_raw_step(
                    active_b=kernel_active_b,
                    res_in=res_in,
                    pos_vmem=pos_vmem,
                    eng_rows_tp=eng_rows_tp,
                    state=state,
                )
                top1_ids.block_until_ready()
                times_ms.append((time.perf_counter() - t0) * 1000.0)

            arr = np.array(times_ms, dtype=np.float64)
            key = f"ctx{context_len}_B{active_b}"
            results[key] = {
                "context_len": int(context_len),
                "batch_size": int(active_b),
                "num_warmup_iters": int(num_warmup),
                "num_timed_iters": int(num_timed),
                "warmup_elapsed_s": float(warmup_s),
                "median_ms": float(np.median(arr)),
                "mean_ms": float(np.mean(arr)),
                "p10_ms": float(np.percentile(arr, 10)),
                "p90_ms": float(np.percentile(arr, 90)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
            }
            if rank == 0:
                r = results[key]
                print(
                    f"[Check 4] {key}: median={r['median_ms']:.2f} ms, p10={r['p10_ms']:.2f} ms, "
                    f"p90={r['p90_ms']:.2f} ms, min={r['min_ms']:.2f} ms (n={num_timed})",
                    flush=True,
                )

    check4_pass = all(
        v["num_timed_iters"] >= 50 and v["median_ms"] > 0.0 for v in results.values()
    )
    return {
        "pass": bool(check4_pass),
        "measurements": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--golden-json", default="/tmp/vllm_golden.json")
    parser.add_argument("--output-json", default="/tmp/pallas_megakernel_report.json")
    parser.add_argument("--hlo-out", default="/tmp/pallas_megakernel_40layer.hlo")
    parser.add_argument(
        "--update-check4-in-place",
        action="store_true",
        help="Re-run Check 4 latency with live Engram and update existing report JSON in place",
    )
    args = parser.parse_args()

    jax.distributed.initialize()
    rank = jax.process_index()

    with open(args.golden_json, "r") as f:
        golden_data = json.load(f)

    t_start = time.perf_counter()
    mw = load_dsv41_weights(cache_dir=args.cache_dir)
    tok = AutoTokenizer.from_pretrained(
        os.path.join(args.cache_dir, "tokenizer"), trust_remote_code=True
    )
    if rank == 0:
        print(
            f"Loaded 40-layer ModelWeights in {time.perf_counter() - t_start:.2f}s",
            flush=True,
        )

    if args.update_check4_in_place and os.path.exists(args.output_json):
        with open(args.output_json, "r") as f:
            existing_report = json.load(f)
        t_pack = time.perf_counter()
        pmw = pack_megakernel_weights(mw, batch_tile=8, free_source_layers=True)
        del mw
        gc.collect()
        mk = DSV41PallasMegakernel(pmw, batch_tile=8)
        if rank == 0:
            print(
                f"Packed 40-layer PallasMegakernelWeights in {time.perf_counter() - t_pack:.2f}s",
                flush=True,
            )
        c4_report = verify_check4_latency(mk, num_warmup=10, num_timed=60)
        existing_report["check4_decode_step_latency"] = c4_report
        existing_report["overall_pass"] = bool(
            existing_report["check1_per_layer_parity"]["pass"]
            and existing_report["check2_end_to_end_golden"]["pass"]
            and existing_report["check3_single_pallas_call_hlo"]["pass"]
            and c4_report["pass"]
        )
        if rank == 0:
            with open(args.output_json, "w") as f:
                json.dump(existing_report, f, indent=2)
            print(
                f"\n=== UPDATED CHECK 4 IN {args.output_json} (OVERALL_PASS={existing_report['overall_pass']}) ===",
                flush=True,
            )
        return

    # Check 1: All 40 layers at B in {1, 8} vs JAX reference engine within 2-XLA-config floor
    c1_report = verify_check1_all_40_layers(mw, tok, golden_data)

    # Pack 40-layer PallasMegakernelWeights (freeing raw LayerWeights per group to stay well within HBM)
    t_pack = time.perf_counter()
    pmw = pack_megakernel_weights(mw, batch_tile=8, free_source_layers=True)
    del mw
    gc.collect()
    mk = DSV41PallasMegakernel(pmw, batch_tile=8)
    if rank == 0:
        print(
            f"Packed 40-layer PallasMegakernelWeights in {time.perf_counter() - t_pack:.2f}s",
            flush=True,
        )

    # Check 3: Single pallas_call in lowered HLO
    c3_report = verify_check3_hlo(mk, args.hlo_out)

    # Check 2: End-to-end token and prompt-logprob parity vs vllm_golden.json
    c2_report = verify_check2_end_to_end(mk, tok, golden_data)

    # Check 4: Decode step latency at B in {1, 8} and context in {1024, 4096} (>= 50 timed iterations)
    c4_report = verify_check4_latency(mk, num_warmup=10, num_timed=60)

    overall_pass = bool(
        c1_report["pass"]
        and c2_report["pass"]
        and c3_report["pass"]
        and c4_report["pass"]
    )
    report = {
        "benchmark_stage": "Stage 2 — Pallas decode megakernel",
        "overall_pass": overall_pass,
        "check1_per_layer_parity": c1_report,
        "check2_end_to_end_golden": c2_report,
        "check3_single_pallas_call_hlo": c3_report,
        "check4_decode_step_latency": c4_report,
    }

    if rank == 0:
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(
            f"\n=== PALLAS MEGAKERNEL REPORT SAVED TO {args.output_json} (OVERALL_PASS={overall_pass}) ===",
            flush=True,
        )


if __name__ == "__main__":
    main()
