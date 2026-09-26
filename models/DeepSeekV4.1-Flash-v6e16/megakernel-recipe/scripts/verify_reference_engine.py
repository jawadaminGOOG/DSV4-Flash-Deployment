#!/usr/bin/env python3
"""Pure-JAX 40-layer reference engine verification vs vLLM golden set + SHA-256 + 5-component perturbation checks."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

os.makedirs("/dev/shm/jax_cache_dsv41", exist_ok=True)
jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache_dsv41")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

RECIPE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists("/workspace/megakernel"):
    sys.path.insert(0, "/workspace")
elif os.path.exists("/tmp/megakernel_pkg/megakernel"):
    sys.path.insert(0, "/tmp/megakernel_pkg")
else:
    sys.path.insert(0, RECIPE_ROOT)

from megakernel.config import DSV41Config
from megakernel.engine_jax import DSV41JaxEngine
from megakernel.load import load_dsv41_weights


def perturb_sharded(arr: jax.Array, fn: Callable[[Any], Any]) -> jax.Array:
    """Apply `fn` to each local shard of `arr` and preserve exact multi-host sharding and dtype."""
    return jax.make_array_from_single_device_arrays(
        arr.shape,
        arr.sharding,
        [jax.device_put(jnp.asarray(fn(s.data), dtype=arr.dtype), s.device) for s in arr.addressable_shards],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="/tmp/vllm_golden.json")
    parser.add_argument("--hashes", default="/tmp/ckpt_hashes.json")
    parser.add_argument("--output", default="/tmp/reference_engine_report.json")
    args = parser.parse_args()

    jax.distributed.initialize()
    host_idx = jax.process_index()
    if host_idx == 0:
        print(f"[Stage 1] Initialized JAX distributed: {jax.device_count()} devices across {jax.process_count()} hosts")

    t0 = time.time()
    weights = load_dsv41_weights(
        reference_hashes_path=args.hashes,
        load_engram_tables=True,
    )
    load_time_s = time.time() - t0
    if host_idx == 0:
        print(f"[Stage 1] Weights loaded & SHA-256 verified in {load_time_s:.1f}s")

    tok = AutoTokenizer.from_pretrained("/dev/shm/dsv41_megakernel_cache_v1/tokenizer", trust_remote_code=True)
    engine = DSV41JaxEngine(weights, max_seq_len=2048)

    with open(args.golden, "r") as f:
        golden = json.load(f)

    run_a = golden["run_a_c1"]
    run_c = golden["run_c_c8"]
    noise_floor = golden["noise_floors"]["batch_composition_c1_vs_c8"]

    prompt_top1_matches = 0
    prompt_top1_tie_matches = 0
    prompt_top1_total = 0

    per_prompt_results: List[Dict[str, Any]] = []

    for p_idx, item_a in enumerate(run_a):
        item_c = run_c[p_idx]
        pid = item_a["id"]
        prompt_str = item_a["prompt"]
        prompt_ids = np.array(tok.encode(prompt_str, add_special_tokens=False), dtype=np.int32)
        p_len = len(prompt_ids)
        assert p_len == len(item_a["prompt_logprobs"]), (
            f"Prompt length mismatch on {pid}: {p_len} vs {len(item_a['prompt_logprobs'])}"
        )

        vllm_toks_a = item_a["tokens"]
        vllm_toks_c = item_c["tokens"]
        vllm_c1_c8_div = noise_floor["first_divergence_step_per_prompt"][p_idx]

        t_p0 = time.time()
        prefill_logits_np, decode_state = engine.prefill(
            prompt_ids, max_total_len=512, return_all_logits=True, enable_engram=False
        )
        t_prefill = time.time() - t_p0

        # 1. Compare prompt logprobs top-1 (strict + 1-BF16-ULP tie-aware)
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
                jax_top1_tok = int(np.argmax(logits_row))
                lse = float(np.max(logits_row) + np.log(np.sum(np.exp(logits_row - np.max(logits_row)))))
                jax_lp_at_vllm = float(logits_row[vllm_top1_tok] - lse)
                max_lp_diff_this = max(max_lp_diff_this, abs(jax_lp_at_vllm - vllm_top1_lp))
                if jax_top1_tok == vllm_top1_tok:
                    p_match_this += 1
                    p_tie_match_this += 1
                    prompt_top1_matches += 1
                    prompt_top1_tie_matches += 1
                else:
                    # Check if jax_top1_tok is tied within 2 BF16 ULPs (0.25 logprob) in vLLM or JAX
                    vllm_jax_info = plp.get(str(jax_top1_tok))
                    vllm_gap = (
                        abs(vllm_top1_lp - float(vllm_jax_info["logprob"]))
                        if vllm_jax_info is not None
                        else 999.0
                    )
                    jax_gap = float(logits_row[jax_top1_tok] - logits_row[vllm_top1_tok])
                    if min(vllm_gap, jax_gap) <= 0.2501:
                        p_tie_match_this += 1
                        prompt_top1_tie_matches += 1
                p_total_this += 1
                prompt_top1_total += 1

        if host_idx == 0:
            print(
                f"[Stage 1 Prefill] {pid} (len={p_len}): strict_top1={p_match_this}/{p_total_this} "
                f"({p_match_this/max(p_total_this,1)*100:.2f}%), "
                f"tie_aware_top1={p_tie_match_this}/{p_total_this} "
                f"({p_tie_match_this/max(p_total_this,1)*100:.2f}%), "
                f"max_lp_diff={max_lp_diff_this:.3f} [{t_prefill:.1f}s]"
            )

        # 2. Decode continuation via engine.decode_step (with vLLM-matching incremental UTF-8 detokenization)
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
            jax_top1 = int(np.argmax(cur_logits))
            cand_ids = gen_token_ids + [jax_top1]
            cand_text = tok.decode(cand_ids, skip_special_tokens=False)
            if cand_text.endswith("\ufffd"):
                jax_str = ""
            else:
                jax_str = cand_text[len(prev_decoded_text) :]
            greedy_tok_strs.append(jax_str)

            match_c1 = (jax_str == vllm_toks_a[s_i])
            match_c8 = (jax_str == vllm_toks_c[s_i])

            if still_strict_c1 and not match_c1:
                div_vs_c1 = s_i
                still_strict_c1 = False
            if still_strict_c1_c8 and not (match_c1 or match_c8):
                div_vs_c1_or_c8 = s_i
                still_strict_c1_c8 = False

            # Check if step s_i is a near-tie in vLLM (<= 4 BF16 ULPs = 0.50)
            next_feed_tok = jax_top1
            if not (match_c1 or match_c8):
                raw_tlp = item_a["top_logprobs"][s_i] if s_i < len(item_a["top_logprobs"]) else {}
                top_lp_dict = dict(raw_tlp) if isinstance(raw_tlp, (dict, list)) else {}
                vllm_best_lp = max(float(v) for v in top_lp_dict.values()) if top_lp_dict else 0.0
                is_vllm_tie = (
                    jax_str in top_lp_dict and abs(vllm_best_lp - float(top_lp_dict[jax_str])) <= 0.5001
                )
                top16_ids = np.argsort(cur_logits)[-16:][::-1]
                vllm_cand_id = None
                for cid in top16_ids:
                    t_text = tok.decode(gen_token_ids + [int(cid)], skip_special_tokens=False)
                    t_str = "" if t_text.endswith("\ufffd") else t_text[len(prev_decoded_text) :]
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
                cur_logits, decode_state = engine.decode_step(next_feed_tok, decode_state, enable_engram=False)

        dt_p = time.time() - t_p0
        res_entry = {
            "id": pid,
            "prompt_len": p_len,
            "prompt_top1_agreement": p_match_this / max(p_total_this, 1),
            "prompt_top1_tie_aware_agreement": p_tie_match_this / max(p_total_this, 1),
            "prompt_top1_matches": p_match_this,
            "prompt_top1_tie_aware_matches": p_tie_match_this,
            "prompt_top1_total": p_total_this,
            "prompt_max_abs_logprob_diff": round(max_lp_diff_this, 4),
            "vllm_c1_vs_c8_first_divergence_step": vllm_c1_c8_div,
            "jax_vs_vllm_c1_first_divergence_step": div_vs_c1,
            "jax_vs_vllm_c1_or_c8_first_divergence_step": div_vs_c1_or_c8,
            "jax_tie_resolved_first_divergence_step": div_tie_resolved,
            "evaluated_steps": n_autoreg,
            "greedy_tokens_first_16": greedy_tok_strs[:16],
            "vllm_c1_tokens_first_16": vllm_toks_a[:16],
            "elapsed_s": round(dt_p, 2),
        }
        per_prompt_results.append(res_entry)
        if host_idx == 0:
            print(
                f"[Stage 1 Decode] {pid}: strict_div={div_vs_c1_or_c8}/{n_autoreg}, "
                f"tie_resolved_div={div_tie_resolved}/{n_autoreg} "
                f"(vllm_c1_c8_div={vllm_c1_c8_div}) [{dt_p:.1f}s]"
            )

    overall_prompt_top1 = prompt_top1_matches / max(prompt_top1_total, 1)
    overall_prompt_tie_top1 = prompt_top1_tie_matches / max(prompt_top1_total, 1)

    # 3. Check 3: Negative control — perturb 5 components and prove each changes output logits/tokens
    p0_ids = np.array(tok.encode(run_a[0]["prompt"], add_special_tokens=False), dtype=np.int32)
    base_logits_np, _ = engine.prefill(p0_ids, max_total_len=512, return_all_logits=True, enable_engram=True)
    base_preds = np.argmax(base_logits_np, axis=-1)

    perturb_results: Dict[str, Any] = {}

    def record_perturbation(name: str, nl: np.ndarray) -> None:
        preds = np.argmax(nl, axis=-1)
        max_diff = float(np.max(np.abs(nl - base_logits_np)))
        changed_positions = int(np.sum(preds != base_preds))
        perturb_results[name] = {
            "max_abs_logit_diff": max_diff,
            "changed_top1_positions": changed_positions,
            "total_positions": int(len(base_preds)),
            "passed_negative_control": bool(max_diff > 1e-2 and changed_positions > 0),
        }
        if host_idx == 0:
            print(
                f"[Stage 1 Perturb] {name}: max_abs_logit_diff={max_diff:.4f}, "
                f"changed_top1_positions={changed_positions}/{len(base_preds)}"
            )

    # (a) Perturb Layer 0 routed_w1 + routed_w1_scale
    orig_rw1 = weights.layers[0].routed_w1
    orig_rw1_s = weights.layers[0].routed_w1_scale
    weights.layers[0].routed_w1 = perturb_sharded(
        orig_rw1, lambda x: (x.astype(jnp.int8) ^ jnp.int8(-7)).astype(x.dtype)
    )
    weights.layers[0].routed_w1_scale = perturb_sharded(
        orig_rw1_s, lambda x: (x + jnp.uint8(12)).astype(x.dtype)
    )
    l_rw1, _ = engine.prefill(p0_ids, max_total_len=512, return_all_logits=True)
    record_perturbation("routed_expert_w1_L0", l_rw1)
    weights.layers[0].routed_w1 = orig_rw1
    weights.layers[0].routed_w1_scale = orig_rw1_s

    # (b) Perturb Layer 2 compressor wkv
    orig_cwkv = weights.layers[2].comp_wkv
    weights.layers[2].comp_wkv = perturb_sharded(orig_cwkv, lambda x: (x.astype(jnp.float32) * -50.0).astype(x.dtype))
    l_comp, _ = engine.prefill(p0_ids, max_total_len=512, return_all_logits=True)
    record_perturbation("compressor_wkv_L2", l_comp)
    weights.layers[2].comp_wkv = orig_cwkv

    # (c) Perturb Layer 2 indexer (with index_topk=2 so top-k scoring is active on short prompt)
    orig_iwproj = weights.layers[2].idx_weights_proj
    engine_idx_test = DSV41JaxEngine(
        type(weights)(
            config=DSV41Config(
                **{
                    **weights.config.__dict__,
                    "index_topk": 2,
                    "num_hidden_layers": 3,
                    "kv_source_layer_ids": (2,),
                    "index_source_layer_ids": (2,),
                    "engram_layer_ids": (1,),
                }
            ),
            mesh=weights.mesh,
            embed_weight=weights.embed_weight,
            norm_weight=weights.norm_weight,
            head_weight=weights.head_weight,
            layers=weights.layers[:3],
            engram_host=weights.engram_host,
            gcs_hashes=weights.gcs_hashes,
        ),
        max_seq_len=2048,
    )
    base_idx_np, _ = engine_idx_test.prefill(p0_ids, max_total_len=512, return_all_logits=True)
    base_idx_preds = np.argmax(base_idx_np, axis=-1)
    weights.layers[2].idx_weights_proj = perturb_sharded(
        orig_iwproj, lambda x: (x.astype(jnp.float32) * -100.0).astype(x.dtype)
    )
    pert_idx_np, _ = engine_idx_test.prefill(p0_ids, max_total_len=512, return_all_logits=True)
    pert_idx_preds = np.argmax(pert_idx_np, axis=-1)
    idx_diff = float(np.max(np.abs(pert_idx_np - base_idx_np)))
    idx_changed = int(np.sum(pert_idx_preds != base_idx_preds))
    perturb_results["indexer_weights_proj_L2"] = {
        "max_abs_logit_diff": idx_diff,
        "changed_top1_positions": idx_changed,
        "total_positions": int(len(base_idx_preds)),
        "passed_negative_control": bool(idx_diff > 1e-2 and idx_changed > 0),
    }
    if host_idx == 0:
        print(
            f"[Stage 1 Perturb] indexer_weights_proj_L2: max_abs_logit_diff={idx_diff:.4f}, "
            f"changed_top1_positions={idx_changed}/{len(base_idx_preds)}"
        )
    weights.layers[2].idx_weights_proj = orig_iwproj

    # (d) Perturb Layer 0 mHC parameter hc_attn_base
    orig_hcb = weights.layers[0].hc_attn_base
    weights.layers[0].hc_attn_base = perturb_sharded(orig_hcb, lambda x: x + jnp.float32(50.0))
    l_mhc, _ = engine.prefill(p0_ids, max_total_len=512, return_all_logits=True)
    record_perturbation("mhc_attn_base_L0", l_mhc)
    weights.layers[0].hc_attn_base = orig_hcb

    # (e) Perturb Layer 1 Engram row for p0
    orig_emmap = weights.engram_host.scale_mmaps[0]
    padded_p0 = np.zeros((64,), dtype=np.int32)
    padded_p0[: len(p0_ids)] = p0_ids
    win_p0 = engine._build_engram_windows_prefill(padded_p0)
    hashes_p0 = weights.engram_host.compute_hashes_for_windows(win_p0)
    active_rows = hashes_p0[:, 0, weights.engram_host.col_start : weights.engram_host.col_end] - weights.engram_host.vocab_start[0]
    mod_scale = np.array(orig_emmap, copy=True)
    mod_scale[active_rows] = np.uint8(145)
    weights.engram_host.scale_mmaps[0] = mod_scale
    l_eng, _ = engine.prefill(p0_ids, max_total_len=512, return_all_logits=True)
    record_perturbation("engram_table_row_L1", l_eng)
    weights.engram_host.scale_mmaps[0] = orig_emmap

    if host_idx == 0:
        report = {
            "benchmark_stage": "reference_engine",
            "load_time_s": round(load_time_s, 2),
            "gcs_sha256_verified": True,
            "gcs_hashes": weights.gcs_hashes,
            "prompt_top1_agreement": overall_prompt_top1,
            "prompt_top1_tie_aware_agreement": overall_prompt_tie_top1,
            "prompt_top1_matches": prompt_top1_matches,
            "prompt_top1_tie_aware_matches": prompt_top1_tie_matches,
            "prompt_top1_total": prompt_top1_total,
            "vllm_batch_composition_noise_floor": noise_floor["prompt_top1_agreement"],
            "vllm_c1_vs_c8_min_first_divergence_step": noise_floor["min_first_divergence_step"],
            "vllm_c1_vs_c8_median_first_divergence_step": noise_floor["median_first_divergence_step"],
            "per_prompt": per_prompt_results,
            "perturbations": perturb_results,
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(
            f"[Stage 1 SUMMARY] strict_prompt_top1={overall_prompt_top1*100:.2f}%, "
            f"tie_aware_prompt_top1={overall_prompt_tie_top1*100:.2f}% "
            f"(floor={noise_floor['prompt_top1_agreement']*100:.2f}%)"
        )


if __name__ == "__main__":
    main()
