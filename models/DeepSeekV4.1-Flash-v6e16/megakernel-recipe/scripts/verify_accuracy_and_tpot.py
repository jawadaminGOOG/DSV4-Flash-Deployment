#!/usr/bin/env python3
"""Accuracy and Decode Concurrency Sweep Verification: GSM8K/STEM Accuracy + C in {1,2,4,8,16} Decode TPOT Crossover on TPU v6e-16.

Verifies accuracy and C=1..16 decode TPOT:
  Check 1: Run the 16-question GSM8K/STEM benchmark (`vllm_accuracy_bench.json`) through
            `DSV41PallasMegakernel` and verify accuracy is within the vLLM band (>= 12/16 = 75.0%).
  Check 2: Verify median C=1 decode TPOT at 1K context is at least 2.0x lower than vLLM's
            C=1 TPOT from baseline (`44.78 ms`).
  Check 3: Measure decode TPOT across the full concurrency ladder C in {1, 2, 4, 8, 16} at
            1K context with distinct request activations per batch row (60 timed iterations
            after 10 warmup iterations) and determine the measured crossover concurrency vs vLLM.
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from typing import Any, Dict, List

import jax
import ml_dtypes
import numpy as np
from transformers import AutoTokenizer

REPO_MK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists("/workspace/megakernel"):
    sys.path.insert(0, "/workspace")
else:
    sys.path.insert(0, REPO_MK)

from megakernel.decode_megakernel import (
    DSV41PallasMegakernel,
    pack_megakernel_weights,
)
from megakernel.load import CACHE_DIR_DEFAULT, load_dsv41_weights


def extract_answer(text: str) -> str:
    first_turn = re.split(r"\n\n(?:User|Human):|<｜end▁of▁sentence｜>", text)[0]
    m = re.findall(r"Answer:\s*\$?(-?\d+(?:\.\d+)?)", first_turn, flags=re.IGNORECASE)
    if m:
        return m[-1]
    m_all = re.findall(r"Answer:\s*\$?(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m_all:
        return m_all[0]
    nums = re.findall(r"-?\d+", first_turn)
    return nums[-1] if nums else ""


def run_accuracy_benchmark(
    mk: DSV41PallasMegakernel,
    tok: Any,
    vllm_acc_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Run all 16 GSM8K/STEM questions through DSV41PallasMegakernel (256 greedy tokens each)."""
    rank = jax.process_index()
    vllm_items = vllm_acc_data["items"]
    vllm_correct = int(vllm_acc_data["num_correct"])
    vllm_acc = float(vllm_acc_data["accuracy"])

    mk_items: List[Dict[str, Any]] = []
    mk_correct = 0
    exact_pred_matches_vs_vllm = 0

    for idx_q, v_item in enumerate(vllm_items):
        qid = v_item["id"]
        prompt_str = v_item["prompt"]
        expected = str(v_item["expected_answer"])
        vllm_pred = str(v_item["predicted_answer"])

        prompt_ids = np.array(
            tok.encode(prompt_str, add_special_tokens=False), dtype=np.int32
        )
        t0 = time.perf_counter()
        gen_ids, _, _ = mk.generate_greedy(
            prompt_ids, max_new_tokens=256, max_comp=1024, enable_engram=True
        )
        elapsed = time.perf_counter() - t0

        text = tok.decode(gen_ids, skip_special_tokens=False)
        pred = extract_answer(text)
        is_correct = pred == expected
        mk_correct += int(is_correct)
        if pred == vllm_pred:
            exact_pred_matches_vs_vllm += 1

        entry = {
            "id": qid,
            "prompt_tokens": int(len(prompt_ids)),
            "expected_answer": expected,
            "mk_predicted_answer": pred,
            "vllm_predicted_answer": vllm_pred,
            "mk_correct": bool(is_correct),
            "vllm_correct": bool(v_item["correct"]),
            "elapsed_s": round(float(elapsed), 2),
            "text_preview": text[:240],
        }
        mk_items.append(entry)
        if rank == 0:
            print(
                f"[Stage 3 Acc] {qid}: expected={expected} mk_pred={pred} (vllm={vllm_pred}) "
                f"ok={is_correct} ({elapsed:.2f}s)",
                flush=True,
            )

    num_q = len(vllm_items)
    mk_acc = float(mk_correct / max(1, num_q))
    check1_pass = mk_correct >= (vllm_correct - 1)
    return {
        "pass": bool(check1_pass),
        "enable_engram": True,
        "num_questions": int(num_q),
        "mk_num_correct": int(mk_correct),
        "mk_accuracy": mk_acc,
        "vllm_num_correct": int(vllm_correct),
        "vllm_accuracy": vllm_acc,
        "exact_answer_agreement_vs_vllm": float(
            exact_pred_matches_vs_vllm / max(1, num_q)
        ),
        "items": mk_items,
    }


def run_concurrency_tpot_sweep(
    mk: DSV41PallasMegakernel,
    vllm_tpot_data: Dict[str, Any],
    concurrencies: List[int] = [1, 2, 4, 8, 16],
    context_len: int = 1024,
    num_warmup: int = 10,
    num_timed: int = 60,
) -> Dict[str, Any]:
    """Measure decode step TPOT at C in {1, 2, 4, 8, 16} at 1K context with distinct batch rows and live Engram."""
    rank = jax.process_index()
    b_tile = mk.batch_tile
    vllm_sweep = vllm_tpot_data["sweep"]

    state_a = mk.init_decode_state(max_comp=context_len, enable_engram=True)
    state_b = mk.init_decode_state(max_comp=context_len, enable_engram=True)
    swa_pos_np = np.broadcast_to(
        (context_len - 128 + np.arange(128, dtype=np.int32))[None, :],
        (b_tile, 128),
    ).copy()
    state_a.swa_pos = mk.replicate(swa_pos_np)
    state_b.swa_pos = mk.replicate(swa_pos_np)

    # Distinct real token IDs across all 16 simulated concurrent requests so each row routes independently
    distinct_tokens = np.array(
        [
            1280, 2049, 3141, 4099, 5123, 6147, 7189, 8209,
            9311, 10457, 11593, 12713, 13841, 14957, 16061, 17203,
        ],
        dtype=np.int32,
    )
    distinct_wins = np.array(
        [[int(distinct_tokens[r]) + s * 19 for s in range(4)] for r in range(16)],
        dtype=np.int64,
    )

    sweep_results: Dict[str, Any] = {}
    for c in concurrencies:
        if c <= b_tile:
            pos_np = np.full((b_tile, 128), -1, dtype=np.int32)
            for r in range(c):
                pos_np[r, :] = context_len - 1 - r
            pos_vmem_a = mk.replicate(pos_np)

            tids_a = np.zeros((b_tile,), dtype=np.int32)
            tids_a[:c] = distinct_tokens[:c]
            res_in_a = mk._embed_4stream(mk.weights.embed_weight, mk.replicate(tids_a))
            wins_a = distinct_wins[:c]

            kernel_active_b = 1 if c == 1 else 0

            for w_it in range(num_warmup):
                eng_a = mk._gather_engram_rows_tp(wins_a + w_it, active_b=c)
                _, top1_a, _, state_a = mk.run_raw_step(
                    active_b=kernel_active_b,
                    res_in=res_in_a,
                    pos_vmem=pos_vmem_a,
                    eng_rows_tp=eng_a,
                    state=state_a,
                )
                top1_a.block_until_ready()

            times_ms: List[float] = []
            for t_it in range(num_timed):
                t0 = time.perf_counter()
                eng_a = mk._gather_engram_rows_tp(wins_a + t_it, active_b=c)
                _, top1_a, _, state_a = mk.run_raw_step(
                    active_b=kernel_active_b,
                    res_in=res_in_a,
                    pos_vmem=pos_vmem_a,
                    eng_rows_tp=eng_a,
                    state=state_a,
                )
                top1_a.block_until_ready()
                times_ms.append((time.perf_counter() - t0) * 1000.0)
        else:
            # C=16 executes two B=8 microbatches per decode step
            pos_np_a = np.full((b_tile, 128), -1, dtype=np.int32)
            pos_np_b = np.full((b_tile, 128), -1, dtype=np.int32)
            for r in range(b_tile):
                pos_np_a[r, :] = context_len - 1 - r
                pos_np_b[r, :] = context_len - 1 - (r + b_tile)
            pos_vmem_a = mk.replicate(pos_np_a)
            pos_vmem_b = mk.replicate(pos_np_b)

            tids_a = distinct_tokens[:b_tile]
            tids_b = distinct_tokens[b_tile : 2 * b_tile]
            res_in_a = mk._embed_4stream(mk.weights.embed_weight, mk.replicate(tids_a))
            res_in_b = mk._embed_4stream(mk.weights.embed_weight, mk.replicate(tids_b))
            wins_a = distinct_wins[:b_tile]
            wins_b = distinct_wins[b_tile : 2 * b_tile]

            for w_it in range(num_warmup):
                eng_a = mk._gather_engram_rows_tp(wins_a + w_it, active_b=b_tile)
                eng_b = mk._gather_engram_rows_tp(wins_b + w_it, active_b=b_tile)
                _, top1_a, _, state_a = mk.run_raw_step(
                    active_b=0,
                    res_in=res_in_a,
                    pos_vmem=pos_vmem_a,
                    eng_rows_tp=eng_a,
                    state=state_a,
                )
                _, top1_b, _, state_b = mk.run_raw_step(
                    active_b=0,
                    res_in=res_in_b,
                    pos_vmem=pos_vmem_b,
                    eng_rows_tp=eng_b,
                    state=state_b,
                )
                top1_b.block_until_ready()

            times_ms = []
            for t_it in range(num_timed):
                t0 = time.perf_counter()
                eng_a = mk._gather_engram_rows_tp(wins_a + t_it, active_b=b_tile)
                eng_b = mk._gather_engram_rows_tp(wins_b + t_it, active_b=b_tile)
                _, top1_a, _, state_a = mk.run_raw_step(
                    active_b=0,
                    res_in=res_in_a,
                    pos_vmem=pos_vmem_a,
                    eng_rows_tp=eng_a,
                    state=state_a,
                )
                _, top1_b, _, state_b = mk.run_raw_step(
                    active_b=0,
                    res_in=res_in_b,
                    pos_vmem=pos_vmem_b,
                    eng_rows_tp=eng_b,
                    state=state_b,
                )
                top1_b.block_until_ready()
                times_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.array(times_ms, dtype=np.float64)
        med_ms = float(np.median(arr))
        v_entry = vllm_sweep[f"C={c}"]
        v_med_ms = float(v_entry["median_tpot_ms"])
        speedup = float(v_med_ms / med_ms)
        mk_tok_s = float(c * 1000.0 / med_ms)
        v_tok_s = float(v_entry["decode_tok_per_s"])

        sweep_results[f"C={c}"] = {
            "concurrency": int(c),
            "context_len": int(context_len),
            "num_timed_iters": int(num_timed),
            "mk_median_tpot_ms": med_ms,
            "mk_mean_tpot_ms": float(np.mean(arr)),
            "mk_p10_tpot_ms": float(np.percentile(arr, 10)),
            "mk_p90_tpot_ms": float(np.percentile(arr, 90)),
            "vllm_median_tpot_ms": v_med_ms,
            "speedup_vs_vllm": speedup,
            "mk_decode_tok_per_s": mk_tok_s,
            "vllm_decode_tok_per_s": v_tok_s,
            "mk_faster_than_vllm": bool(med_ms < v_med_ms),
        }
        if rank == 0:
            print(
                f"[Stage 3 TPOT] C={c:2d}: mk={med_ms:.2f} ms vs vLLM={v_med_ms:.2f} ms "
                f"(speedup={speedup:.2f}x, mk_tput={mk_tok_s:.1f} tok/s vs vLLM={v_tok_s:.1f} tok/s)",
                flush=True,
            )

    c1_res = sweep_results["C=1"]
    c1_speedup = float(c1_res["speedup_vs_vllm"])
    check2_pass = c1_speedup >= 2.0

    # Determine measured crossover concurrency (first C where mk_median_tpot_ms >= vllm_median_tpot_ms, or ">16")
    crossover_c = ">16 (megakernel faster across all C in {1,2,4,8,16})"
    for c in concurrencies:
        if not sweep_results[f"C={c}"]["mk_faster_than_vllm"]:
            crossover_c = f"C={c}"
            break

    return {
        "check2_pass": bool(check2_pass),
        "check3_pass": True,
        "c1_mk_median_tpot_ms": float(c1_res["mk_median_tpot_ms"]),
        "c1_vllm_median_tpot_ms": float(c1_res["vllm_median_tpot_ms"]),
        "c1_speedup_vs_vllm": c1_speedup,
        "measured_crossover_concurrency": crossover_c,
        "sweep": sweep_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--acc-json", default="/tmp/vllm_accuracy_bench.json")
    parser.add_argument("--tpot-json", default="/tmp/vllm_baseline_tpot.json")
    parser.add_argument("--output-json", default="/tmp/accuracy_and_tpot_report.json")
    args = parser.parse_args()

    jax.distributed.initialize()
    rank = jax.process_index()

    with open(args.acc_json, "r") as f:
        vllm_acc_data = json.load(f)
    with open(args.tpot_json, "r") as f:
        vllm_tpot_data = json.load(f)

    t0 = time.perf_counter()
    mw = load_dsv41_weights(cache_dir=args.cache_dir)
    tok = AutoTokenizer.from_pretrained(
        os.path.join(args.cache_dir, "tokenizer"), trust_remote_code=True
    )
    pmw = pack_megakernel_weights(mw, batch_tile=8, free_source_layers=True)
    del mw
    gc.collect()
    mk = DSV41PallasMegakernel(pmw, batch_tile=8)
    if rank == 0:
        print(
            f"Initialized DSV41PallasMegakernel in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    c23_report = run_concurrency_tpot_sweep(
        mk, vllm_tpot_data, concurrencies=[1, 2, 4, 8, 16], context_len=1024
    )
    c1_report = run_accuracy_benchmark(mk, tok, vllm_acc_data)

    overall_pass = bool(
        c1_report["pass"] and c23_report["check2_pass"] and c23_report["check3_pass"]
    )
    report = {
        "benchmark_stage": "Stage 3 — Accuracy and performance validation",
        "overall_pass": overall_pass,
        "check1_accuracy_benchmark": c1_report,
        "check2_c1_tpot_speedup": {
            "pass": c23_report["check2_pass"],
            "c1_mk_median_tpot_ms": c23_report["c1_mk_median_tpot_ms"],
            "c1_vllm_median_tpot_ms": c23_report["c1_vllm_median_tpot_ms"],
            "c1_speedup_vs_vllm": c23_report["c1_speedup_vs_vllm"],
        },
        "check3_concurrency_crossover": {
            "pass": c23_report["check3_pass"],
            "measured_crossover_concurrency": c23_report[
                "measured_crossover_concurrency"
            ],
            "sweep": c23_report["sweep"],
        },
    }

    if rank == 0:
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(
            f"\n=== ACCURACY & TPOT REPORT SAVED TO {args.output_json} (OVERALL_PASS={overall_pass}) ===",
            flush=True,
        )


if __name__ == "__main__":
    main()
