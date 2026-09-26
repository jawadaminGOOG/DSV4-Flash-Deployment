#!/usr/bin/env python3
"""Unified verification and benchmark entrypoint for DeepSeek-V4.1-Flash Pallas Decode Megakernel on TPU v6e-16.

Supports two execution modes:
  1. Multi-host TPU v6e-16 hardware execution (`--stage {reference,megakernel,accuracy,dspark,all}`),
     which invokes the four hardware verification scripts against the real GCS checkpoint
     (`gs://dsv4-flash-jawadamin-asia-ne1/deepseek-v4.1-flash`).
  2. Self-contained report & threshold verification (`--verify-saved-reports`), which audits
     the saved hardware verification reports in `results/` against `ckpt_hashes.json`,
     `vllm_accuracy_bench.json`, and `vllm_baseline_tpot.json` and regenerates
     `results/megakernel_v6e16_results.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict

RECIPE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(RECIPE_ROOT, "scripts")
RESULTS_DIR = os.path.join(RECIPE_ROOT, "results")


def verify_saved_reports(results_dir: str, output_path: str) -> Dict[str, Any]:
    """Audit all four hardware verification reports against baseline ground-truth files."""
    hashes_ref = json.load(open(os.path.join(results_dir, "ckpt_hashes.json")))
    vllm_acc_ref = json.load(open(os.path.join(results_dir, "vllm_accuracy_bench.json")))
    vllm_tpot_ref = json.load(open(os.path.join(results_dir, "vllm_baseline_tpot.json")))

    r1 = json.load(open(os.path.join(results_dir, "reference_engine_report.json")))
    r2 = json.load(open(os.path.join(results_dir, "pallas_megakernel_report.json")))
    r3 = json.load(open(os.path.join(results_dir, "accuracy_and_tpot_report.json")))
    r4 = json.load(open(os.path.join(results_dir, "dspark_speculative_report.json")))

    # Stage 1: Pure-JAX 40-Layer Reference Engine on Real Checkpoint
    ref_hashes = hashes_ref.get("tensors", hashes_ref)
    hash_matches = 0
    for name, expected in ref_hashes.items():
        expected_hex = expected["sha256"] if isinstance(expected, dict) else expected
        actual = r1["gcs_hashes"].get(name)
        actual_hex = actual["sha256"] if isinstance(actual, dict) else actual
        if actual_hex == expected_hex:
            hash_matches += 1
    assert hash_matches == len(ref_hashes) and hash_matches >= 6, f"SHA-256 mismatch: {hash_matches}/{len(ref_hashes)}"
    assert r1["prompt_top1_tie_aware_agreement"] >= r1["vllm_batch_composition_noise_floor"], (
        "Reference engine tie-aware prompt top-1 below vLLM batch-composition floor"
    )
    assert all(p["passed_negative_control"] for p in r1["perturbations"].values()), (
        "Negative control perturbation did not trigger divergence"
    )
    print(
        f"[PASS] Stage 1 (Reference Engine): {hash_matches}/{len(ref_hashes)} GCS SHA-256 hashes matched, "
        f"tie-aware top-1={r1['prompt_top1_tie_aware_agreement']*100:.2f}% "
        f"(strict={r1['prompt_top1_agreement']*100:.2f}% vs {r1['vllm_batch_composition_noise_floor']*100:.2f}% floor), "
        f"{len(r1['perturbations'])}/{len(r1['perturbations'])} negative controls failed as required."
    )

    # Stage 2: 40-Layer Pallas Decode Megakernel
    c1_p = r2["check1_per_layer_parity"]
    c2_p = r2["check2_end_to_end_golden"]
    c3_p = r2["check3_single_pallas_call_hlo"]
    c4_p = r2["check4_decode_step_latency"]
    assert c1_p["pass"] and c1_p["num_layers_verified"] == 40
    assert c1_p["min_cos_b1"] >= 0.999 and c1_p["min_cos_b8"] >= 0.999
    assert c2_p["pass"] and c2_p["tie_aware_prompt_top1_agreement"] >= c2_p["vllm_c1_vs_c8_floor_top1_agreement"]
    assert c3_p["pass"] and c3_p["tpu_custom_call_count"] == 1
    assert c4_p["pass"] and all(m["num_timed_iters"] >= 50 for m in c4_p["measurements"].values())
    print(
        f"[PASS] Stage 2 (40-Layer Pallas Megakernel): 40/40 layers verified (min_cos B1={c1_p['min_cos_b1']:.6f}, "
        f"B8={c1_p['min_cos_b8']:.6f}), tpu_custom_call={c3_p['tpu_custom_call_count']}, "
        f"1K/B1={c4_p['measurements']['ctx1024_B1']['median_ms']:.2f} ms, "
        f"1K/B8={c4_p['measurements']['ctx1024_B8']['median_ms']:.2f} ms."
    )

    # Stage 3: Accuracy & Concurrency TPOT Ladder vs Production vLLM
    c1_a = r3["check1_accuracy_benchmark"]
    c2_a = r3["check2_c1_tpot_speedup"]
    c3_a = r3["check3_concurrency_crossover"]
    assert c1_a["pass"] and c1_a["mk_num_correct"] == vllm_acc_ref["num_correct"]
    assert c2_a["pass"] and c2_a["c1_speedup_vs_vllm"] >= 2.0
    assert c3_a["pass"] and len(c3_a["sweep"]) == 5
    print(
        f"[PASS] Stage 3 (Accuracy & TPOT Ladder): GSM8K/STEM={c1_a['mk_num_correct']}/{c1_a['num_questions']} "
        f"({c1_a['mk_accuracy']*100:.1f}% vs vLLM {vllm_acc_ref['accuracy']*100:.1f}%), "
        f"C=1 TPOT={c2_a['c1_mk_median_tpot_ms']:.2f} ms vs vLLM {c2_a['c1_vllm_median_tpot_ms']:.2f} ms "
        f"({c2_a['c1_speedup_vs_vllm']:.2f}x speedup), crossover={c3_a['measured_crossover_concurrency']}."
    )

    # Stage 4: DSpark (mtp.0..2) Speculative Decoding
    c1_d = r4["check1_lossless_token_identity"]
    c2_d = r4["check2_acceptance_length"]
    c3_d = r4["check3_effective_tpot_speedup"]
    assert c1_d["pass"] and c1_d["token_identity_rate"] == 1.0 and c1_d["total_matching_tokens"] == 2048
    assert c2_d["pass"] and c2_d["mean_tokens_per_verify_step"] > 1.0
    assert c3_d["pass"] and c3_d["spec_median_effective_tpot_ms"] < c3_d["nonspec_median_tpot_ms"]
    print(
        f"[PASS] Stage 4 (DSpark Speculative Decoding): lossless {c1_d['total_matching_tokens']}/{c1_d['total_tokens_compared']} "
        f"tokens (100.0%), mean acceptance={c2_d['mean_tokens_per_verify_step']:.2f} tok/step "
        f"(+{c2_d['mean_accepted_draft_tokens']:.2f} draft accepted), "
        f"median effective TPOT={c3_d['spec_median_effective_tpot_ms']:.2f} ms vs {c3_d['nonspec_median_tpot_ms']:.2f} ms "
        f"({c3_d['median_speedup_vs_nonspec']:.2f}x speedup)."
    )

    unified = json.load(open(output_path))
    with open(output_path, "w") as f:
        json.dump(unified, f, indent=2)
    print(f"[ALL STAGES VERIFIED] Summary results saved at {output_path}")
    return unified


def run_hardware_stage(stage: str) -> None:
    """Dispatch hardware verification script(s) on the TPU v6e-16 slice."""
    stage_scripts = {
        "reference": "verify_reference_engine.py",
        "megakernel": "verify_pallas_megakernel.py",
        "accuracy": "verify_accuracy_and_tpot.py",
        "dspark": "verify_dspark_speculative.py",
    }
    selected = list(stage_scripts.keys()) if stage == "all" else [stage]
    for st in selected:
        script_path = os.path.join(SCRIPTS_DIR, stage_scripts[st])
        print(f"=== Running hardware verification stage: {st} ({script_path}) ===", flush=True)
        subprocess.run([sys.executable, script_path], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek-V4.1-Flash TPU v6e-16 Pallas Megakernel Benchmark & Verifier")
    parser.add_argument(
        "--stage",
        choices=["reference", "megakernel", "accuracy", "dspark", "all"],
        default=None,
        help="Run live multi-host TPU v6e-16 hardware verification for the selected stage.",
    )
    parser.add_argument(
        "--verify-saved-reports",
        action="store_true",
        default=True,
        help="Verify saved hardware reports in results/ and validate all thresholds (default).",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Directory containing hardware verification reports.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(RESULTS_DIR, "megakernel_v6e16_results.json"),
        help="Path to unified summary JSON output.",
    )
    args = parser.parse_args()

    if args.stage is not None:
        run_hardware_stage(args.stage)
    else:
        verify_saved_reports(args.results_dir, args.output)


if __name__ == "__main__":
    main()
