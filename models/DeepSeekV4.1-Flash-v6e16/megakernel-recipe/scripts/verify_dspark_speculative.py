#!/usr/bin/env python3
"""DSpark (mtp.0..2) Speculative Decoding Verification: Real DSpark (`mtp.0..2`) Speculative Decoding on TPU v6e-16.

Verifies DSpark speculative decoding:
  Check 1 (Lossless): Speculative greedy output is token-for-token identical to non-speculative
                       greedy output on all 8 golden prompts (`8 x 256 = 2,048` tokens).
  Check 2 (Acceptance length): Mean acceptance length (`mean_accepted_draft_tokens` and
                       `mean_tokens_per_verify_step`) is measured on the 8 real golden prompts.
  Check 3 (Speedup):  Effective C=1 decode TPOT with DSpark speculative decoding is lower than
                       non-speculative C=1 decode TPOT from timed runs.
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List

import jax
import ml_dtypes
import numpy as np
from transformers import AutoTokenizer

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_compilation_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

REPO_MK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists("/workspace/megakernel"):
    sys.path.insert(0, "/workspace")
else:
    sys.path.insert(0, REPO_MK)

from megakernel.decode_megakernel import (
    DSV41PallasMegakernel,
    pack_megakernel_weights,
)
from megakernel.dspark import DSparkDraftEngine, load_dspark_weights
from megakernel.load import CACHE_DIR_DEFAULT, load_dsv41_weights


def warmup_engines(
    mk: DSV41PallasMegakernel,
    dspark_engine: DSparkDraftEngine,
    max_comp: int = 1024,
    spec_k: int = 5,
) -> None:
    """Compile and warm up `(active_b=0, max_comp=max_comp, spec_k=spec_k)` and `DSparkDraftEngine`."""
    state = mk.init_decode_state(max_comp=max_comp, enable_engram=False)
    # 1. Warm up 1-token step + 5-token speculative verify step (shares the same compiled Pallas kernel)
    _, _, dspark_h, state = mk.verify_speculative_step(
        [1280], state, enable_engram=False, spec_k=spec_k
    )
    _, _, dspark_h, state = mk.verify_speculative_step(
        [1280, 2049, 3141, 4099, 5123], state, enable_engram=False, spec_k=spec_k
    )
    # 2. Warm up DSpark KV update + 3-stage MTP draft_block
    d_kv, d_pos = dspark_engine.init_kv_state()
    mh_dummy = np.zeros((1, 15360), dtype=ml_dtypes.bfloat16)
    d_kv, d_pos = dspark_engine.update_kv_cache(
        mh_dummy, np.array([0], dtype=np.int32), d_kv, d_pos
    )
    _, d_kv, d_pos = dspark_engine.draft_block_on_device(
        root_token_id=1280,
        dspark_h=dspark_h,
        prev_n_acc=1,
        prev_p0=1,
        swa_kv=d_kv,
        swa_pos=d_pos,
    )


def run_nonspec_greedy_timed(
    mk: DSV41PallasMegakernel,
    prompt_ids_np: np.ndarray,
    max_new_tokens: int = 256,
    max_comp: int = 1024,
    spec_k: int = 5,
) -> Dict[str, Any]:
    """Run non-speculative greedy decoding (1 token per step) and time the post-prefill decode loop."""
    state = mk.init_decode_state(max_comp=max_comp, enable_engram=False)
    first_tok = 0
    for tid in prompt_ids_np:
        emitted, _, _, state = mk.verify_speculative_step(
            [int(tid)], state, enable_engram=False, spec_k=spec_k
        )
        first_tok = emitted[0]

    generated: List[int] = [first_tok]
    step_times_ms: List[float] = []
    t0_dec = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        t0_s = time.perf_counter()
        emitted, _, _, state = mk.verify_speculative_step(
            [generated[-1]], state, enable_engram=False, spec_k=spec_k
        )
        step_times_ms.append((time.perf_counter() - t0_s) * 1000.0)
        generated.append(emitted[0])
    decode_elapsed_ms = (time.perf_counter() - t0_dec) * 1000.0

    return {
        "generated_tokens": generated,
        "decode_elapsed_ms": float(decode_elapsed_ms),
        "median_step_tpot_ms": float(np.median(step_times_ms)),
        "mean_tpot_ms": float(decode_elapsed_ms / max(1, len(generated) - 1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--golden-json", default="/tmp/vllm_golden.json")
    parser.add_argument("--output-json", default="/tmp/dspark_speculative_report.json")
    parser.add_argument("--max-comp", type=int, default=512)
    parser.add_argument("--spec-k", type=int, default=5)
    parser.add_argument("--num-draft", type=int, default=4)
    args = parser.parse_args()

    os.makedirs("/tmp/jax_compilation_cache", exist_ok=True)
    jax.distributed.initialize()
    rank = jax.process_index()

    with open(args.golden_json, "r") as f:
        golden_data = json.load(f)

    t0 = time.perf_counter()
    mw = load_dsv41_weights(cache_dir=args.cache_dir)
    tok = AutoTokenizer.from_pretrained(
        os.path.join(args.cache_dir, "tokenizer"), trust_remote_code=True
    )
    dspark_w = load_dspark_weights(
        cfg=mw.config,
        mesh=mw.mesh,
        embed_weight=mw.embed_weight,
        head_weight=mw.head_weight,
        cache_dir=args.cache_dir,
    )
    dspark_engine = DSparkDraftEngine(dspark_w, max_seq_len=4096)
    pmw = pack_megakernel_weights(mw, batch_tile=8, free_source_layers=True)
    del mw
    gc.collect()
    mk = DSV41PallasMegakernel(pmw, batch_tile=8)
    if rank == 0:
        print(
            f"Loaded 40-layer Megakernel + 3-layer DSpark weights in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    t_w = time.perf_counter()
    warmup_engines(mk, dspark_engine, max_comp=args.max_comp, spec_k=args.spec_k)
    if rank == 0:
        print(
            f"Compiled & warmed up speculative megakernel + DSpark draft engine in {time.perf_counter() - t_w:.2f}s",
            flush=True,
        )

    prompts = golden_data["run_a_c1"]
    prompt_reports: List[Dict[str, Any]] = []
    exact_matches = 0
    total_tokens_compared = 0
    total_matching_tokens = 0
    all_accepted_counts: List[int] = []
    nonspec_tpots: List[float] = []
    spec_tpots: List[float] = []

    for p_entry in prompts:
        pid = p_entry["id"]
        prompt_ids = np.array(
            tok.encode(p_entry["prompt"], add_special_tokens=False), dtype=np.int32
        )
        n_gen = 256

        nonspec_res = run_nonspec_greedy_timed(
            mk,
            prompt_ids,
            max_new_tokens=n_gen,
            max_comp=args.max_comp,
            spec_k=args.spec_k,
        )
        spec_tokens, spec_stats, _ = mk.generate_speculative_greedy(
            prompt_ids,
            dspark_engine,
            max_new_tokens=n_gen,
            max_comp=args.max_comp,
            enable_engram=False,
            spec_k=args.spec_k,
            num_draft=args.num_draft,
        )

        nonspec_tokens = nonspec_res["generated_tokens"]
        is_exact = spec_tokens == nonspec_tokens
        exact_matches += int(is_exact)
        n_match = sum(
            1 for a, b in zip(spec_tokens, nonspec_tokens) if int(a) == int(b)
        )
        total_tokens_compared += n_gen
        total_matching_tokens += n_match

        first_div = None
        for idx_t, (a, b) in enumerate(zip(spec_tokens, nonspec_tokens)):
            if int(a) != int(b):
                first_div = idx_t
                break

        all_accepted_counts.extend(spec_stats["accepted_draft_counts"])
        ns_tpot = float(nonspec_res["mean_tpot_ms"])
        sp_tpot = float(spec_stats["effective_tpot_ms"])
        speedup = float(ns_tpot / sp_tpot)
        nonspec_tpots.append(ns_tpot)
        spec_tpots.append(sp_tpot)

        p_rep = {
            "id": pid,
            "prompt_len": int(len(prompt_ids)),
            "max_new_tokens": int(n_gen),
            "exact_token_match": bool(is_exact),
            "matching_tokens": int(n_match),
            "first_divergence_step": first_div,
            "num_verify_steps": int(spec_stats["num_verify_steps"]),
            "mean_accepted_draft_tokens": float(
                spec_stats["mean_accepted_draft_tokens"]
            ),
            "mean_tokens_per_verify_step": float(
                spec_stats["mean_tokens_per_verify_step"]
            ),
            "median_draft_ms": float(spec_stats["median_draft_ms"]),
            "median_verify_ms": float(spec_stats["median_verify_ms"]),
            "nonspec_mean_tpot_ms": ns_tpot,
            "nonspec_median_step_ms": float(nonspec_res["median_step_tpot_ms"]),
            "spec_effective_tpot_ms": sp_tpot,
            "speedup_vs_nonspec": speedup,
            "generated_tokens_head32": spec_tokens[:32],
        }
        prompt_reports.append(p_rep)
        if rank == 0:
            print(
                f"[Stage 4 DSpark] {pid}: exact_match={is_exact} ({n_match}/{n_gen}) | "
                f"steps={spec_stats['num_verify_steps']} "
                f"accept_len={spec_stats['mean_tokens_per_verify_step']:.2f} tok/step "
                f"(+{spec_stats['mean_accepted_draft_tokens']:.2f} draft) | "
                f"TPOT: spec={sp_tpot:.2f} ms vs nonspec={ns_tpot:.2f} ms "
                f"(speedup={speedup:.2f}x, draft={spec_stats['median_draft_ms']:.2f}ms, "
                f"verify={spec_stats['median_verify_ms']:.2f}ms)",
                flush=True,
            )

    check1_pass = exact_matches == len(prompts)
    mean_acc_draft = float(np.mean(all_accepted_counts))
    mean_tok_per_step = float(1.0 + mean_acc_draft)
    check2_pass = mean_tok_per_step > 1.0

    med_nonspec_tpot = float(np.median(nonspec_tpots))
    med_spec_tpot = float(np.median(spec_tpots))
    mean_nonspec_tpot = float(np.mean(nonspec_tpots))
    mean_spec_tpot = float(np.mean(spec_tpots))
    overall_speedup = float(med_nonspec_tpot / med_spec_tpot)
    check3_pass = med_spec_tpot < med_nonspec_tpot

    overall_pass = bool(check1_pass and check2_pass and check3_pass)
    report = {
        "benchmark_stage": "Stage 4 — DSpark speculative decoding (real draft head)",
        "overall_pass": overall_pass,
        "check1_lossless_token_identity": {
            "pass": bool(check1_pass),
            "num_prompts": len(prompts),
            "exact_prompt_matches": int(exact_matches),
            "total_tokens_compared": int(total_tokens_compared),
            "total_matching_tokens": int(total_matching_tokens),
            "token_identity_rate": float(
                total_matching_tokens / max(1, total_tokens_compared)
            ),
        },
        "check2_acceptance_length": {
            "pass": bool(check2_pass),
            "num_draft_proposed_per_step": int(args.num_draft),
            "total_verify_steps": int(len(all_accepted_counts)),
            "mean_accepted_draft_tokens": mean_acc_draft,
            "mean_tokens_per_verify_step": mean_tok_per_step,
            "acceptance_rate_per_draft_token": float(
                mean_acc_draft / max(1, args.num_draft)
            ),
        },
        "check3_effective_tpot_speedup": {
            "pass": bool(check3_pass),
            "nonspec_median_tpot_ms": med_nonspec_tpot,
            "spec_median_effective_tpot_ms": med_spec_tpot,
            "nonspec_mean_tpot_ms": mean_nonspec_tpot,
            "spec_mean_effective_tpot_ms": mean_spec_tpot,
            "median_speedup_vs_nonspec": overall_speedup,
            "mean_speedup_vs_nonspec": float(mean_nonspec_tpot / mean_spec_tpot),
        },
        "prompts": prompt_reports,
    }

    if rank == 0:
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(
            f"\n=== DSPARK SPECULATIVE REPORT SAVED TO {args.output_json} (OVERALL_PASS={overall_pass}) ===",
            flush=True,
        )


if __name__ == "__main__":
    main()
