#!/usr/bin/env python3
"""Concurrency sweep over three workload shapes against a live vLLM endpoint.

This extends the single-shape benchmark client, which measures one shape at six
concurrency levels. The goal needs three shapes at ten levels, from C=1 to
C=512, in the format of the sglang-rtx-pro-6000 DeepSeek-V4-Flash report.

The request path, the calibration and the summary statistics are kept identical
to `bench.py` on purpose, so that the `1k/1k` rows here are comparable to every
`1k/1k` number already in the record. Three things are added: a per-shape ISL
target, the mean TTFT and the request rate, which the reference report needs.

Each point writes its JSON as soon as it completes. The reasoning shape takes
about two hours, and a crash at the end must not lose the start.

Usage: sweep3.py <shape> <out.json> [concurrencies...]
       shape is one of: 1k1k, 8k1k, 1k8k
"""
import asyncio
import json
import os
import statistics
import sys
import time

import aiohttp

BASE = "http://localhost:8000"
# Set SWEEP_MODEL to point the same request path at another served model id.
MODEL = os.environ.get("SWEEP_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
NUM_CHIPS = 16

# The server enforces prompt_tokens + max_tokens <= MAX_MODEL_LEN, inclusive.
# 9216 is exactly 1024 + 8192, so both 8k shapes fit it exactly. A server that
# serves a shorter context needs SWEEP_MAX_MODEL_LEN, which only changes the
# client-side shape check, not the request path.
MAX_MODEL_LEN = int(os.environ.get("SWEEP_MAX_MODEL_LEN", "9216"))

DEFAULT_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Request count per point, and the ISL/OSL, fixed in review.md F3 before any
# measurement. The reasoning shape gets one request at C=1 because one request
# already emits 8192 tokens, about twelve minutes.
SHAPES = {
    "1k1k": {"isl": 1024, "osl": 1024, "min_reqs": 4,
             "label": "1k/1k (balanced)"},
    "8k1k": {"isl": 8192, "osl": 1024, "min_reqs": 4,
             "label": "8k/1k (prefill-heavy)"},
    "1k8k": {"isl": 1024, "osl": 8192, "min_reqs": 1,
             "label": "1k/8k (reasoning)"},
}

# A deterministic pseudo-prompt, built from words. One word is not one token:
# this list averages about 1.44 tokens per word, so a prompt of N words is much
# longer than N tokens. `calibrate` finds the word count that hits the target.
PROMPT_WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india "
                "juliet kilo lima mike november oscar papa").split()


def make_prompt(seed, n_words):
    out, x = [], seed * 2654435761 % 2**32
    for _ in range(n_words):
        x = (x * 1103515245 + 12345) % 2**31
        out.append(PROMPT_WORDS[x % len(PROMPT_WORDS)])
    return " ".join(out)


async def count_tokens(session, prompt):
    async with session.post(f"{BASE}/tokenize",
                            json={"model": MODEL, "prompt": prompt}) as r:
        r.raise_for_status()
        return (await r.json())["count"]


async def calibrate(session, n_prompts, target_isl):
    """Pick the word count whose longest prompt still fits target_isl tokens.

    Different seeds pick different words, so token count varies a little across
    prompts. Size against the longest one, not the first, or a single long
    prompt in the batch fails the whole run with a 400.
    """
    lo, hi = 1, target_isl
    while lo < hi:                                  # largest n_words with
        mid = (lo + hi + 1) // 2                    # tokens(seed 0) <= target
        if await count_tokens(session, make_prompt(0, mid)) <= target_isl:
            lo = mid
        else:
            hi = mid - 1
    n_words = lo
    counts = await asyncio.gather(*(count_tokens(session, make_prompt(i, n_words))
                                    for i in range(n_prompts)))
    while max(counts) > target_isl and n_words > 1:
        n_words -= 1
        counts = await asyncio.gather(*(count_tokens(session, make_prompt(i, n_words))
                                        for i in range(n_prompts)))
    return n_words, min(counts), max(counts)


async def one(session, prompt, max_tokens, temperature, ignore_eos):
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature, "stream": True,
            "ignore_eos": ignore_eos}
    async with session.post(f"{BASE}/v1/completions", json=body) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:300]}")
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)["choices"][0].get("text", "")
            if chunk:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                ntok += 1
    return {"ttft_s": ttft, "total_s": time.perf_counter() - t0, "ntok": ntok}


async def run_batch(prompts, concurrency, max_tokens, temperature,
                    ignore_eos):
    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency + 8)
    # The reasoning shape's C=512 point moves 4.2 million tokens. One hour is
    # not enough for it, so the ceiling is four.
    timeout = aiohttp.ClientTimeout(total=14400)

    async def guarded(session, p):
        async with sem:
            return await one(session, p, max_tokens, temperature, ignore_eos)

    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
        t0 = time.perf_counter()
        res = await asyncio.gather(*(guarded(s, p) for p in prompts),
                                   return_exceptions=True)
        return res, time.perf_counter() - t0


def summarise(res, wall, concurrency):
    ok = [r for r in res if isinstance(r, dict)]
    err = [r for r in res if not isinstance(r, dict)]
    ntok = sum(r["ntok"] for r in ok)
    ttfts = sorted(r["ttft_s"] * 1000 for r in ok if r["ttft_s"] is not None)
    tpots = [(r["total_s"] - (r["ttft_s"] or 0)) / max(r["ntok"] - 1, 1) * 1000
             for r in ok if r["ntok"] > 1]

    def pct(xs, q):
        return xs[min(int(q * len(xs)), len(xs) - 1)] if xs else None

    return {"concurrency": concurrency, "num_prompts": len(res),
            "completed": len(ok), "failed": len(err),
            "success_rate_pct": 100.0 * len(ok) / len(res) if res else 0.0,
            # Keep the first error. A run that fails must say why in the record,
            # not just report zero throughput.
            "first_error": f"{type(err[0]).__name__}: {err[0]}" if err else None,
            "wall_time_s": wall, "output_tokens": ntok,
            "output_throughput_tok_s": ntok / wall,
            "output_tok_s_per_chip": ntok / wall / NUM_CHIPS,
            "requests_per_s": len(ok) / wall,
            "ttft_mean_ms": statistics.mean(ttfts) if ttfts else None,
            "ttft_p50_ms": pct(ttfts, 0.50), "ttft_p90_ms": pct(ttfts, 0.90),
            "tpot_mean_ms": statistics.mean(tpots) if tpots else None}


def fmt(x, spec):
    return format(x, spec) if x is not None else "n/a"


async def sweep(shape_key, out_path, concurrencies):
    shape = SHAPES[shape_key]
    target_isl, osl, min_reqs = shape["isl"], shape["osl"], shape["min_reqs"]

    n_prompts = max(max(concurrencies), min_reqs)
    async with aiohttp.ClientSession() as s:
        n_words, isl_min, isl_max = await calibrate(s, n_prompts, target_isl)

    if isl_max + osl > MAX_MODEL_LEN:
        sys.exit(f"shape does not fit: isl_max {isl_max} + osl {osl} > "
                 f"{MAX_MODEL_LEN}")

    print(f"shape {shape['label']}: {n_words} words -> ISL "
          f"{isl_min}..{isl_max} tokens, OSL {osl}", flush=True)
    prompts = [make_prompt(i, n_words) for i in range(n_prompts)]

    rows = []
    for c in concurrencies:
        n = max(c, min_reqs)
        t_start = time.time()
        res, wall = await run_batch(prompts[:n], c, osl, 0.0, True)
        row = summarise(res, wall, c)
        row.update(shape=shape_key, label=shape["label"], isl_min=isl_min,
                   isl_max=isl_max, osl=osl, n_words=n_words,
                   requests_issued=n, started_at=t_start)
        rows.append(row)
        print(f"C={c:4d} n={n:4d}  {row['output_throughput_tok_s']:9.1f} tok/s  "
              f"{row['requests_per_s']:7.3f} req/s  "
              f"{row['completed']}/{row['num_prompts']} ok  "
              f"ttft_mean {fmt(row['ttft_mean_ms'], '.0f')} ms  "
              f"tpot {fmt(row['tpot_mean_ms'], '.1f')} ms  "
              f"[{wall:.0f}s]", flush=True)
        if row["first_error"]:
            print(f"        first error: {row['first_error']}", flush=True)
        # Write after every point. The reasoning shape runs for about two
        # hours, and a crash at the end must not lose the start.
        json.dump(rows, open(out_path, "w"), indent=1)
    return rows


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in SHAPES:
        sys.exit(f"usage: sweep3.py <{'|'.join(SHAPES)}> <out.json> [C...]")
    shape_key, out_path = sys.argv[1], sys.argv[2]
    cs = [int(x) for x in sys.argv[3:]] or DEFAULT_CONCURRENCIES
    asyncio.run(sweep(shape_key, out_path, cs))


if __name__ == "__main__":
    main()
