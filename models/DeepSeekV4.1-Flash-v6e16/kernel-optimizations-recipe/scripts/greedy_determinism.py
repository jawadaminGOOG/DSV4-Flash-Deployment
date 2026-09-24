#!/usr/bin/env python3
"""Greedy determinism over 200 prompts against a live vLLM endpoint.

A wrong KV slot, a stale cache row, or a race in the attention kernel changes
the output without raising an error. Temperature 0 removes sampling noise, so
two runs of the same prompt set must return the same bytes. Any difference is a
real defect.

The script runs the set twice at the same concurrency and compares those two.
That is the gating comparison. It then runs the set a third time at a different
concurrency and reports that comparison as information only, because vLLM
batches continuously and a different batch composition can change the reduction
order inside a kernel. A batch-composition difference is worth seeing; it is
not by itself a defect.

Half the prompts are real English and half are a repeating synthetic pattern.
The real prompts catch numerical drift. The pattern prompts catch a wrong KV
slot, because the model can only continue the cycle if attention reads the
right cache rows.

Usage: greedy200.py <out.json> [n_prompts] [concurrency] [alt_concurrency]
"""
import asyncio
import json
import os
import sys
import time

import aiohttp

BASE = "http://localhost:8000"
MODEL = os.environ.get("GREEDY_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
MAX_TOKENS = 128

# CAUTION: keep every real prompt open-ended, and never make prompts unique by
# appending a suffix such as "(variant 7)". A prompt that ends in a full stop
# makes a base model emit an end-of-sequence token at once, which looks like an
# empty completion but is correct. A repeated suffix makes the model continue
# that suffix, which looks like the repeated-token NaN signature but is also
# correct. Both defects appeared in an earlier version of this script and both
# were the script's fault, not the model's.
#
# The real prompts are built combinatorially from a subject and an opening, so
# 100 of them are distinct without any suffix.
SUBJECTS = [
    "the water cycle", "the Roman Republic", "a binary search tree",
    "photosynthesis", "the Doppler effect", "the printing press",
    "plate tectonics", "a relational database index", "the Silk Road",
    "protein folding",
]
OPENINGS = [
    "The main idea behind {s} is",
    "One reason {s} matters is that",
    "A short description of {s} follows. {S}",
    "Students often ask about {s}. The answer is that",
    "Compared with other topics, {s} is unusual because",
    "The history of {s} begins when",
    "To understand {s}, first consider",
    "A common mistake about {s} is",
    "In practice, {s} depends on",
    "The simplest example of {s} is",
]

PROMPT_WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india "
                "juliet kilo lima mike november oscar papa").split()

def make_prompt(seed, n_words):
    out, x = [], seed * 2654435761 % 2**32
    for _ in range(n_words):
        x = (x * 1103515245 + 12345) % 2**31
        out.append(PROMPT_WORDS[x % len(PROMPT_WORDS)])
    return " ".join(out)


def build(n):
    """Return n prompts: half real English, half the synthetic pattern."""
    half = n // 2
    real = []
    for i in range(half):
        subject = SUBJECTS[i % len(SUBJECTS)]
        opening = OPENINGS[(i // len(SUBJECTS)) % len(OPENINGS)]
        real.append(opening.format(s=subject,
                                   S=subject[0].upper() + subject[1:]))
    synth = [make_prompt(i, 48 + (i % 17)) for i in range(n - half)]
    return real + synth


async def one(session, prompt):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": MAX_TOKENS,
            "temperature": 0.0}
    async with session.post(f"{BASE}/v1/completions", json=body) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:200]}")
        out = await r.json()
    return out["choices"][0]["text"]


async def run_pass(prompts, concurrency, label):
    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency + 8)
    timeout = aiohttp.ClientTimeout(total=7200)

    async def guarded(session, p):
        async with sem:
            return await one(session, p)

    t0 = time.perf_counter()
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
        res = await asyncio.gather(*(guarded(s, p) for p in prompts),
                                   return_exceptions=True)
    wall = time.perf_counter() - t0
    ok = sum(1 for r in res if isinstance(r, str))
    print(f"{label}: {ok}/{len(res)} completed in {wall:.1f} s", flush=True)
    return [r if isinstance(r, str) else f"ERROR {type(r).__name__}: {r}"
            for r in res]


def compare(a, b, prompts, label):
    diffs = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        n = min(len(x), len(y))
        at = next((k for k in range(n) if x[k] != y[k]), n)
        diffs.append({"i": i, "char": at, "prompt": prompts[i][:80],
                      "a": x[max(0, at - 40):at + 40],
                      "b": y[max(0, at - 40):at + 40]})
    same = len(a) - len(diffs)
    print(f"{label}: {same}/{len(a)} byte-identical, {len(diffs)} differ",
          flush=True)
    for d in diffs[:5]:
        print(f"  [{d['i']}] at char {d['char']}: {d['a']!r} vs {d['b']!r}")
    return {"identical": same, "total": len(a), "diffs": diffs[:20]}


def degenerate(text):
    """True when the text is one token repeated, the NaN signature."""
    words = text.split()
    return len(words) >= 8 and len(set(words)) == 1


async def main():
    out_path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    conc = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    alt = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    prompts = build(n)
    print(f"{len(prompts)} prompts, model={MODEL}, max_tokens={MAX_TOKENS}",
          flush=True)

    run_a = await run_pass(prompts, conc, f"run A (C={conc})")
    run_b = await run_pass(prompts, conc, f"run B (C={conc})")
    run_c = await run_pass(prompts, alt, f"run C (C={alt})")

    gating = compare(run_a, run_b, prompts, f"GATING  A vs B (both C={conc})")
    info = compare(run_a, run_c, prompts, f"INFO    A vs C (C={conc} vs {alt})")

    errors = sum(1 for t in run_a if t.startswith("ERROR "))
    empty = sum(1 for t in run_a if not t.strip())
    degen = sum(1 for t in run_a if degenerate(t))

    summary = {"model": MODEL, "n_prompts": len(prompts),
               "concurrency": conc, "alt_concurrency": alt,
               "gating_identical": gating["identical"],
               "info_identical": info["identical"],
               "errors": errors, "empty": empty, "degenerate": degen}
    with open(out_path, "w") as fh:
        json.dump({"summary": summary, "gating": gating, "info": info,
                   "run_a": run_a, "run_b": run_b, "run_c": run_c,
                   "prompts": prompts}, fh, indent=1)

    print(json.dumps(summary, indent=1))
    verdict = ("PASS" if gating["identical"] == len(prompts) and not errors
               and not empty and not degen else "FAIL")
    print(f"\ngreedy determinism: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
