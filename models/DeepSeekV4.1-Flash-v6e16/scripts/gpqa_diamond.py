#!/usr/bin/env python3
"""GPQA Diamond Pass@1 against a live vLLM endpoint.

The prompt template and the scoring rule follow openai/simple-evals, so the
result is comparable to a published GPQA Diamond number. The differences from
simple-evals are deliberate and listed here:

* The four answers are permuted once per question with a fixed seed, not ten
  times. Ten permutations cost ten times the tokens and this endpoint serves
  one model on one slice.
* The request goes to ``/v1/chat/completions``, so the model receives its own
  chat template. The raw ``/v1/completions`` path gives base-model
  continuation, which loops at temperature 0 and does not answer the question.
* The answer letter comes from the LAST ``Answer: X`` in the completion, and
  only from the text after ``</think>`` when that marker is present. A
  reasoning model states a candidate answer inside its chain of thought, so the
  first match is often wrong even when the final answer is right.
* A completion that hits the token limit without an ``Answer:`` line scores as
  wrong, and the script counts those separately. A low score with a high
  truncation count means the context limit, not the model, set the result.

The script reads the CSV from a local path. Fetch it once with:

    gcloud storage cp gs://<bucket>/v41-stack/gpqa_diamond.csv /tmp/

Usage: gpqa_diamond.py <csv> <out.json> [concurrency] [max_tokens]
"""
import asyncio
import csv
import json
import os
import random
import re
import statistics
import sys
import time

import aiohttp

BASE = "http://localhost:8000"
MODEL = os.environ.get("GPQA_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
SEED = 0

TEMPLATE = """Answer the following multiple choice question. The last line of \
your response should be of the following format: 'Answer: $LETTER' (without \
quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {a}
B) {b}
C) {c}
D) {d}
"""

ANSWER_RE = re.compile(r"Answer\s*:\s*\**\s*([ABCD])\b")
# A fallback for a final answer given as a bare letter, e.g. "**C**" or "C)".
BARE_RE = re.compile(r"(?:^|\n)\s*\**\s*([ABCD])\s*[\).:]?\**\s*$")


def extract(text):
    """Return the answer letter the model settled on.

    The text after ``</think>`` is the model's final answer, so it wins over
    anything said inside the chain of thought. Within that final text an
    explicit ``Answer: X`` wins over a bare letter. Only when the final text
    holds no letter at all does the chain of thought decide.
    """
    final = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    matches = ANSWER_RE.findall(final)
    if matches:
        return matches[-1]
    match = BARE_RE.search(final.strip())
    if match:
        return match.group(1)
    matches = ANSWER_RE.findall(text)
    return matches[-1] if matches else None


def build(rows):
    """Return one prompt and one correct letter for each question."""
    rng = random.Random(SEED)
    items = []
    for i, row in enumerate(rows):
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        shuffled = [choices[k] for k in order]
        prompt = TEMPLATE.format(question=row["Question"], a=shuffled[0],
                                 b=shuffled[1], c=shuffled[2], d=shuffled[3])
        items.append({"i": i, "prompt": prompt,
                      "correct": "ABCD"[order.index(0)]})
    return items


async def one(session, item, max_tokens):
    t0 = time.perf_counter()
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": max_tokens, "temperature": 0.0}
    async with session.post(f"{BASE}/v1/chat/completions", json=body) as r:
        if r.status != 200:
            text = (await r.text())[:300]
            return {"i": item["i"], "correct": item["correct"],
                    "error": f"HTTP {r.status}: {text}"}
        out = await r.json()
    choice = out["choices"][0]
    message = choice.get("message", {})
    # The reasoning parser is not enabled on this endpoint, so the chain of
    # thought arrives inside `content` ahead of a `</think>` marker. Join both
    # fields so the extractor sees the same text either way.
    text = (message.get("reasoning_content") or "") + (message.get("content")
                                                       or "")
    return {"i": item["i"], "correct": item["correct"],
            "picked": extract(text),
            "finish_reason": choice.get("finish_reason"),
            "ntok": out.get("usage", {}).get("completion_tokens"),
            "latency_s": time.perf_counter() - t0,
            "text_tail": text[-400:]}


async def main():
    csv_path, out_path = sys.argv[1], sys.argv[2]
    concurrency = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 1536

    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    items = build(rows)
    print(f"{len(items)} questions, concurrency={concurrency}, "
          f"max_tokens={max_tokens}, model={MODEL}", flush=True)

    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency + 8)
    timeout = aiohttp.ClientTimeout(total=14400)
    done = [0]

    async def guarded(session, item):
        async with sem:
            res = await one(session, item, max_tokens)
        done[0] += 1
        if done[0] % 10 == 0:
            print(f"  {done[0]}/{len(items)}", flush=True)
        return res

    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
        results = await asyncio.gather(*(guarded(s, it) for it in items))

    errors = [r for r in results if r.get("error")]
    scored = [r for r in results if not r.get("error")]
    right = sum(1 for r in scored if r["picked"] == r["correct"])
    unparsed = sum(1 for r in scored if r["picked"] is None)
    truncated = sum(1 for r in scored if r["finish_reason"] == "length")
    toks = [r["ntok"] for r in scored if r["ntok"]]

    summary = {
        "model": MODEL,
        "questions": len(items),
        "scored": len(scored),
        "errors": len(errors),
        "correct": right,
        "pass_at_1": right / len(items) * 100.0,
        "unparsed": unparsed,
        "truncated": truncated,
        "max_tokens": max_tokens,
        "concurrency": concurrency,
        "median_completion_tokens": statistics.median(toks) if toks else None,
    }
    with open(out_path, "w") as fh:
        json.dump({"summary": summary, "results": results}, fh, indent=1)

    print(json.dumps(summary, indent=1))
    print(f"\nPass@1 = {summary['pass_at_1']:.1f}%  "
          f"({right}/{len(items)}), truncated={truncated}, "
          f"unparsed={unparsed}, errors={len(errors)}")


if __name__ == "__main__":
    asyncio.run(main())
