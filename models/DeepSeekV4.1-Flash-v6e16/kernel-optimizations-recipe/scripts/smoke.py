#!/usr/bin/env python3
"""Send a few short questions to a live vLLM endpoint and check the answers.

This is the first check after a serving revision starts. It answers two
questions in order:

1. Does the model emit real text, or the repeated-token output that a NaN in
   the forward pass produces?
2. Does it get a known answer right? A forward pass can stay finite and still
   be wrong, so a liveness check alone proves very little.

The requests go to ``/v1/chat/completions``, because this model is a reasoning
model and expects its own chat template. The raw ``/v1/completions`` path gives
base-model continuation, which loops at temperature 0 even when the model is
healthy. A raw-path sample is printed at the end for information only.

The script uses only the standard library, so it runs inside the serving image
without an extra install. Run it from the head pod, where the endpoint is on
localhost.

Usage: smoke_v41.py [model-id]
"""
import json
import re
import sys
import urllib.request

BASE = "http://localhost:8000"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "deepseek-ai/DeepSeek-V4.1-Flash"

# Each case is a question and a pattern its answer must contain.
CASES = [
    ("What is the capital of France? Answer in one word.", r"\bParis\b"),
    ("What is 17 * 23? Reply with the number only.", r"\b391\b"),
    ("Write a Python function `add` that returns the sum of a and b. "
     "Reply with code only.", r"a\s*\+\s*b"),
    ("Which planet is closest to the Sun? Answer in one word.",
     r"\bMercury\b"),
]


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(question, max_tokens=512):
    out = post("/v1/chat/completions",
               {"model": MODEL,
                "messages": [{"role": "user", "content": question}],
                "max_tokens": max_tokens, "temperature": 0.0})
    choice = out["choices"][0]
    message = choice.get("message", {})
    # The reasoning parser is not enabled here, so the chain of thought arrives
    # inside `content` ahead of a `</think>` marker.
    text = (message.get("reasoning_content") or "") + (message.get("content")
                                                       or "")
    final = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    return final, choice.get("finish_reason")


def degenerate(text):
    """True when the text repeats one token, the signature of a NaN."""
    words = text.split()
    return len(words) >= 8 and len(set(words)) == 1


def main():
    bad = 0
    for question, pattern in CASES:
        try:
            final, finish = ask(question)
        except Exception as exc:                    # noqa: BLE001
            print(f"FAIL  {question!r}: {type(exc).__name__}: {exc}")
            bad += 1
            continue
        if degenerate(final):
            flag = "DEGENERATE"
        elif not final.strip():
            flag = "EMPTY"
        elif not re.search(pattern, final, re.IGNORECASE):
            flag = "WRONG"
        else:
            flag = "ok"
        if flag != "ok":
            bad += 1
        print(f"[{flag:10s}] {question}")
        print(f"             -> {final.strip()[:200]!r}  ({finish})")

    print(f"\n{len(CASES) - bad}/{len(CASES)} answers correct")

    # Information only. A looping continuation here is normal for a reasoning
    # model given a bare prompt, and it is not a fault.
    try:
        raw = post("/v1/completions",
                   {"model": MODEL, "prompt": "def add(a, b):",
                    "max_tokens": 24, "temperature": 0.0})
        print(f"\nraw completion (informational): "
              f"{raw['choices'][0]['text']!r}")
    except Exception as exc:                        # noqa: BLE001
        print(f"\nraw completion probe failed: {type(exc).__name__}: {exc}")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
