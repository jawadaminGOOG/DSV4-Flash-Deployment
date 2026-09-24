#!/usr/bin/env python3
"""Parse XProf trace.json.gz to extract per-step MoE, ICI collective, and attention durations."""

import argparse
import gzip
import json
from pathlib import Path


def analyze_trace(trace_path: Path) -> dict[str, float]:
  with gzip.open(trace_path, "rt", encoding="utf-8") as f:
    data = json.load(f)
  events = data.get("traceEvents", [])
  totals_us = {
      "gmm_moe": 0.0,
      "ici_collectives": 0.0,
      "rpa_attention": 0.0,
      "dense_and_mhc": 0.0,
  }
  steps = 0
  for ev in events:
    if ev.get("ph") != "X":
      continue
    name = ev.get("name", "")
    dur = float(ev.get("dur", 0.0))
    if "jit_step_fun_impl" in name:
      steps += 1
    elif "gmm_v2" in name or "fused_w13_w2" in name:
      totals_us["gmm_moe"] += dur
    elif any(k in name for k in ("all-reduce", "all-gather", "reduce-scatter", "collective-permute")):
      totals_us["ici_collectives"] += dur
    elif "ragged_paged_attention" in name or "rpa_v3" in name:
      totals_us["rpa_attention"] += dur
  denom = max(steps, 1) * 1000.0
  return {k: round(v / denom, 3) for k, v in totals_us.items()}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("trace", type=Path, help="Path to XProf .trace.json.gz file")
  args = parser.parse_args()
  summary = analyze_trace(args.trace)
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
