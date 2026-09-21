#!/usr/bin/env python3
"""Builds benchmark_sweep_report.md and the three PNG charts from the sweep JSON files.

Reads one JSON per shape, as written by `benchmark_sweep.py`, and emits the report in
the format of the sglang-rtx-pro-6000 DeepSeek-V4-Flash README: a six-column
summary table, one full per-concurrency table per shape, and charts of
throughput, TTFT and TPOT against concurrency.

A missing point is printed as its stated reason, never as a blank and never as
an interpolation.

Usage: report.py <logs_dir> <out_dir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

SHAPES = [("1k1k", "1k/1k (balanced)"),
          ("8k1k", "8k/1k (prefill-heavy)"),
          ("1k8k", "1k/8k (reasoning)")]

# Categorical slots 1 to 3, assigned in fixed order and never cycled. This
# triple validates on the all-pairs list: worst CVD delta-E 9.2, worst
# normal-vision delta-E 24.0. Aqua sits below 3:1 against the light surface, so
# the charts carry direct labels and the report carries the full tables, which
# is the relief that warning requires.
SERIES_COLOR = {"1k1k": "#2a78d6", "8k1k": "#eb6834", "1k8k": "#1baf7a"}
# Marker shape repeats the identity, so the series are separable without colour.
SERIES_MARKER = {"1k1k": "o", "8k1k": "s", "1k8k": "^"}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# The published figures for the same model on a different accelerator, read
# from the reference README on 2026-09-02. They are quoted for context only.
# The two systems are not the same hardware, so this is not a cost, a power or
# a per-accelerator comparison. See the caveat printed beside the table.
REFERENCE = {
    "name": "SGLang on 2 × g4-standard-384 (16 × RTX PRO 6000 Blackwell, 96 GB)",
    "1k1k": {"peak": 4710.94, "conc": 512, "req_s": 4.62,
             "ttft_s": 1.23, "tpot_ms": 106.50},
    "8k1k": {"peak": 4209.22, "conc": 512, "req_s": 4.13,
             "ttft_s": 7.19, "tpot_ms": 113.38},
    "1k8k": {"peak": 1606.27, "conc": 512, "req_s": 0.68,
             "ttft_s": 1.55, "tpot_ms": 107.64},
}

# The server prints `GPU KV cache size` per DP rank, not per slice. With 16-way
# DP attention the slice total is 16 times that figure, and it differs between
# the two arms because the context length changes the block layout.
#   2048 context:  81,427 tokens per rank -> 1,302,832 across the slice
#   9216 context: 180,744 tokens per rank -> 2,891,904 across the slice
KV_TOKENS_BY_SHAPE = {"1k1k": 1302832, "8k1k": 2891904, "1k8k": 2891904}


def load(logs_dir, key):
    path = os.path.join(logs_dir, f"{key}.json")
    if not os.path.exists(path):
        return {}
    return {r["concurrency"]: r for r in json.load(open(path))}


def f(x, spec, dash="—"):
    return format(x, spec) if x is not None else dash


def summary_table(data):
    lines = ["| Pattern | Peak output tok/s | @ conc | Req/s | "
             "TTFT mean @ 512 | TPOT @ 512 |",
             "|---|---|---|---|---|---|"]
    peak_overall = max(
        (r["output_throughput_tok_s"] for rows in data.values()
         for r in rows.values()), default=0.0)
    for key, label in SHAPES:
        rows = data.get(key, {})
        if not rows:
            lines.append(f"| `{key[:2]}/{key[2:]}` ({label.split('(')[1]} | "
                         "not run | — | — | — | — |")
            continue
        best = max(rows.values(), key=lambda r: r["output_throughput_tok_s"])
        at512 = rows.get(512)
        peak = best["output_throughput_tok_s"]
        cell = f"**{peak:,.2f}**" if peak == peak_overall else f"{peak:,.2f}"
        lines.append(
            f"| `{label.split(' ')[0]}` ({label.split('(')[1][:-1]}) | {cell} | "
            f"{best['concurrency']} | {best['requests_per_s']:.2f} | "
            f"{f(at512 and at512['ttft_mean_ms'] and at512['ttft_mean_ms'] / 1000, '.2f')} s | "
            f"{f(at512 and at512['tpot_mean_ms'], '.2f')} ms |")
    return "\n".join(lines)


def reference_table(data):
    """Puts the measured peak beside the published figure for the same model.

    The ratio column divides one by the other. It compares two whole systems on
    one workload, and nothing else.
    """
    lines = ["| Pattern | This slice, peak tok/s | Reference, peak tok/s | "
             "Ratio | This slice, TPOT @ 512 | Reference, TPOT @ 512 |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, label in SHAPES:
        rows = data.get(key, {})
        ref = REFERENCE[key]
        if not rows:
            lines.append(f"| `{label.split(' ')[0]}` | *not run* | "
                         f"{ref['peak']:,.2f} | — | — | "
                         f"{ref['tpot_ms']:.2f} ms |")
            continue
        peak = max(r["output_throughput_tok_s"] for r in rows.values())
        at512 = rows.get(512)
        tpot = at512 and at512["tpot_mean_ms"]
        lines.append(
            f"| `{label.split(' ')[0]}` | **{peak:,.2f}** | {ref['peak']:,.2f} | "
            f"**{peak / ref['peak']:.2f}×** | {f(tpot, '.2f')} ms | "
            f"{ref['tpot_ms']:.2f} ms |")
    return "\n".join(lines)


def full_table(rows, osl):
    lines = ["| Concurrency | Requests | Output tok/s | tok/s per chip | "
             "Req/s | TTFT mean (ms) | TTFT p90 (ms) | TPOT mean (ms) | "
             "Success | Wall (s) |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONCURRENCIES:
        r = rows.get(c)
        if r is None:
            lines.append(f"| {c} | — | *not run* | — | — | — | — | — | — | — |")
            continue
        ok = f"{r['success_rate_pct']:.0f}%"
        if r["failed"]:
            ok += f" ({r['failed']} failed: {(r['first_error'] or '')[:60]})"
        lines.append(
            f"| {c} | {r['requests_issued']} | "
            f"**{r['output_throughput_tok_s']:,.1f}** | "
            f"{r['output_tok_s_per_chip']:,.1f} | {r['requests_per_s']:.3f} | "
            f"{f(r['ttft_mean_ms'], ',.0f')} | {f(r['ttft_p90_ms'], ',.0f')} | "
            f"{f(r['tpot_mean_ms'], '.2f')} | {ok} | {r['wall_time_s']:,.0f} |")
    return "\n".join(lines)


def chart(data, metric, ylabel, title, path, scale=1.0, logy=False,
          label_fmt=",.0f"):
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=160,
                           facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ends = []
    for key, label in SHAPES:
        rows = data.get(key, {})
        xs = [c for c in CONCURRENCIES
              if c in rows and rows[c].get(metric) is not None]
        if not xs:
            continue
        ys = [rows[c][metric] * scale for c in xs]
        ax.plot(xs, ys, linewidth=2, marker=SERIES_MARKER[key],
                markersize=6, color=SERIES_COLOR[key], label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        ends.append((xs[-1], ys[-1]))

    ax.set_xscale("log", base=2)
    ax.set_xticks(CONCURRENCIES)
    ax.set_xticklabels([str(c) for c in CONCURRENCIES])
    ax.minorticks_off()
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Concurrency (streams)", color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, pad=12)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(True, which="major", alpha=0.18, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d2")
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    leg.set_zorder(5)

    # One direct label per series, at its last point. A number on every point is
    # noise, and the labels are also the relief the contrast check asks for.
    # Two series can end close together, so the labels are separated here: the
    # end points are placed on the axis as a fraction, then pushed apart until
    # each is at least MIN_GAP from the one below it.
    lo, hi = ax.get_ylim()
    if logy:
        import math
        span = math.log10(hi) - math.log10(lo)

        def frac(y):
            return (math.log10(y) - math.log10(lo)) / span
    else:
        def frac(y):
            return (y - lo) / (hi - lo)

    MIN_GAP = 0.055               # about 16 points on a 5-inch axis
    AXIS_PTS = 290.0              # axis height in points, after tight_layout
    placed = []
    for x, y in sorted(ends, key=lambda p: p[1]):
        f = frac(y)
        if placed and f - placed[-1] < MIN_GAP:
            f = placed[-1] + MIN_GAP
        placed.append(f)
        ax.annotate(format(y, label_fmt), (x, y), textcoords="offset points",
                    xytext=(-8, 9 + (f - frac(y)) * AXIS_PTS), fontsize=9,
                    ha="right", color=INK_PRIMARY, zorder=6)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    logs_dir, out_dir = sys.argv[1], sys.argv[2]
    charts_dir = os.path.join(out_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    data = {key: load(logs_dir, key) for key, _ in SHAPES}

    chart(data, "output_throughput_tok_s", "Output tokens / s",
          "DeepSeek-V4-Flash on TPU v6e-16 — output throughput vs concurrency",
          os.path.join(charts_dir, "throughput_vs_concurrency.png"))
    # TTFT spans about 0.5 s to 130 s, more than two orders of magnitude. On a
    # linear axis every point below ten seconds collapses onto the baseline.
    chart(data, "ttft_mean_ms", "TTFT mean (s, log scale)",
          "DeepSeek-V4-Flash on TPU v6e-16 — TTFT vs concurrency",
          os.path.join(charts_dir, "ttft_vs_concurrency.png"), scale=1e-3,
          logy=True, label_fmt=",.1f")
    # TPOT spans about 25 to 53 ms, so a whole number hides the difference
    # between two series that end one millisecond apart.
    chart(data, "tpot_mean_ms", "TPOT mean (ms)",
          "DeepSeek-V4-Flash on TPU v6e-16 — TPOT vs concurrency",
          os.path.join(charts_dir, "tpot_vs_concurrency.png"),
          label_fmt=",.1f")

    parts = [
        "# DeepSeek-V4-Flash on TPU v6e-16 — benchmark results "
        "(concurrency sweep 1 → 512)",
        "",
        "Three workload patterns, run with a streaming client against a live "
        "vLLM endpoint on a 16-chip TPU v6e slice. Every concurrency level "
        "from 1 to 512 is measured, not interpolated.",
        "",
        "## Configuration",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Hardware | TPU v6e-16, one `4x4` slice, 4 × `ct6e-standard-4t` on "
        "GKE |",
        "| Model | `deepseek-ai/DeepSeek-V4-Flash` |",
        "| Serving | vLLM with the `tpu-inference` backend, TP=16, expert "
        "parallel |",
        "| Quantisation | INT8 weights, `--kv-cache-dtype=fp8` |",
        "| Flags | `--gpu-memory-utilization 0.85 "
        "--no-enable-prefix-caching --max-num-batched-tokens 2048` |",
        "| Client | streaming `/v1/completions`, `ignore_eos`, temperature 0, "
        "run inside the serving pod |",
        "",
        "**Two serving arms, not one.** `max_num_seqs × max_model_len` must "
        "stay at or below 4,194,304 on this stack, because the block table is "
        "prefetched into a 1 MiB SMEM. The `1k/1k` rows therefore come from "
        "`max_model_len 2048, max_num_seqs 512`, and the two 8k shapes come "
        "from `max_model_len 9216, max_num_seqs 256`. **The three shapes are "
        "not all measured on one configuration**, and a reader must not treat "
        "the three rows as one system state.",
        "",
        "**The reasoning shape is limited by the KV pool, not by the sequence "
        "cap.** A `1k/8k` request occupies only its 1024-token prompt at "
        "admission, so the scheduler admits far more than 256 of them, and "
        "each one then grows. The pool holds about 313 sequences at the full "
        "9216-token length. A separate run raised the cap to 384 and measured "
        "7992.80 tok/s at C=512, which is 0.93% below the figure in the "
        "table below and "
        "inside the reproducibility band, so the cap is not the constraint. "
        "The throughput and the mean time per output token imply 323.7 "
        "concurrent streams at C=512, which agrees with the 313 the pool "
        "predicts.",
        "",
        "**8k prompts arrive as four prefill chunks.** "
        "`--max-num-batched-tokens` stays at 2048, because a higher value "
        "fails to compile on this slice. An 8192-token prompt therefore takes "
        "four chunked prefill steps. This is a property of the measurement, "
        "and it is the main cause of the `8k/1k` result below.",
        "",
        "## Summary",
        "",
        summary_table(data),
        "",
        "![Output throughput vs concurrency]"
        "(charts/throughput_vs_concurrency.png)",
        "",
        "## Beside the published reference",
        "",
        f"The reference is **{REFERENCE['name']}**, read from its README on "
        "2026-09-02. It runs the same model on different accelerators, so "
        "**this is a comparison of two whole systems on one workload**. It is "
        "not a per-accelerator, per-watt or per-dollar comparison, and it "
        "does not hold the software stack constant.",
        "",
        reference_table(data),
        "",
    ]

    for key, label in SHAPES:
        rows = data.get(key, {})
        parts += [f"## {label}", ""]
        if not rows:
            parts += ["*Not run.*", ""]
            continue
        any_row = next(iter(rows.values()))
        isl, osl = any_row["isl_max"], any_row["osl"]
        # A sequence occupies its prompt at admission and grows by one token per
        # step. The pool therefore holds many more of a long-output request at
        # the start than at the end, which is why the reasoning shape keeps
        # scaling where the prefill-heavy shape does not.
        kv_total = KV_TOKENS_BY_SHAPE[key]
        start, end = kv_total // isl, kv_total // (isl + osl)
        occupancy = (
            f"so it holds **{end} of them at once**"
            if start // 2 <= end else
            f"so it holds **{start} of them at admission and only {end} once "
            f"they are complete**")
        parts += [
            f"Measured ISL {any_row['isl_min']} to {isl} tokens, OSL {osl}, "
            f"`ignore_eos`, temperature 0. A request reaches {isl + osl} "
            f"tokens at completion. The KV pool holds "
            f"{kv_total:,} tokens across the slice, {occupancy}; "
            f"concurrency above that queues at the server.",
            "",
            full_table(rows, any_row["osl"]),
            "",
        ]

    parts += [
        "## Charts",
        "",
        "![TTFT vs concurrency](charts/ttft_vs_concurrency.png)",
        "",
        "![TPOT vs concurrency](charts/tpot_vs_concurrency.png)",
        "",
    ]

    out = os.path.join(out_dir, "benchmark_sweep_report.md")
    open(out, "w").write("\n".join(parts))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
