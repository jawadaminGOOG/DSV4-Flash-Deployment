#!/usr/bin/env python3
"""Generate the 4-panel hardware & throughput analysis chart for Fused W13+SiLU+W2 + Hybrid EP=8 x TP=2."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
  out_path = (
      Path(__file__).resolve().parent.parent
      / "results"
      / "charts"
      / "fused-moe-ep8-tp2-analysis.png"
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)

  plt.rcParams.update({
      "font.family": "DejaVu Sans",
      "font.size": 10,
      "axes.titlesize": 11.5,
      "axes.titleweight": "bold",
      "axes.labelsize": 10,
      "xtick.labelsize": 9,
      "ytick.labelsize": 9,
      "legend.fontsize": 8.5,
      "figure.titlesize": 13.5,
      "figure.titleweight": "bold",
  })

  fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), dpi=180)
  fig.suptitle(
      "DeepSeek-V4.1-Flash on TPU v6e-16: Single-Kernel Fused W13+SiLU+W2 MoE + Hybrid EP=8×TP=2 Sharding",
      y=0.98,
  )

  c_base = "#64748B"
  c_vmem = "#3B82F6"
  c_fused = "#10B981"
  c_warn = "#EF4444"

  # Panel A: 1k/1k Output Throughput across Concurrency
  ax = axes[0, 0]
  concurrencies = ["C=64\n(Warm)", "C=128\n(Warm)", "C=185\n(1-Wave)", "C=190\n(KV Max)", "C=256\n(2-Wave)"]
  x = np.arange(len(concurrencies))
  w = 0.26

  y_base = [1524.2, 2521.0, 3896.2, 3992.6, 2447.8]
  y_vmem = [1525.1, 2539.7, 3925.4, 4003.7, 2445.0]
  y_fused = [1740.4, 2984.5, 4643.6, 4758.5, 2882.6]

  b1 = ax.bar(x - w, y_base, width=w, label="XProf Baseline (EP=16, 2-Kernel MoE)", color=c_base)
  b2 = ax.bar(x, y_vmem, width=w, label="Local VMEM Prefetch Only (EP=16)", color=c_vmem)
  b3 = ax.bar(x + w, y_fused, width=w, label="Fused W13+SiLU+W2 + Hybrid EP=8×TP=2", color=c_fused)

  for bar, base_val in zip(b3, y_base):
    val = bar.get_height()
    gain = (val - base_val) / base_val * 100.0
    ax.annotate(
        f"{val:,.0f}\n(+{gain:.1f}%)",
        xy=(bar.get_x() + bar.get_width() / 2, val),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#065F46",
    )

  ax.set_title("A. End-to-End Output Throughput (1k in / 1k out, 16K Context)")
  ax.set_ylabel("Output Throughput (tokens/s)")
  ax.set_xticks(x)
  ax.set_xticklabels(concurrencies)
  ax.set_ylim(0, 5600)
  ax.grid(axis="y", linestyle="--", alpha=0.35)
  ax.legend(loc="upper left")

  # Panel B: Multi-Rank XProf Decode Step Breakdown (C=64)
  ax = axes[0, 1]
  configs = [
      "XProf Baseline\n(EP=16, 2-Pass GMM)\n[39.86 ms XLA / 39.18 ms p50]",
      "Local VMEM Prefetch\n(EP=16, 23.35 MiB VMEM)\n[39.59 ms XLA / 38.79 ms p50]",
      "Fused W13+SiLU+W2\n+ Hybrid EP=8×TP=2\n[35.75 ms XLA / 32.72 ms p50]",
  ]
  y_pos = np.arange(len(configs))

  gmm_ms = np.array([10.748, 10.358, 10.118])
  ici_ms = np.array([11.066, 11.066, 6.967])
  attn_ms = np.array([5.512, 5.512, 5.512])
  dense_ms = np.array([12.535, 12.657, 13.151])

  ax.barh(y_pos, gmm_ms, label="Routed MoE Kernel (gmm_v2 / fused_w13_w2)", color="#2563EB", height=0.48)
  ax.barh(y_pos, ici_ms, left=gmm_ms, label="ICI Collectives & Straggler Wait (all-reduce + ppermute)", color="#F59E0B", height=0.48)
  ax.barh(y_pos, attn_ms, left=gmm_ms + ici_ms, label="CSA2 + SWA Paged Attention (rpa_v3)", color="#8B5CF6", height=0.48)
  ax.barh(y_pos, dense_ms, left=gmm_ms + ici_ms + attn_ms, label="Dense Projections + Shared Expert + 4-Stream mHC", color="#64748B", height=0.48)

  for idx, (g, ic, tot) in enumerate(zip(gmm_ms, ici_ms, gmm_ms + ici_ms + attn_ms + dense_ms)):
    ax.text(g / 2, idx, f"{g:.2f} ms", va="center", ha="center", color="white", fontweight="bold", fontsize=8.5)
    ax.text(g + ic / 2, idx, f"{ic:.2f} ms", va="center", ha="center", color="white", fontweight="bold", fontsize=8.5)
    ax.text(tot + 0.4, idx, f"{tot:.2f} ms", va="center", ha="left", fontweight="bold", fontsize=9)

  ax.set_title("B. Multi-Rank XProf Decode Step Time Breakdown (C=64, 40 MoE Layers)")
  ax.set_xlabel("Hardware Execution Time per Decode Step (ms)")
  ax.set_yticks(y_pos)
  ax.set_yticklabels(configs)
  ax.set_xlim(0, 46)
  ax.invert_yaxis()
  ax.grid(axis="x", linestyle="--", alpha=0.35)
  ax.legend(loc="lower right", fontsize=8)

  # Panel C: TPU v6e Scoped VMEM Capacity & Double-Buffered Footprint
  ax = axes[1, 0]
  vmem_configs = [
      "XProf Baseline\n(EP=16, 2-Pass,\nmegablocks=1)",
      "Pure EP=16 Fused\nW13+W2 Attempt\n(24 full experts)",
      "Hybrid EP=8×TP=2\n+ Fused W13+SiLU+W2\n(48 half-width experts)",
  ]
  vmem_vals = [8.55, 39.64, 23.35]
  vmem_colors = [c_base, c_warn, c_fused]

  bars = ax.bar(vmem_configs, vmem_vals, color=vmem_colors, width=0.48)
  ax.axhline(30.72, color=c_warn, linestyle="--", linewidth=2, label="TPU v6e Scoped VMEM Hard Ceiling (30.72 MiB)")
  ax.axhline(32.00, color="#991B1B", linestyle=":", linewidth=1.5, label="Physical VMEM Capacity (32.00 MiB)")

  for bar, val in zip(bars, vmem_vals):
    status = "OOM (>30.72 MiB)" if val > 30.72 else f"{val / 30.72 * 100:.1f}% of Ceiling"
    ax.annotate(
        f"{val:.2f} MiB\n({status})",
        xy=(bar.get_x() + bar.get_width() / 2, val),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )

  ax.set_title("C. Why Hybrid EP=8×TP=2 Unlocks Single-Kernel W13+W2 VMEM Fusion")
  ax.set_ylabel("Peak Scoped VMEM Allocation (MiB)")
  ax.set_ylim(0, 46)
  ax.grid(axis="y", linestyle="--", alpha=0.35)
  ax.legend(loc="upper right")

  # Panel D: Mean Decode TPOT Latency & Per-Chip Efficiency
  ax = axes[1, 1]
  tpot_c = ["C=64\n(Warm)", "C=128\n(Warm)", "C=190\n(1-Wave Max)", "C=256\n(2-Wave)"]
  tx = np.arange(len(tpot_c))
  tw = 0.32
  tpot_base = [39.2, 45.5, 45.0, 50.2]
  tpot_fused = [34.5, 38.5, 37.8, 42.9]

  tb1 = ax.bar(tx - tw / 2, tpot_base, width=tw, label="XProf Baseline TPOT (ms/tok)", color=c_base)
  tb2 = ax.bar(tx + tw / 2, tpot_fused, width=tw, label="Fused W13+SiLU+W2 + EP=8×TP=2 TPOT (ms/tok)", color=c_fused)

  for bar, b_val in zip(tb2, tpot_base):
    val = bar.get_height()
    delta = val - b_val
    ax.annotate(
        f"{val:.1f} ms\n({delta:.1f} ms)",
        xy=(bar.get_x() + bar.get_width() / 2, val),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#065F46",
    )

  ax.set_title("D. Mean Time-Per-Output-Token (TPOT) Reduction Across Concurrency")
  ax.set_ylabel("Mean TPOT Latency (ms / token, lower is better)")
  ax.set_xticks(tx)
  ax.set_xticklabels(tpot_c)
  ax.set_ylim(0, 60)
  ax.grid(axis="y", linestyle="--", alpha=0.35)
  ax.legend(loc="upper left")

  plt.tight_layout(rect=[0, 0, 1, 0.95])
  plt.savefig(out_path, dpi=180)
  print(f"Saved {out_path}")


if __name__ == "__main__":
  main()
