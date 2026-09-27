#!/usr/bin/env python3
"""Generate unified 2-panel Workload Throughput & Decode Latency chart AND 2-panel Megakernel + DSpark chart
using live `enable_engram=True` hardware measurements from `results/megakernel_v6e16_results.json`.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    return ImageFont.truetype(path, size=size)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p1: tuple[float, float],
    p2: tuple[float, float],
    fill: tuple[int, int, int, int],
    width: int = 3,
    dash_len: int = 14,
    gap_len: int = 10,
) -> None:
    import math

    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    ux, uy = dx / dist, dy / dist
    pos = 0.0
    while pos < dist:
        end = min(pos + dash_len, dist)
        draw.line(
            [(x1 + ux * pos, y1 + uy * pos), (x1 + ux * end, y1 + uy * end)],
            fill=fill,
            width=width,
        )
        pos += dash_len + gap_len


def draw_circle(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    r: float,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] = (255, 255, 255, 255),
    width: int = 3,
) -> None:
    draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=width)


def draw_diamond(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    r: float,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> None:
    pts = [(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
    draw.polygon(pts, fill=fill, outline=outline)


def draw_centered_badge(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill_bg: tuple[int, int, int, int],
    outline_col: tuple[int, int, int, int],
    text_col: tuple[int, int, int, int],
) -> None:
    draw.rounded_rectangle(box, radius=10, fill=fill_bg, outline=outline_col, width=2)
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    cx = (box[0] + box[2] - tw) / 2
    cy = (box[1] + box[3] - th) / 2 - 2
    draw.text((cx, cy), text, fill=text_col, font=font)


def generate_three_recipe_chart(repo_root: Path) -> None:
    W, H = 3800, 2040
    img = Image.new("RGBA", (W, H), (250, 252, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    f_title = load_font(46, bold=True)
    f_sub = load_font(27, bold=False)
    f_panel = load_font(33, bold=True)
    f_axis = load_font(26, bold=True)
    f_tick = load_font(23, bold=True)
    f_leg = load_font(21, bold=False)
    f_leg_b = load_font(21, bold=True)
    f_inset_hdr = load_font(20, bold=True)
    f_val = load_font(21, bold=True)
    f_val_sm = load_font(18, bold=True)
    f_badge = load_font(18, bold=True)
    f_foot = load_font(21, bold=False)

    draw.rectangle([0, 0, W, 165], fill=(15, 23, 42, 255))
    title = "DeepSeek-V4.1-Flash (552B / 16B Active) on TPU v6e-16: Megakernel vs. Fused Kernel vs. Base XLA"
    tb = draw.textbbox((0, 0), title, font=f_title)
    draw.text(((W - (tb[2] - tb[0])) / 2, 26), title, fill=(248, 250, 252, 255), font=f_title)

    sub = (
        "Workload Output Throughput (tok/s) & Per-Token Decode Latency (TPOT ms) Across Concurrency C = 1..256  |  "
        "40 MoE Layers (Top-6/384 MXFP4) + 200.5 GB Engram + 3 DSpark MTP Layers"
    )
    sb = draw.textbbox((0, 0), sub, font=f_sub)
    draw.text(((W - (sb[2] - sb[0])) / 2, 98), sub, fill=(148, 163, 184, 255), font=f_sub)

    c_labels = ["C=1", "C=2", "C=4", "C=8", "C=16", "C=64", "C=128", "C=185", "C=190\n(KV Max)", "C=256"]
    n_pts = len(c_labels)

    # 1. Unoptimized Batched XLA (d85ce9c1, 16K ctx)
    unopt_toks = [18.3, 36.0, 70.6, 144.9, 361.2, 858.3, 2017.6, 3100.0, 3164.1, 2150.0]
    unopt_tpot = [54.6, 55.6, 56.6, 55.2, 44.3, 68.6, 57.1, 56.4, 55.2, 69.4]

    # 2. Base Batched XLA (Two-Pass EP=16 XProf Optimized, kernel-optimizations-recipe 4b8edd8e)
    base_toks = [22.3, 43.7, 107.5, 198.6, 370.0, 1524.2, 2521.0, 3896.2, 3992.6, 2447.8]
    base_tpot = [44.78, 45.73, 37.19, 39.59, 42.62, 39.20, 45.50, 44.80, 45.10, 50.20]

    # 3. Fused W13+SiLU+W2 MoE + Hybrid EP=8xTP=2 (fused-kernels-recipe 623b2904, C=64..256)
    fused_indices = [5, 6, 7, 8, 9]
    fused_toks = [1740.4, 2984.5, 4643.6, 4758.5, 2882.6]
    fused_tpot = [34.50, 38.50, 37.60, 37.80, 42.90]

    # 4. 40-Layer Pallas Decode Megakernel with live 200.5 GB Engram (`enable_engram=True`, C=1..16)
    mk_indices = [0, 1, 2, 3, 4]
    mk_toks = [107.6, 180.0, 275.2, 386.6, 392.8]
    mk_tpot = [9.30, 11.11, 14.53, 20.69, 40.74]
    mk_speedups = ["4.82x", "4.11x", "2.56x", "1.91x", "1.05x"]

    # 5. Pallas Megakernel + DSpark (`enable_engram=True`, mtp.0..2 lossless 1+4 speculative at C=1)
    dspark_med_toks = 137.5
    dspark_best_toks = 229.5
    dspark_med_tpot = 7.27
    dspark_best_tpot = 4.36

    col_unopt = (148, 163, 184, 255)
    col_base = (71, 85, 105, 255)
    col_fused = (16, 185, 129, 255)
    col_fused_dark = (4, 120, 87, 255)
    col_mk = (14, 165, 233, 255)
    col_mk_dark = (3, 105, 161, 255)
    col_dspark = (139, 92, 246, 255)
    col_dspark_dark = (109, 40, 217, 255)

    # PANEL A (LEFT): WORKLOAD OUTPUT THROUGHPUT (tok/s)
    p1_l, p1_t, p1_r, p1_b = 190, 290, 1810, 1690
    p1_w, p1_h = p1_r - p1_l, p1_b - p1_t

    t1 = "A. Workload Output Throughput (tok/s) — Higher is Better"
    tb1 = draw.textbbox((0, 0), t1, font=f_panel)
    draw.text((p1_l + (p1_w - (tb1[2] - tb1[0])) / 2, 215), t1, fill=(15, 23, 42, 255), font=f_panel)

    def x_at_1(idx: float) -> float:
        return p1_l + 65 + idx * ((p1_w - 130) / (n_pts - 1))

    x_split_1 = (x_at_1(4) + x_at_1(5)) / 2
    draw.rectangle([p1_l, p1_t, x_split_1, p1_b], fill=(224, 242, 254, 110))
    draw.rectangle([x_split_1, p1_t, p1_r, p1_b], fill=(209, 250, 229, 95))
    draw.rectangle([p1_l, p1_t, p1_r, p1_b], outline=(148, 163, 184, 255), width=3)

    draw_centered_badge(
        draw,
        [p1_l + 16, p1_t + 14, x_split_1 - 12, p1_t + 62],
        "Interactive Regime (C = 1..16): Megakernel + DSpark (up to 10.28x)",
        f_badge,
        (14, 165, 233, 38),
        (14, 165, 233, 180),
        col_mk_dark,
    )
    draw_centered_badge(
        draw,
        [x_split_1 + 12, p1_t + 14, p1_r - 16, p1_t + 62],
        "Batched Regime (C = 64..256): Fused W13+W2 + EP=8×TP=2 (+19.2%)",
        f_badge,
        (16, 185, 129, 38),
        (16, 185, 129, 180),
        col_fused_dark,
    )

    draw_dashed_line(draw, (x_split_1, p1_t), (x_split_1, p1_b), fill=(100, 116, 139, 180), width=3)

    y1_max = 5200.0

    def y_at_1(val: float) -> float:
        return p1_b - (val / y1_max) * (p1_h - 90)

    for y_val in range(0, 5001, 1000):
        yy = y_at_1(y_val)
        draw.line([(p1_l, yy), (p1_r, yy)], fill=(203, 213, 225, 180), width=2)
        lbl = f"{y_val:,}"
        bb = draw.textbbox((0, 0), lbl, font=f_tick)
        draw.text((p1_l - (bb[2] - bb[0]) - 14, yy - (bb[3] - bb[1]) / 2), lbl, fill=(51, 65, 85, 255), font=f_tick)

    for i, lbl in enumerate(c_labels):
        xx = x_at_1(i)
        draw.line([(xx, p1_b), (xx, p1_b + 10)], fill=(51, 65, 85, 255), width=3)
        for line_idx, part in enumerate(lbl.split("\n")):
            bb = draw.textbbox((0, 0), part, font=f_tick)
            draw.text((xx - (bb[2] - bb[0]) / 2, p1_b + 16 + line_idx * 26), part, fill=(15, 23, 42, 255), font=f_tick)

    ax_lbl_1 = "Active Serving Concurrency (1k-in / 1k-out Batched & 1k-in / 256-out Low-Concurrency Ladder)"
    ab1 = draw.textbbox((0, 0), ax_lbl_1, font=f_axis)
    draw.text((p1_l + (p1_w - (ab1[2] - ab1[0])) / 2, p1_b + 76), ax_lbl_1, fill=(15, 23, 42, 255), font=f_axis)

    pts_unopt_1 = [(x_at_1(i), y_at_1(v)) for i, v in enumerate(unopt_toks)]
    for i in range(len(pts_unopt_1) - 1):
        draw_dashed_line(draw, pts_unopt_1[i], pts_unopt_1[i + 1], fill=col_unopt, width=5)
    for x, y in pts_unopt_1:
        draw_circle(draw, x, y, 8, fill=col_unopt)

    pts_base_1 = [(x_at_1(i), y_at_1(v)) for i, v in enumerate(base_toks)]
    draw.line(pts_base_1, fill=col_base, width=6)
    for x, y in pts_base_1:
        draw_circle(draw, x, y, 10, fill=col_base)

    pts_fused_1 = [(x_at_1(i), y_at_1(v)) for i, v in zip(fused_indices, fused_toks)]
    draw.line(pts_fused_1, fill=col_fused, width=8)
    for (x, y), val, bval in zip(pts_fused_1, fused_toks, [base_toks[i] for i in fused_indices]):
        draw_circle(draw, x, y, 12, fill=col_fused)
        gain = (val - bval) / bval * 100.0
        txt = f"{val:,.0f}\n(+{gain:.1f}%)"
        for li, line in enumerate(txt.split("\n")):
            bb = draw.textbbox((0, 0), line, font=f_val)
            draw.text((x - (bb[2] - bb[0]) / 2, y - 62 + li * 24), line, fill=col_fused_dark, font=f_val)

    for idx in [5, 6, 7, 8, 9]:
        x, y = pts_base_1[idx]
        lbl = f"{base_toks[idx]:,.0f}"
        bb = draw.textbbox((0, 0), lbl, font=f_val)
        draw.text((x - (bb[2] - bb[0]) / 2, y + 16), lbl, fill=col_base, font=f_val)

    pts_mk_1 = [(x_at_1(i), y_at_1(v)) for i, v in zip(mk_indices, mk_toks)]
    draw.line(pts_mk_1, fill=col_mk, width=7)
    for x, y in pts_mk_1:
        draw_circle(draw, x, y, 11, fill=col_mk)

    x_c1 = x_at_1(0)
    draw_diamond(draw, x_c1, y_at_1(dspark_med_toks), 14, fill=col_dspark)
    draw_diamond(draw, x_c1, y_at_1(dspark_best_toks), 14, fill=col_dspark)

    # Magnified Inset inside Panel A for C = 1 .. 16
    in_l, in_t, in_r, in_b = p1_l + 24, p1_t + 88, x_split_1 - 18, p1_t + 890
    draw.rounded_rectangle([in_l, in_t, in_r, in_b], radius=14, fill=(255, 255, 255, 250), outline=(14, 165, 233, 220), width=3)
    in_hdr = "Zoom Inset: Low-Concurrency Throughput (C = 1..16, tok/s)"
    hb = draw.textbbox((0, 0), in_hdr, font=f_inset_hdr)
    draw.text((in_l + ((in_r - in_l) - (hb[2] - hb[0])) / 2, in_t + 14), in_hdr, fill=(15, 23, 42, 255), font=f_inset_hdr)

    in_plot_l, in_plot_t, in_plot_r, in_plot_b = in_l + 78, in_t + 78, in_r - 32, in_b - 54
    in_w, in_h = in_plot_r - in_plot_l, in_plot_b - in_plot_t
    in_ymax = 480.0

    def in_x(i: int) -> float:
        return in_plot_l + 32 + i * ((in_w - 64) / 4.0)

    def in_y(v: float) -> float:
        return in_plot_b - (v / in_ymax) * in_h

    for gv in [0, 100, 200, 300, 400]:
        yy = in_y(gv)
        draw.line([(in_plot_l, yy), (in_plot_r, yy)], fill=(226, 232, 240, 255), width=2)
        bb = draw.textbbox((0, 0), str(gv), font=f_val_sm)
        draw.text((in_plot_l - (bb[2] - bb[0]) - 10, yy - (bb[3] - bb[1]) / 2), str(gv), fill=(71, 85, 105, 255), font=f_val_sm)

    for i in range(5):
        xx = in_x(i)
        bb = draw.textbbox((0, 0), c_labels[i], font=f_val_sm)
        draw.text((xx - (bb[2] - bb[0]) / 2, in_plot_b + 12), c_labels[i], fill=(15, 23, 42, 255), font=f_val_sm)

    in_pts_base = [(in_x(i), in_y(base_toks[i])) for i in range(5)]
    in_pts_mk = [(in_x(i), in_y(mk_toks[i])) for i in range(5)]
    draw.line(in_pts_base, fill=col_base, width=5)
    draw.line(in_pts_mk, fill=col_mk, width=7)

    for i, (x, y) in enumerate(in_pts_base):
        draw_circle(draw, x, y, 9, fill=col_base)
        lbl = f"{base_toks[i]:.1f}"
        bb = draw.textbbox((0, 0), lbl, font=f_val_sm)
        draw.text((x - (bb[2] - bb[0]) / 2, y + 14), lbl, fill=col_base, font=f_val_sm)

    for i, (x, y) in enumerate(in_pts_mk):
        draw_circle(draw, x, y, 10, fill=col_mk)
        sp_str = mk_speedups[i]
        if i == 0:
            lbl = f"{mk_toks[i]:.1f} ({sp_str})"
            draw.text((x + 14, y - 8), lbl, fill=col_mk_dark, font=f_val_sm)
        else:
            lines = [f"{mk_toks[i]:.1f}", f"({sp_str})"]
            for li, line in enumerate(lines):
                bb = draw.textbbox((0, 0), line, font=f_val_sm)
                x_shift = -8 if i == 3 else (8 if i == 4 else 0)
                draw.text((x - (bb[2] - bb[0]) / 2 + x_shift, y - 48 + li * 20), line, fill=col_mk_dark, font=f_val_sm)

    ix0 = in_x(0)
    iy_med = in_y(dspark_med_toks)
    iy_best = in_y(dspark_best_toks)
    draw.line([(ix0, in_y(mk_toks[0])), (ix0, iy_best)], fill=col_dspark, width=4)
    draw_diamond(draw, ix0, iy_med, 13, fill=col_dspark)
    draw_diamond(draw, ix0, iy_best, 13, fill=col_dspark)

    callout_box = [in_plot_l + 6, in_plot_t + 10, in_plot_l + 315, in_plot_t + 92]
    draw.rounded_rectangle(callout_box, radius=8, fill=(245, 243, 255, 245), outline=(139, 92, 246, 180), width=2)
    draw.line([(ix0, iy_best - 14), (ix0 + 18, callout_box[3])], fill=(139, 92, 246, 190), width=3)
    draw.text((callout_box[0] + 12, callout_box[1] + 10), "DSpark C=1: 229.5 best (10.28x)", fill=col_dspark_dark, font=f_val_sm)
    draw.text((callout_box[0] + 12, callout_box[1] + 44), "DSpark C=1: 137.5 med (6.16x)", fill=col_dspark_dark, font=f_val_sm)

    # PANEL B (RIGHT): PER-TOKEN DECODE LATENCY (TPOT ms/token)
    p2_l, p2_t, p2_r, p2_b = 2030, 290, 3650, 1690
    p2_w, p2_h = p2_r - p2_l, p2_b - p2_t

    t2 = "B. Per-Token Decode Latency (TPOT ms/tok) — Lower is Better"
    tb2 = draw.textbbox((0, 0), t2, font=f_panel)
    draw.text((p2_l + (p2_w - (tb2[2] - tb2[0])) / 2, 215), t2, fill=(15, 23, 42, 255), font=f_panel)

    def x_at_2(idx: float) -> float:
        return p2_l + 65 + idx * ((p2_w - 130) / (n_pts - 1))

    x_split_2 = (x_at_2(4) + x_at_2(5)) / 2
    draw.rectangle([p2_l, p2_t, x_split_2, p2_b], fill=(224, 242, 254, 110))
    draw.rectangle([x_split_2, p2_t, p2_r, p2_b], fill=(209, 250, 229, 95))
    draw.rectangle([p2_l, p2_t, p2_r, p2_b], outline=(148, 163, 184, 255), width=3)

    draw_centered_badge(
        draw,
        [p2_l + 16, p2_t + 14, x_split_2 - 12, p2_t + 62],
        "Sub-21 ms Interactive Regime: 4.36–9.30 ms @ C=1",
        f_badge,
        (14, 165, 233, 38),
        (14, 165, 233, 180),
        col_mk_dark,
    )
    draw_centered_badge(
        draw,
        [x_split_2 + 12, p2_t + 14, p2_r - 16, p2_t + 62],
        "Batched Regime: -4.7 to -7.3 ms/tok TPOT Reduction",
        f_badge,
        (16, 185, 129, 38),
        (16, 185, 129, 180),
        col_fused_dark,
    )

    draw_dashed_line(draw, (x_split_2, p2_t), (x_split_2, p2_b), fill=(100, 116, 139, 180), width=3)

    y2_max = 78.0

    def y_at_2(val: float) -> float:
        return p2_b - (val / y2_max) * (p2_h - 90)

    for y_val in range(0, 71, 10):
        yy = y_at_2(y_val)
        draw.line([(p2_l, yy), (p2_r, yy)], fill=(203, 213, 225, 180), width=2)
        lbl = f"{y_val} ms"
        bb = draw.textbbox((0, 0), lbl, font=f_tick)
        draw.text((p2_l - (bb[2] - bb[0]) - 14, yy - (bb[3] - bb[1]) / 2), lbl, fill=(51, 65, 85, 255), font=f_tick)

    for i, lbl in enumerate(c_labels):
        xx = x_at_2(i)
        draw.line([(xx, p2_b), (xx, p2_b + 10)], fill=(51, 65, 85, 255), width=3)
        for line_idx, part in enumerate(lbl.split("\n")):
            bb = draw.textbbox((0, 0), part, font=f_tick)
            draw.text((xx - (bb[2] - bb[0]) / 2, p2_b + 16 + line_idx * 26), part, fill=(15, 23, 42, 255), font=f_tick)

    ax_lbl_2 = "Active Serving Concurrency (Crossover Between Megakernel & Batched XLA Above C = 16)"
    ab2 = draw.textbbox((0, 0), ax_lbl_2, font=f_axis)
    draw.text((p2_l + (p2_w - (ab2[2] - ab2[0])) / 2, p2_b + 76), ax_lbl_2, fill=(15, 23, 42, 255), font=f_axis)

    pts_unopt_2 = [(x_at_2(i), y_at_2(v)) for i, v in enumerate(unopt_tpot)]
    for i in range(len(pts_unopt_2) - 1):
        draw_dashed_line(draw, pts_unopt_2[i], pts_unopt_2[i + 1], fill=col_unopt, width=5)
    for x, y in pts_unopt_2:
        draw_circle(draw, x, y, 8, fill=col_unopt)

    pts_base_2 = [(x_at_2(i), y_at_2(v)) for i, v in enumerate(base_tpot)]
    draw.line(pts_base_2, fill=col_base, width=6)
    for i, (x, y) in enumerate(pts_base_2):
        draw_circle(draw, x, y, 10, fill=col_base)
        lbl = f"{base_tpot[i]:.1f}" if i >= 5 else f"{base_tpot[i]:.2f}"
        bb = draw.textbbox((0, 0), lbl, font=f_val)
        draw.text((x - (bb[2] - bb[0]) / 2, y - 34), lbl, fill=col_base, font=f_val)

    pts_fused_2 = [(x_at_2(i), y_at_2(v)) for i, v in zip(fused_indices, fused_tpot)]
    draw.line(pts_fused_2, fill=col_fused, width=8)
    for (x, y), val, bval in zip(pts_fused_2, fused_tpot, [base_tpot[i] for i in fused_indices]):
        draw_circle(draw, x, y, 12, fill=col_fused)
        diff = val - bval
        txt = f"{val:.1f} ms\n({diff:.1f})"
        for li, line in enumerate(txt.split("\n")):
            bb = draw.textbbox((0, 0), line, font=f_val)
            draw.text((x - (bb[2] - bb[0]) / 2, y + 18 + li * 24), line, fill=col_fused_dark, font=f_val)

    pts_mk_2 = [(x_at_2(i), y_at_2(v)) for i, v in zip(mk_indices, mk_tpot)]
    draw.line(pts_mk_2, fill=col_mk, width=8)
    for i, (x, y) in enumerate(pts_mk_2):
        draw_circle(draw, x, y, 12, fill=col_mk)
        sp_str = mk_speedups[i]
        txt = f"{mk_tpot[i]:.2f} ms\n({sp_str})"
        for li, line in enumerate(txt.split("\n")):
            bb = draw.textbbox((0, 0), line, font=f_val)
            y_base = y - 58 if i < 4 else y + 18
            draw.text((x - (bb[2] - bb[0]) / 2, y_base + li * 24), line, fill=col_mk_dark, font=f_val)

    x2_c1 = x_at_2(0)
    y2_med = y_at_2(dspark_med_tpot)
    y2_best = y_at_2(dspark_best_tpot)
    draw.line([(x2_c1, y_at_2(mk_tpot[0])), (x2_c1, y2_best)], fill=col_dspark, width=5)
    draw_diamond(draw, x2_c1, y2_med, 14, fill=col_dspark)
    draw_diamond(draw, x2_c1, y2_best, 14, fill=col_dspark)
    draw.text((x2_c1 + 22, y2_med - 18), "7.27 ms DSpark med (6.16x vs Base XLA)", fill=col_dspark_dark, font=f_val)
    draw.text((x2_c1 + 22, y2_best + 4), "4.36 ms DSpark best (10.28x vs Base XLA)", fill=col_dspark_dark, font=f_val)

    # SHARED LEGEND & FOOTER
    leg_y = 1815
    draw.rounded_rectangle([190, leg_y, 3650, leg_y + 125], radius=14, fill=(241, 245, 249, 255), outline=(203, 213, 225, 255), width=2)

    items = [
        (col_dspark, "diamond", "40L Pallas Megakernel + Engram + DSpark (mtp.0..2 Speculative): 7.27 ms med (137.5 tok/s) / 4.36 ms best (229.5 tok/s) @ C=1"),
        (col_mk, "solid", "40L Pallas Decode Megakernel + 200.5 GB Engram (1 tpu_custom_call): 9.30 ms (107.6 tok/s) @ C=1 .. 40.74 ms (392.8 tok/s) @ C=16"),
        (col_fused, "solid", "Fused W13+SiLU+W2 MoE + Hybrid EP=8×TP=2 (623b2904): 32.72 ms/step, 1,740 tok/s @ C=64 .. 4,758.5 tok/s @ C=190 (+19.2%)"),
        (col_base, "solid", "Base Batched XLA (Two-Pass EP=16 XProf Optimized, 4b8edd8e): 39.18 ms/step, 22.3 tok/s @ C=1 .. 3,992.6 tok/s @ C=190"),
        (col_unopt, "dashed", "Unoptimized Batched XLA Baseline (d85ce9c1): 61.71 ms/step, 18.3 tok/s @ C=1 .. 3,164.1 tok/s @ C=190"),
    ]

    positions = [
        (225, leg_y + 22),
        (2020, leg_y + 22),
        (225, leg_y + 72),
        (2020, leg_y + 72),
    ]
    for idx, (col, style, text) in enumerate(items[:4]):
        lx, ly = positions[idx]
        if style == "diamond":
            draw_diamond(draw, lx + 25, ly + 13, 13, fill=col)
        else:
            draw.line([(lx, ly + 13), (lx + 50, ly + 13)], fill=col, width=7)
            draw_circle(draw, lx + 25, ly + 13, 10, fill=col)
        draw.text((lx + 64, ly), text, fill=(15, 23, 42, 255), font=f_leg_b if idx < 3 else f_leg)

    foot_y = 1968
    draw_dashed_line(draw, (225, foot_y + 13), (280, foot_y + 13), fill=col_unopt, width=5)
    draw_circle(draw, 252, foot_y + 13, 8, fill=col_unopt)
    draw.text(
        (295, foot_y),
        items[4][2]
        + "   |   Hardware: 16-chip TPU v6e (4×4 2D Mesh)   |   Checkpoint: gs://dsv4-flash-jawadamin-asia-ne1/deepseek-v4.1-flash",
        fill=(71, 85, 105, 255),
        font=f_foot,
    )

    final_img = img.convert("RGB").resize((W // 2, H // 2), resample=Image.Resampling.LANCZOS)
    out_paths = [
        repo_root
        / "models/DeepSeekV4.1-Flash-v6e16/megakernel-recipe/results/charts/megakernel-vs-fused-vs-base-xla.png",
        repo_root
        / "models/DeepSeekV4.1-Flash-v6e16/kernel-optimizations-recipe/results/charts/v41-three-recipes-throughput-latency.png",
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        final_img.save(p, format="PNG", optimize=True)
        print(f"Saved {p} ({p.stat().st_size:,} bytes)")


def generate_megakernel_detail_chart(repo_root: Path) -> None:
    results_path = (
        repo_root
        / "models/DeepSeekV4.1-Flash-v6e16/megakernel-recipe/results/megakernel_v6e16_results.json"
    )
    data = json.loads(results_path.read_text())

    W, H = 3600, 1600
    img = Image.new("RGBA", (W, H), (15, 23, 42, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    f_title = load_font(42, bold=True)
    f_sub = load_font(25, bold=False)
    f_panel = load_font(31, bold=True)
    f_axis = load_font(24, bold=True)
    f_tick = load_font(22, bold=True)
    f_leg = load_font(21, bold=False)
    f_val = load_font(21, bold=True)
    f_val_sm = load_font(19, bold=True)

    title = "DeepSeek-V4.1-Flash (40L + 3 MTP, 384 MXFP4 Experts Top-6, mHC x4, CSA+SWA, 200.5 GB Engram) on TPU v6e-16"
    tb = draw.textbbox((0, 0), title, font=f_title)
    draw.text(((W - (tb[2] - tb[0])) / 2, 24), title, fill=(248, 250, 252, 255), font=f_title)

    sub = "Real Checkpoint gs://dsv4-flash-jawadamin-asia-ne1/deepseek-v4.1-flash  |  GSM8K/STEM: 16/16 (100.0%, enable_engram=True)  |  Prompt Top-1 Parity: 98.24% Tie-Aware"
    sb = draw.textbbox((0, 0), sub, font=f_sub)
    draw.text(((W - (sb[2] - sb[0])) / 2, 86), sub, fill=(148, 163, 184, 255), font=f_sub)

    # Panel 1: Decode TPOT Ladder C=1..16
    p1_l, p1_t, p1_r, p1_b = 165, 220, 1700, 1400
    p1_w, p1_h = p1_r - p1_l, p1_b - p1_t
    draw.rectangle([p1_l, p1_t, p1_r, p1_b], fill=(30, 41, 59, 255), outline=(71, 85, 105, 255), width=2)

    t1 = "1. Decode TPOT Ladder: Pallas Megakernel (+Engram) vs. vLLM (1K Context)"
    tb1 = draw.textbbox((0, 0), t1, font=f_panel)
    draw.text((p1_l + (p1_w - (tb1[2] - tb1[0])) / 2, 162), t1, fill=(248, 250, 252, 255), font=f_panel)

    ladder = data["accuracy_and_tpot_validation"]["concurrency_ladder_1k_context"]
    concs = [1, 2, 4, 8, 16]
    ymax1 = 55.0

    def y1(v: float) -> float:
        return p1_b - (v / ymax1) * (p1_h - 60)

    for gv in range(0, 51, 10):
        yy = y1(gv)
        draw.line([(p1_l, yy), (p1_r, yy)], fill=(51, 65, 85, 255), width=2)
        lbl = f"{gv} ms"
        bb = draw.textbbox((0, 0), lbl, font=f_tick)
        draw.text((p1_l - (bb[2] - bb[0]) - 14, yy - (bb[3] - bb[1]) / 2), lbl, fill=(203, 213, 225, 255), font=f_tick)

    col_vllm = (100, 116, 139, 255)
    col_mk = (14, 165, 233, 255)
    col_dsp = (16, 185, 129, 255)

    for idx, c in enumerate(concs):
        cx = p1_l + 150 + idx * ((p1_w - 300) / 4.0)
        entry = ladder[f"C={c}"]
        v_ms = entry["vllm_median_tpot_ms"]
        m_ms = entry["mk_median_tpot_ms"]
        sp = entry["speedup_vs_vllm"]

        bw = 74
        # vLLM bar
        x0 = cx - bw - 6
        y0 = y1(v_ms)
        draw.rectangle([x0, y0, x0 + bw, p1_b], fill=col_vllm, outline=(203, 213, 225, 255), width=2)
        vt = f"{v_ms:.2f} ms"
        vb = draw.textbbox((0, 0), vt, font=f_val_sm)
        draw.text((x0 + (bw - (vb[2] - vb[0])) / 2, y0 - 28), vt, fill=(226, 232, 240, 255), font=f_val_sm)

        # Megakernel bar
        x1 = cx + 6
        ym = y1(m_ms)
        draw.rectangle([x1, ym, x1 + bw, p1_b], fill=col_mk, outline=(224, 242, 254, 255), width=2)
        for li, line in enumerate([f"{m_ms:.2f} ms", f"({sp:.2f}x)"]):
            mb = draw.textbbox((0, 0), line, font=f_val_sm)
            draw.text((x1 + (bw - (mb[2] - mb[0])) / 2, ym - 48 + li * 22), line, fill=(56, 189, 248, 255), font=f_val_sm)

        if c == 1:
            d_ms = data["dspark_speculative_decoding"]["effective_tpot_speedup"]["spec_median_effective_tpot_ms"]
            d_sp = v_ms / d_ms
            xd = x1 + bw + 10
            yd = y1(d_ms)
            draw.rectangle([xd, yd, xd + 62, p1_b], fill=col_dsp, outline=(209, 250, 229, 255), width=2)
            for li, line in enumerate([f"{d_ms:.2f}ms", f"({d_sp:.2f}x)"]):
                db = draw.textbbox((0, 0), line, font=f_val_sm)
                draw.text((xd + (62 - (db[2] - db[0])) / 2, yd - 48 + li * 22), line, fill=(52, 211, 153, 255), font=f_val_sm)

        cl = f"C = {c}"
        cb = draw.textbbox((0, 0), cl, font=f_tick)
        draw.text((cx - (cb[2] - cb[0]) / 2, p1_b + 16), cl, fill=(248, 250, 252, 255), font=f_tick)

    ax1 = "Active Decode Concurrency at 1,024-Token Context (60 Timed Iterations, enable_engram=True, Crossover > 16)"
    ab1 = draw.textbbox((0, 0), ax1, font=f_axis)
    draw.text((p1_l + (p1_w - (ab1[2] - ab1[0])) / 2, p1_b + 66), ax1, fill=(248, 250, 252, 255), font=f_axis)

    # Legend Box Panel 1
    draw.rectangle([p1_l + 35, p1_t + 25, p1_l + 960, p1_t + 190], fill=(15, 23, 42, 235), outline=(71, 85, 105, 255), width=2)
    leg1 = [
        (col_vllm, "Production vLLM Batched XLA (FP8 KV + MXFP4, tpu-inference)"),
        (col_mk, "40-Layer Pallas Decode Megakernel + 200.5 GB Engram (9.30 ms @ C=1, 4.82x)"),
        (col_dsp, "C=1 Megakernel + Engram + DSpark (mtp.0..2): 7.27 ms med / 4.36 ms best (10.28x)"),
    ]
    for li, (col, txt) in enumerate(leg1):
        ly = p1_t + 42 + li * 46
        draw.rectangle([p1_l + 55, ly, p1_l + 85, ly + 26], fill=col, outline=(255, 255, 255, 220), width=2)
        draw.text((p1_l + 102, ly - 1), txt, fill=(226, 232, 240, 255), font=f_leg)

    # Panel 2: DSpark Speculative Decoding across 8 Prompts
    p2_l, p2_t, p2_r, p2_b = 1920, 220, 3500, 1400
    p2_w, p2_h = p2_r - p2_l, p2_b - p2_t
    draw.rectangle([p2_l, p2_t, p2_r, p2_b], fill=(30, 41, 59, 255), outline=(71, 85, 105, 255), width=2)

    t2 = "2. DSpark (mtp.0..2) Lossless Speculative Decoding across 8 Prompts (Engram ON)"
    tb2 = draw.textbbox((0, 0), t2, font=f_panel)
    draw.text((p2_l + (p2_w - (tb2[2] - tb2[0])) / 2, 162), t2, fill=(248, 250, 252, 255), font=f_panel)

    ymax2 = 14.0

    def y2(v: float) -> float:
        return p2_b - (v / ymax2) * (p2_h - 60)

    for gv in range(0, 13, 2):
        yy = y2(gv)
        draw.line([(p2_l, yy), (p2_r, yy)], fill=(51, 65, 85, 255), width=2)
        lbl = f"{gv} ms"
        bb = draw.textbbox((0, 0), lbl, font=f_tick)
        draw.text((p2_l - (bb[2] - bb[0]) - 14, yy - (bb[3] - bb[1]) / 2), lbl, fill=(203, 213, 225, 255), font=f_tick)

    prompts = data["dspark_speculative_decoding"]["per_prompt_results"]
    short_names = ["p0_cap", "p1_arith", "p2_py", "p3_phys", "p4_chem", "p5_sort", "p6_lalg", "p7_tpu"]

    med_spec = data["dspark_speculative_decoding"]["effective_tpot_speedup"]["spec_median_effective_tpot_ms"]
    draw.line([(p2_l, y2(med_spec)), (p2_r, y2(med_spec))], fill=col_dsp, width=3)

    for idx, pr in enumerate(prompts):
        cx = p2_l + 105 + idx * ((p2_w - 210) / 7.0)
        ns_ms = pr["nonspec_mean_tpot_ms"]
        sp_ms = pr["spec_effective_tpot_ms"]
        acc_tok = pr["mean_tokens_per_verify_step"]
        bw = 52

        x0 = cx - bw - 4
        y0 = y2(ns_ms)
        draw.rectangle([x0, y0, x0 + bw, p2_b], fill=col_mk, outline=(224, 242, 254, 255), width=2)

        x1 = cx + 4
        y1_p = y2(sp_ms)
        draw.rectangle([x1, y1_p, x1 + bw, p2_b], fill=col_dsp, outline=(209, 250, 229, 255), width=2)

        for li, line in enumerate([f"{sp_ms:.2f}ms", f"{acc_tok:.2f}tok"]):
            sb = draw.textbbox((0, 0), line, font=f_val_sm)
            draw.text((x1 + (bw - (sb[2] - sb[0])) / 2, y1_p - 46 + li * 21), line, fill=(52, 211, 153, 255), font=f_val_sm)

        sn = short_names[idx]
        nb = draw.textbbox((0, 0), sn, font=f_tick)
        draw.text((cx - (nb[2] - nb[0]) / 2, p2_b + 16), sn, fill=(248, 250, 252, 255), font=f_tick)

    ax2 = "Golden Prompts (256 Greedy Tokens Each — 2,048 / 2,048 Lossless Token Match = 100.0%)"
    ab2 = draw.textbbox((0, 0), ax2, font=f_axis)
    draw.text((p2_l + (p2_w - (ab2[2] - ab2[0])) / 2, p2_b + 66), ax2, fill=(248, 250, 252, 255), font=f_axis)

    draw.rectangle([p2_l + 35, p2_t + 25, p2_l + 1090, p2_t + 190], fill=(15, 23, 42, 235), outline=(71, 85, 105, 255), width=2)
    leg2 = [
        (col_mk, "Non-Speculative Pallas Megakernel + Engram (10.22 ms/tok on 256-tok trajectory)"),
        (col_dsp, "Megakernel + Engram + DSpark mtp.0..2 (3.15 tok/step mean, 53.7% draft accept)"),
        (col_dsp, "Median Effective TPOT: 7.27 ms/tok (1.41x med, 2.34x peak @ 4.36 ms on p0_cap)"),
    ]
    for li, (col, txt) in enumerate(leg2):
        ly = p2_t + 42 + li * 46
        if li < 2:
            draw.rectangle([p2_l + 55, ly, p2_l + 85, ly + 26], fill=col, outline=(255, 255, 255, 220), width=2)
        else:
            draw.line([(p2_l + 55, ly + 13), (p2_l + 85, ly + 13)], fill=col, width=4)
        draw.text((p2_l + 102, ly - 1), txt, fill=(226, 232, 240, 255), font=f_leg)

    final_img = img.convert("RGB").resize((W // 2, H // 2), resample=Image.Resampling.LANCZOS)
    out_p = (
        repo_root
        / "models/DeepSeekV4.1-Flash-v6e16/megakernel-recipe/results/charts/megakernel-vs-batched-xla.png"
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    final_img.save(out_p, format="PNG", optimize=True)
    print(f"Saved {out_p} ({out_p.stat().st_size:,} bytes)")


def main() -> None:
    repo_root = Path("/usr/local/google/home/jawadamin/Repos/DSV4-Flash-Deployment")
    generate_three_recipe_chart(repo_root)
    generate_megakernel_detail_chart(repo_root)


if __name__ == "__main__":
    main()
