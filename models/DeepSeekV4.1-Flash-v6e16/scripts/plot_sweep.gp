# DeepSeek-V4.1-Flash against DeepSeek-V4-Flash on TPU v6e-16.
# Three panels: output throughput, TTFT p90, and TPOT, each against concurrency.
#
# The x axis is log2 so the ten concurrency levels are evenly spaced. The TTFT
# axis is log10 because the values span 500 ms to 34,300 ms and a linear axis
# hides everything below C=64.
#
# Column 1 is concurrency. Columns 2 and 3 are throughput, 4 and 5 are TTFT p90,
# 6 and 7 are TPOT. The odd column of each pair is V4.1; the even column is V4.
#
# Both arms ran the same client, the same 1k-in / 1k-out shape, and the same
# patched tpu-inference build. They differ in two settings that the V4.1 memory
# budget forced: V4.1 ran --max-model-len 2048 and --max-num-batched-tokens 256,
# against 9216 and 2048 for V4. The prefill-batch difference is 8x and it works
# against V4.1 at every point.
#
# Usage: gnuplot -e "outfile='out.png'" plot_v41_sweep.gp

$d << EOD
1        18.9     35.3     12000      1235   49.12   27.69
2        39.5     70.7      1126       505   49.62   27.81
4        78.7    140.5      1093       504   49.79   28.01
8       154.4    277.5      1040       508   50.84   28.36
16      444.5    598.6      1112       500   34.94   26.26
32      821.6   1125.4      2469      1645   36.95   26.85
64     1601.4   2089.5      5298      2613   36.50   28.83
128    2778.7   4113.0      9758      3869   40.03   28.60
256    4096.9   6671.9     17317      7511   51.39   33.74
512    5137.3   8463.7     34300     18745   77.31   50.19
EOD

set terminal pngcairo size 1680,700 enhanced font 'DejaVu Sans,11' background rgb 'white'
set output outfile

blue   = '#2a78d6'
orange = '#eb6834'
ink    = '#1a1a1a'
muted  = '#6b6b6b'
grid   = '#e3e3e3'

set logscale x 2
set xrange [0.8:640]
set xtics (1,2,4,8,16,32,64,128,256,512) nomirror textcolor rgb muted
set mxtics 1
set ytics nomirror textcolor rgb muted
set grid xtics ytics lc rgb grid lw 1 back
set border 3 lc rgb grid
set tics scale 0.4
set key noautotitle
set xlabel 'Concurrency' textcolor rgb muted offset 0,0.4

set style line 1 lc rgb blue   lw 3 pt 7 ps 1.4
set style line 2 lc rgb orange lw 3 pt 5 ps 1.3

set multiplot layout 1,3 margins 0.050,0.990,0.140,0.790 spacing 0.070,0.10

set label 21 "DeepSeek-V4.1-Flash on TPU v6e-16 - 1k in / 1k out" \
    at screen 0.5,0.955 center textcolor rgb ink font 'DejaVu Sans Bold,17'
set label 22 "V4.1-Flash, patched tpu-inference with the RoPE byte-plane fix, against V4-Flash on the same slice and the same client.   1,028 requests, 0 failures, greedy sampling, kv-cache-dtype fp8." \
    at screen 0.5,0.905 center textcolor rgb muted font 'DejaVu Sans,10.5'
set label 23 "V4.1 ran --max-model-len 2048 and --max-num-batched-tokens 256; V4 ran 9216 and 2048. The 8x prefill-batch difference works against V4.1 at every point." \
    at screen 0.5,0.030 center textcolor rgb muted font 'DejaVu Sans,9.5'

# ------------------------------- throughput ---------------------------------
set ylabel 'Output throughput (tok/s)' textcolor rgb muted offset 0.5,0
set title 'Throughput - V4.1 peaks at 5,137 tok/s, 321 tok/s/chip' \
    textcolor rgb ink font 'DejaVu Sans,12.5' offset 0,0.5
set yrange [0:9000]
set format y '%.0f'
set key top left reverse Left samplen 1.6 spacing 1.35 textcolor rgb ink
plot $d using 1:2 with linespoints ls 1 title ' V4.1-Flash', \
     $d using 1:3 with linespoints ls 2 title ' V4-Flash'
unset key
unset label 21
unset label 22
unset label 23

# ---------------------------------- TTFT ------------------------------------
set ylabel 'TTFT p90 (ms)' textcolor rgb muted offset 0.5,0
set title 'TTFT p90 - the gap is prefill, not decode' \
    textcolor rgb ink font 'DejaVu Sans,12.5' offset 0,0.5
set logscale y 10
set yrange [300:60000]
set ytics (500,1000,2000,5000,10000,20000,50000) format '%.0f'
set label 31 "C=1 is the sampler compile,\nnot steady state" at 1.15,12000 left \
    textcolor rgb muted font 'DejaVu Sans,9'
plot $d using 1:4 with linespoints ls 1, \
     $d using 1:5 with linespoints ls 2
unset label 31
unset logscale y

# ---------------------------------- TPOT ------------------------------------
set ylabel 'TPOT mean (ms per token)' textcolor rgb muted offset 0.5,0
set title 'TPOT - flat to C=128, then both arms saturate' \
    textcolor rgb ink font 'DejaVu Sans,12.5' offset 0,0.5
set yrange [0:90]
set ytics autofreq 15 format '%.0f'
plot $d using 1:6 with linespoints ls 1, \
     $d using 1:7 with linespoints ls 2

unset multiplot
unset output
