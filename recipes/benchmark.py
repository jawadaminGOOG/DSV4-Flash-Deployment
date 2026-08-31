#!/usr/bin/env python3
"""
DeepSeek-V4-Flash Serving Benchmark Suite
Evaluates throughput, TTFT, and TPOT on TPU v6e.

Workload:
- Exact token ID sequences matching Input Sequence Length (ISL, default: 1024)
- Fixed Output Sequence Length (OSL, default: 1024)
- Streaming Server-Sent Events (SSE) with ignore_eos=True and burst concurrency
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

def calc_percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    d = k - f
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * d

def calc_mean(data: List[float]) -> float:
    return statistics.mean(data) if data else 0.0

async def send_async_streaming_request(
    host: str,
    port: int,
    model: str,
    prompt_token_ids: List[int],
    req_id: int,
    isl: int,
    osl: int,
    timeout: float = 300.0
) -> Optional[Dict[str, Any]]:
    payload = json.dumps({
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": osl,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True}
    })
    
    req_bytes = (
        f"POST /v1/completions HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(payload.encode('utf-8'))}\r\n"
        f"Connection: close\r\n\r\n"
        f"{payload}"
    ).encode("utf-8")

    t_start = time.perf_counter()
    ttft = None
    t_prev = t_start
    inter_token_latencies: List[float] = []
    output_tokens_count = 0
    prompt_tokens_count = isl
    
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=30.0
        )
        writer.write(req_bytes)
        await writer.drain()

        # Read HTTP status line
        status_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        status_str = status_line.decode("utf-8", errors="ignore").strip()
        if not status_str.startswith("HTTP/1.1 200") and not status_str.startswith("HTTP/1.0 200"):
            print(f"Request {req_id} failed with HTTP status: {status_str}", file=sys.stderr)
            writer.close()
            await writer.wait_closed()
            return None

        # Read remaining HTTP headers
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if not line or line in (b"\r\n", b"\n"):
                break

        # Read chunked SSE stream
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if not line:
                break
            line_str = line.decode("utf-8", errors="ignore").strip()
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                if "usage" in data and data["usage"]:
                    usage = data["usage"]
                    output_tokens_count = usage.get("completion_tokens", output_tokens_count)
                    prompt_tokens_count = usage.get("prompt_tokens", prompt_tokens_count)
                choices = data.get("choices", [])
                if choices:
                    text_chunk = choices[0].get("text", "")
                    if text_chunk:
                        t_now = time.perf_counter()
                        if ttft is None:
                            ttft = t_now - t_start
                        else:
                            inter_token_latencies.append(t_now - t_prev)
                        t_prev = t_now
                        output_tokens_count += 1
            except Exception:
                pass

        writer.close()
        await writer.wait_closed()
    except Exception as e:
        print(f"Request {req_id} failed with exception: {e}", file=sys.stderr)
        return None

    t_end = time.perf_counter()
    total_latency = t_end - t_start

    if output_tokens_count == 0:
        print(f"Request {req_id} generated 0 tokens.", file=sys.stderr)
        return None

    return {
        "req_id": req_id,
        "total_latency": total_latency,
        "ttft": ttft if ttft is not None else total_latency,
        "inter_token_latencies": inter_token_latencies,
        "output_tokens": output_tokens_count,
        "prompt_tokens": prompt_tokens_count,
    }

async def run_benchmark_point(
    host: str,
    port: int,
    model: str,
    concurrency: int,
    isl: int,
    osl: int,
    num_prompts: int,
    num_chips: int
) -> Dict[str, Any]:
    print(f"\n{'='*75}")
    print(f"Running Benchmark: Concurrency={concurrency}, Prompts={num_prompts}")
    print(f"Workload Config: ISL={isl}, OSL={osl}, ignore_eos=True, rate=inf")
    print(f"{'='*75}")
    
    # Generate exact token ID sequences of length ISL
    prompt_token_ids = [
        [100 + ((i * 37 + j) % 500) for j in range(isl)]
        for i in range(num_prompts)
    ]
    
    # Warmup on low concurrency
    if concurrency <= 16:
        await send_async_streaming_request(host, port, model, prompt_token_ids[0], -1, isl, min(16, osl))
        
    sem = asyncio.Semaphore(concurrency)
    
    async def bounded_req(i: int):
        async with sem:
            return await send_async_streaming_request(host, port, model, prompt_token_ids[i], i, isl, osl)
            
    wall_start = time.perf_counter()
    tasks = [asyncio.create_task(bounded_req(i)) for i in range(num_prompts)]
    results = await asyncio.gather(*tasks)
    wall_end = time.perf_counter()
    
    valid_results = [r for r in results if r is not None]
    wall_time = wall_end - wall_start
    
    total_prompt_tok = sum(r["prompt_tokens"] for r in valid_results)
    total_output_tok = sum(r["output_tokens"] for r in valid_results)
    total_tok = total_prompt_tok + total_output_tok
    
    output_throughput = total_output_tok / wall_time if wall_time > 0 else 0.0
    total_throughput = total_tok / wall_time if wall_time > 0 else 0.0
    
    ttfts = [r["ttft"] * 1000 for r in valid_results]
    latencies = [r["total_latency"] * 1000 for r in valid_results]
    
    all_itls: List[float] = []
    for r in valid_results:
        all_itls.extend(r["inter_token_latencies"])
    tpot = (calc_mean(all_itls) * 1000) if all_itls else 0.0
    
    print(f"Results (Concurrency={concurrency}):")
    print(f"  Completed Requests  : {len(valid_results)}/{num_prompts}")
    print(f"  Wall Time           : {wall_time:.2f} s")
    print(f"  Output Tokens       : {total_output_tok}")
    print(f"  Output Throughput   : {output_throughput:.2f} tok/s ({output_throughput/num_chips:.2f} tok/s/chip)")
    print(f"  Aggregate Throughput: {total_throughput:.2f} tok/s")
    print(f"  TTFT P50            : {calc_percentile(ttfts, 50):.2f} ms")
    print(f"  TTFT P90            : {calc_percentile(ttfts, 90):.2f} ms")
    print(f"  TPOT (Mean)         : {tpot:.2f} ms")
    print(f"  Per-stream Speed    : {(1000.0 / tpot if tpot > 0 else 0):.2f} tok/s")
    print(f"  Latency P50         : {calc_percentile(latencies, 50):.2f} ms")
    
    return {
        "concurrency": concurrency,
        "num_prompts": num_prompts,
        "completed": len(valid_results),
        "wall_time_s": wall_time,
        "output_tokens": total_output_tok,
        "output_throughput_tok_s": output_throughput,
        "output_tok_s_per_chip": output_throughput / num_chips,
        "total_throughput_tok_s": total_throughput,
        "ttft_p50_ms": float(calc_percentile(ttfts, 50)),
        "ttft_p90_ms": float(calc_percentile(ttfts, 90)),
        "tpot_mean_ms": float(tpot),
    }

async def main_async(args: argparse.Namespace) -> None:
    summary: List[Dict[str, Any]] = []
    for c in args.concurrencies:
        num_prompts = max(c * args.prompts_multiplier, 32)
        res = await run_benchmark_point(
            host=args.host,
            port=args.port,
            model=args.model,
            concurrency=c,
            isl=args.isl,
            osl=args.osl,
            num_prompts=num_prompts,
            num_chips=args.num_chips
        )
        summary.append(res)
        await asyncio.sleep(2)
        
    print("\n" + "="*95)
    print(f"DEEPSEEK-V4-FLASH BENCHMARK SUMMARY (ISL={args.isl}, OSL={args.osl}, ignore_eos=True)")
    print(f"Target: {args.host}:{args.port} | Chips: {args.num_chips}")
    print("="*95)
    print(f"{'Concurrency':<12} | {'Output tok/s':<14} | {'Tok/s/Chip':<12} | {'Agg tok/s':<14} | {'TTFT P50':<12} | {'TPOT':<10}")
    print("-" * 95)
    for s in summary:
        print(f"{s['concurrency']:<12} | {s['output_throughput_tok_s']:<14.1f} | {s['output_tok_s_per_chip']:<12.1f} | {s['total_throughput_tok_s']:<14.1f} | {s['ttft_p50_ms']:<10.1f} ms | {s['tpot_mean_ms']:<8.2f} ms")
    print("="*95)
    
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {args.output_json}")

def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Flash Serving Benchmark Suite")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host IP or domain")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V4-Flash", help="Model name")
    parser.add_argument("--isl", type=int, default=1024, help="Input Sequence Length (tokens)")
    parser.add_argument("--osl", type=int, default=1024, help="Output Sequence Length (tokens)")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512], help="Concurrency levels to sweep")
    parser.add_argument("--prompts-multiplier", type=int, default=2, help="Multiplier for num_prompts = max(C * multiplier, 32)")
    parser.add_argument("--num-chips", type=int, default=16, help="Number of accelerator chips for per-chip calculations")
    parser.add_argument("--output-json", type=str, default="", help="Optional path to save JSON results")
    args = parser.parse_args()
    
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
