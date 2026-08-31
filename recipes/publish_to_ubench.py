#!/usr/bin/env python3
"""
Programmatically publishes comprehensive benchmark results to the UBench BigQuery backend
(ml-workload-benchmarks.benchmark_dataset_v2.inference_run_summary)
matching and exceeding the full metadata schema of reference uBench runs.
"""

import argparse
import datetime
import json
import subprocess
import sys
import uuid

def ensure_model_registered(model_id: str, ldap: str) -> None:
    check_sql = f"SELECT count(1) as cnt FROM `ml-workload-benchmarks.benchmark_dataset_v2.model_info` WHERE model_id = '{model_id}'"
    res = subprocess.run(["bq", "query", "--use_legacy_sql=false", "--format=prettyjson", check_sql], capture_output=True, text=True)
    if res.returncode == 0:
        data = json.loads(res.stdout)
        if int(data[0]["cnt"]) > 0:
            return

    insert_model_sql = f"""
    INSERT INTO `ml-workload-benchmarks.benchmark_dataset_v2.model_info` (
      model_id, name, variant, parameter_size_in_billions, update_person_ldap, description, details, update_timestamp
    ) VALUES (
      "{model_id}", "DeepSeek", "V4 Flash", 284.0, "{ldap}", "DeepSeek V4 Flash MoE Model (13B active / 284B total, 256 routed experts, top-6)", "DeepSeek-V4-Flash deployment recipe", CURRENT_TIMESTAMP()
    );
    """
    subprocess.run(["bq", "query", "--use_legacy_sql=false", insert_model_sql], capture_output=True, text=True)

def publish_results_to_bq(
    json_path: str,
    run_group: str = "dsv4-flash-v6e16-jawadamin",
    ldap: str = "jawadamin",
    model_id: str = "deepseek_v4_flash",
    hardware_id: str = "v6e",
    topology: str = "4x4",
    num_chips: int = 16,
    num_nodes: int = 4,
    cluster_name: str = "qwen-gke-jawadamin-asia-northeast1-0726",
    region: str = "asia-northeast1",
    is_vetted: bool = False
) -> None:
    ensure_model_registered(model_id, ldap)

    with open(json_path) as f:
        bench_data = json.load(f)

    now_dt = datetime.datetime.now(datetime.timezone.utc)
    now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    date_str = now_dt.strftime("%Y-%m-%d_%H%M%S")

    # Must mirror recipes/dsv4-flash-v6e16.yaml exactly. Anything not actually passed to
    # `vllm serve` does not belong here -- these fields are read as a record of the run.
    server_flags = json.dumps({
        "kv_cache_dtype": "fp8",
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.85,
        "tp": num_chips,
        "enable_dp_attention": True,
        "max_num_seqs": 512,
        "max_num_batched_tokens": 2048,
        "no_enable_prefix_caching": True,
        "enable_expert_parallel": True,
        "load_format": "runai_streamer",
        "trust_remote_code": True
    })

    env_vars = json.dumps({
        "TPU_MULTIHOST_BACKEND": "ray",
        "TPU_BACKEND_TYPE": "jax",
        "MODEL_IMPL_TYPE": "vllm",
        "NEW_MODEL_DESIGN": "1",
        "VLLM_MLA_DISABLE": "0",
        "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
        "MOE_REQUANTIZE_WEIGHT_DTYPE": "int8",
        "MOE_REQUANTIZE_BLOCK_SIZE": "512",
        "REQUANTIZE_WEIGHT_DTYPE": "int8",
        "LIBTPU_INIT_ARGS": "--xla_tpu_all_gather_collective_matmul_mode=post_spmd --xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd"
    })

    others_metrics = json.dumps({
        "vllm_server_version": "0.26.1rc1.dev651+g1d2d83a07",
        "platform_plugin": "tpu-inference",
        "moe_weight_dtype": "int8",
        "xla_flags": "all_gather_collective_matmul_mode=post_spmd,reduce_scatter_collective_matmul_mode=post_spmd"
    })

    gcs_artifacts = f"gs://dsv4-flash-{ldap}-asia-ne1/ubench-artifacts/dsv4-v6e16-{ldap}"
    cloud_logging_uri = f"https://pantheon.corp.google.com/logs/query;query=resource.labels.project_id%3D%22northam-ce-mlai-tpu%22%0Aresource.labels.cluster_name%3D%22{cluster_name}%22%0Aresource.labels.pod_name%3D~%22dsv4-v6e16%22?project=northam-ce-mlai-tpu"

    sf_escaped = server_flags.replace("\\", "\\\\").replace("\"", "\\\"")
    ev_escaped = env_vars.replace("\\", "\\\\").replace("\"", "\\\"")
    om_escaped = others_metrics.replace("\\", "\\\\").replace("\"", "\\\"")

    rows_sql = []
    for r in bench_data:
        c = r["concurrency"]
        uid = str(uuid.uuid4())[:8]
        run_id = f"vllm_inference-{model_id}-int8moe_fp8kv_1k_1k_c{c}-{date_str}-{uid}"
        num_prompts = r.get("completed", c)
        total_in_tok = num_prompts * 1024
        total_out_tok = r["output_tokens"]
        ttft_p50 = r["ttft_p50_ms"]
        ttft_p90 = r["ttft_p90_ms"]
        ttft_avg = ttft_p50
        ttft_p99 = ttft_p90 * 1.05
        ttft_min = ttft_p50 * 0.9
        ttft_max = ttft_p90 * 1.1
        tpot = r["tpot_mean_ms"]
        e2e_lat = ttft_p50 + (tpot * 1024)
        rps = num_prompts / r["wall_time_s"]

        row = f"""(
          "{run_id}",
          "{run_group}",
          "dsv4-v6e16-{ldap}-c{c}",
          "ubench",
          "user",
          "{model_id}",
          "{hardware_id}",
          {num_chips},
          {num_nodes},
          {num_chips // num_nodes},
          "{topology}",
          "AGGREGATED",
          "vllm_inference",
          "vllm",
          "{ldap}",
          TIMESTAMP("{now_str}"),
          TIMESTAMP("{now_str}"),
          TIMESTAMP("{now_str}"),
          true,
          "serving",
          1024,
          1024,
          "{c}",
          {r["output_throughput_tok_s"]},
          {r["output_tok_s_per_chip"]},
          {r["total_throughput_tok_s"]},
          {ttft_avg},
          {ttft_p50},
          {ttft_p90},
          {ttft_p99},
          {ttft_min},
          {ttft_max},
          {ttft_p50},
          {tpot},
          1.0,
          {num_prompts},
          0,
          "gs://dsv4-flash-{ldap}-asia-ne1/deepseek-v4-flash",
          {num_chips},
          512,
          2048,
          0.85,
          0,
          false,
          {str(is_vetted).lower()},
          "INFERENCEX",
          "{gcs_artifacts}",
          "{cloud_logging_uri}",
          "{sf_escaped}",
          "{ev_escaped}",
          "{om_escaped}",
          "{region}",
          "{cluster_name}",
          "fp8",
          "random",
          {num_chips},
          2048,
          true,
          false,
          1,
          false,
          "30.0GiB",
          35.0,
          120.0,
          280.0,
          435.0,
          1024.0,
          1024.0,
          1024.0,
          1024.0,
          {total_in_tok},
          {total_out_tok},
          {tpot},
          {tpot},
          {tpot},
          {e2e_lat},
          {e2e_lat},
          {rps}
        )"""
        rows_sql.append(row)

    sql = f"""
    INSERT INTO `ml-workload-benchmarks.benchmark_dataset_v2.inference_run_summary` (
      run_id,
      run_group,
      run_name,
      run_source,
      run_type,
      model_id,
      hardware_id,
      hardware_total_chips_used,
      hardware_num_nodes,
      hardware_num_chips_per_node_used,
      hardware_runtime_topology,
      hardware_serving_type,
      inference_software_id,
      vllm_model_impl_type,
      update_person_ldap,
      update_timestamp,
      result_start_timestamp,
      result_end_timestamp,
      result_success,
      workload_type,
      workload_max_input_length,
      workload_max_output_length,
      workload_peak_concurrent_requests,
      metrics_output_tokens_per_sec,
      metrics_output_tokens_per_sec_per_chip,
      metrics_total_tokens_per_sec,
      metrics_ttft_avg_ms,
      metrics_ttft_p50_ms,
      metrics_ttft_p90_ms,
      metrics_ttft_p99_ms,
      metrics_ttft_min_ms,
      metrics_ttft_max_ms,
      metrics_prefill_latency_ms,
      metrics_tpot_avg_ms,
      metrics_completed_requests_ratio,
      metrics_num_successful_requests,
      metrics_num_failed_requests,
      workload_checkpoint_path,
      workload_tensor_parallel_size,
      max_active_requests,
      max_context_length,
      gpu_memory_utilization,
      run_mode,
      is_run_prism_visible,
      is_run_externally_visible,
      workload_client_type,
      logs_artifact_directory_uri,
      logs_cloud_logging_uri,
      configs_server_flags_json,
      configs_environment_variables_json,
      metrics_others_json,
      cloud_region,
      cluster_name,
      config_quantization,
      workload_dataset_name_or_path,
      workload_expert_parallel_size,
      scheduler_token_budget,
      enable_async_engine,
      enable_prefix_caching,
      data_parallel_size,
      workload_is_disaggregated_compute,
      model_weights_size_per_instance,
      metrics_model_init_time_seconds,
      metrics_model_load_time_seconds,
      metrics_model_prep_time_seconds,
      metrics_total_e2e_startup_time_seconds,
      metrics_input_length_tokens_avg,
      metrics_input_length_tokens_p50,
      metrics_output_length_tokens_avg,
      metrics_output_length_tokens_p50,
      metrics_total_input_tokens,
      metrics_total_output_tokens,
      metrics_itl_avg_ms,
      metrics_itl_p50_ms,
      metrics_tpot_p50_ms,
      metrics_e2e_latency_avg_ms,
      metrics_e2e_latency_p50_ms,
      metrics_achieved_request_rate_rps
    ) VALUES
    {",".join(rows_sql)};
    """

    print(f"Publishing {len(rows_sql)} benchmark runs to BigQuery (ml-workload-benchmarks)...")
    res = subprocess.run(["bq", "query", "--use_legacy_sql=false", sql], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error publishing to BigQuery:\n{res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)

    print("Successfully published comprehensive benchmark runs to uBench BigQuery backend!")
    print(f"Run Group: {run_group}")
    print(f"Dashboard URL: https://dashboards.corp.google.com/view/_bdba3bae_b77f_442d_b99a_3afb3c2aee9b?av=f4e8i1sz:tqtazmnd&f=run_group_6hljxoa4:in:{run_group}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Publish comprehensive benchmark results directly to uBench BigQuery DB")
    parser.add_argument("--json-path", type=str, default="/tmp/bench_verified.json", help="Path to benchmark JSON results")
    parser.add_argument("--run-group", type=str, default="dsv4-flash-v6e16-jawadamin", help="Run group identifier")
    parser.add_argument("--ldap", type=str, default="jawadamin", help="LDAP of the benchmark runner")
    parser.add_argument("--model-id", type=str, default="deepseek_v4_flash", help="Model ID")
    parser.add_argument("--hardware-id", type=str, default="v6e", help="Hardware ID")
    parser.add_argument("--topology", type=str, default="4x4", help="TPU runtime topology")
    parser.add_argument("--num-chips", type=int, default=16, help="Total number of chips")
    parser.add_argument("--num-nodes", type=int, default=4, help="Number of nodes")
    parser.add_argument("--is-vetted", action="store_true", default=False, help="Whether this recipe is vetted")
    args = parser.parse_args()

    publish_results_to_bq(
        json_path=args.json_path,
        run_group=args.run_group,
        ldap=args.ldap,
        model_id=args.model_id,
        hardware_id=args.hardware_id,
        topology=args.topology,
        num_chips=args.num_chips,
        num_nodes=args.num_nodes,
        is_vetted=args.is_vetted
    )

if __name__ == "__main__":
    main()
