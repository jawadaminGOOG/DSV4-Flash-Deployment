"""Renders the GKE JobSet that runs one command on a tpu7x slice.

The three capacity tiers share one shape: ``hosts`` pods, each owning every chip of one
tpu7x VM, pinned to one node pool. Multi-host pods reach process 0 through the JobSet's
headless service, and each pod's process id is its ``JOB_COMPLETION_INDEX``.

    python jobset.py --name probe-c-spot --tier c --pool v7x16-spot-jawadamin-us-central1-0926 \\
        --image IMAGE -- python deepseek_v41/tpu-v7x/scripts/probe_v7x.py --out gs://...
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys

TIERS = {
    # tier: (hosts, chips per host, topology, machine family, cpu, memory, shm)
    "a": (1, 1, "1x1x1", "tpu7x-standard-1t", "40", "150Gi", "100Gi"),
    "b": (1, 4, "2x2x1", "tpu7x-standard-4t", "200", "800Gi", "600Gi"),
    "c": (4, 4, "2x2x4", "tpu7x-standard-4t", "200", "800Gi", "600Gi"),
}
PLACEMENT_POLICY = "wp2x2x4-jawadamin-us-central1-0926"
NAMESPACE = "dsv41"
SERVICE_ACCOUNT = "dsv41-runner"
COORDINATOR_PORT = 8476


def render(name: str, tier: str, pool: str | None, image: str, command: list[str], env: dict[str, str],
           max_restarts: int, code_tgz: str | None, ttl: int) -> dict:
    """Without ``pool`` the pods match every pool of the tier's shape, so the autoscaler
    tries spot and flex-start pools for one slice rather than provisioning two."""
    hosts, chips, topology, _, cpu, memory, shm = TIERS[tier]
    selector = {
        "cloud.google.com/gke-tpu-accelerator": "tpu7x",
        "cloud.google.com/gke-tpu-topology": topology,
    }
    if pool:
        selector["cloud.google.com/gke-nodepool"] = pool
    if hosts > 1:
        selector["cloud.google.com/placement-policy-name"] = PLACEMENT_POLICY
    envs = {
        "NUM_PROCESSES": str(hosts),
        "COORDINATOR_ADDRESS": f"{name}-w-0-0.{name}:{COORDINATOR_PORT}",
        "JAX_PLATFORMS": "tpu",
        "XLA_FLAGS": "--xla_allow_excess_precision=false",
        "JAX_COMPILATION_CACHE_DIR": "gs://dsv41-v7x-jawadamin-us-central1-0926/jax-cache",
        "RUN_NAME": name,
        "TIER": tier,
        **env,
    }
    if code_tgz:
        envs["CODE_TGZ"] = code_tgz
    container = {
        "name": "main",
        "image": image,
        "command": ["/bin/bash", "/app/deepseek_v41/tpu-v7x/scripts/entrypoint.sh"],
        "args": command,
        "env": [{"name": k, "value": v} for k, v in envs.items()]
        + [{"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}}],
        "ports": [{"containerPort": COORDINATOR_PORT}],
        "resources": {
            "requests": {"google.com/tpu": str(chips), "cpu": cpu, "memory": memory},
            "limits": {"google.com/tpu": str(chips)},
        },
        "volumeMounts": [{"name": "shm", "mountPath": "/dev/shm"}],
    }
    pod_spec = {
        "serviceAccountName": SERVICE_ACCOUNT,
        "restartPolicy": "Never",
        "nodeSelector": selector,
        "tolerations": [{"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}],
        "terminationGracePeriodSeconds": 25,
        "containers": [container],
        "volumes": [{"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": shm}}],
    }
    return {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "labels": {"owner": "jawadamin", "track": "wave-11-v7x", "tier": tier},
            "annotations": {"alpha.jobset.sigs.k8s.io/exclusive-topology": "cloud.google.com/gke-nodepool"},
        },
        "spec": {
            "ttlSecondsAfterFinished": ttl,
            "failurePolicy": {"maxRestarts": max_restarts},
            "replicatedJobs": [
                {
                    "name": "w",
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "parallelism": hosts,
                            "completions": hosts,
                            "completionMode": "Indexed",
                            "backoffLimit": 0,
                            "template": {"metadata": {"labels": {"track": "wave-11-v7x"}}, "spec": pod_spec},
                        }
                    },
                }
            ],
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True)
    p.add_argument("--tier", choices=sorted(TIERS), required=True)
    p.add_argument("--pool", help="pin to one node pool; default: any pool of the tier shape")
    p.add_argument("--image", required=True)
    p.add_argument("--code-tgz")
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE")
    p.add_argument("--max-restarts", type=int, default=3)
    p.add_argument("--ttl", type=int, default=3600)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    env = dict(kv.split("=", 1) for kv in a.env)
    doc = render(a.name, a.tier, a.pool, a.image, command, env, a.max_restarts, a.code_tgz, a.ttl)
    print(json.dumps(doc, indent=2))
    print("command: " + " ".join(shlex.quote(c) for c in command), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
