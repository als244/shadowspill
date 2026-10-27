"""Record case inputs, providers and the shared measurement contract."""

import hashlib
import importlib.metadata
import os
import platform
from dataclasses import asdict
from pathlib import Path


def versions():
    versions = {}
    for name in (
        "torch",
        "mlops",
        "sonic-moe",
        "quack-kernels",
        "moonep",
        "nvidia-cutlass-dsl",
        "transformer-engine",
        "megatron-core",
        "megatron-bridge",
        "deep-ep",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def fingerprint(value):
    import torch

    sample = value[:16, :16].detach().contiguous().cpu()
    return hashlib.sha256(sample.view(torch.uint16).numpy().tobytes()).hexdigest()


def describe(args, shape, implementation, x, dy, *, rank, world, group_reused, session):
    import torch

    return {
        **vars(args),
        "output": str(args.output),
        "rank": rank,
        "world_size": world,
        "process_id": os.getpid(),
        "process_group_reused": group_reused,
        "model_reused": False if session is None else session.reused,
        "model_instance": id(implementation.module),
        "shape": asdict(shape),
        "device": str(torch.cuda.get_device_properties(0)),
        "hostname": platform.node(),
        "versions": versions(),
        "flops_per_rank": shape.flops(args.tokens, args.training),
        "gradient_dtype": "float32",
        "quack_tuning": {
            "distributed": os.environ.get("MLOPS_QUACK_TUNE_DISTRIBUTED") == "1",
            "warmup_ms": float(os.environ.get("MLOPS_QUACK_TUNE_WARMUP_MS", "50")),
            "force_cache_update": bool(os.environ.get("QUACK_FORCE_CACHE_UPDATE")),
            "cache_dir": os.environ.get(
                "QUACK_CACHE_DIR", str(Path.home() / ".quack/cache")
            ),
            "timed_ms": float(os.environ.get("MLOPS_QUACK_TUNE_TIME_MS", "180")),
            "timing_windows": int(os.environ.get("MLOPS_QUACK_TUNE_SAMPLES", "3")),
        },
        "input_dtype": "bfloat16",
        "lora_compute_dtype": "bfloat16",
        "lora_alpha": shape.lora_alpha,
        "implementation": implementation.metadata,
        "routing": "real top-k then renormalized softmax",
        "input_seed": args.seed + 100 + rank,
        "gradient_seed": args.seed + 200 + rank,
        "input_fingerprint": fingerprint(x),
        "gradient_fingerprint": fingerprint(dy),
        "memory_measurement": "extra warmed untimed forward/backward pass",
        "rank_synchronization": (
            "barriers before and after each sample, outside timing"
        ),
        "auxiliary_loss": False,
        "activation_transport": "bf16",
        "optimizer": False,
        "dp_gradient_sync": False,
        "ep_router_sync": False,
        "timing": (
            "forward and backward; host wall plus GPU events; "
            "slowest rank per iteration"
        ),
        "performance_result": not args.tiny,
    }
