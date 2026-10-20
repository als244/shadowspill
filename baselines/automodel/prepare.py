"""Prepare a comparable full-training case from the pinned NVIDIA example."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from baselines.common import accumulation_steps, write_json

REFERENCES = Path(__file__).resolve().parents[1] / "references"


def configuration(
    *,
    microbatch_tokens=16384,
    sequence_length=4096,
    global_tokens=2097152,
    world_size=8,
    routing,
    recompute=True,
    cpu_offload=False,
    warmup_steps=3,
    steps=5,
):
    if world_size != 8:
        raise ValueError("this pinned EP8 recipe requires eight GPUs")
    accumulation = accumulation_steps(
        global_tokens, sequence_length, microbatch_tokens, world_size
    )
    if routing not in ("real", "balanced"):
        raise ValueError("routing must be real or balanced")
    if min(warmup_steps, steps) < 1:
        raise ValueError("warmup and measured steps must be positive")
    path = REFERENCES / "qwen3_moe_30b_te_deepep.upstream.yaml"
    provenance = json.loads((REFERENCES / "automodel.json").read_text())
    original = path.read_bytes()
    value = yaml.safe_load(original)
    value["model"]["dtype"] = "float32"
    value["model"]["backend"]["fake_balanced_gate"] = routing == "balanced"
    value["distributed"]["activation_checkpointing"] = recompute
    if cpu_offload:
        # FSDP2's own policy, as AutoModel's parallelizer hands it to every
        # fully_shard call: parameter shards, gradients and optimizer state on
        # the host. AutoModel offloads no activations.
        value["distributed"]["offload_policy"] = {
            "_target_": "torch.distributed.fsdp.CPUOffloadPolicy",
            "pin_memory": True,
        }
    value["distributed"]["mp_policy"] = {
        "_target_": "torch.distributed.fsdp.MixedPrecisionPolicy",
        "param_dtype": "bfloat16",
        "reduce_dtype": "float32",
        "output_dtype": None,
        "cast_forward_inputs": False,
    }
    value["optimizer"] = {
        "_target_": "torch.optim.AdamW",
        "lr": 3e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "fused": True,
    }
    scheduler = value["step_scheduler"]
    scheduler["global_batch_size"] = global_tokens // sequence_length
    scheduler["local_batch_size"] = microbatch_tokens // sequence_length
    scheduler["max_steps"] = warmup_steps + steps
    value["dataset"]["seq_len"] = sequence_length
    value["benchmark"]["warmup_steps"] = warmup_steps
    value["checkpoint"]["enabled"] = False
    value.pop("ci", None)
    metadata = {
        "upstream": provenance,
        "upstream_yaml_sha256": hashlib.sha256(original).hexdigest(),
        "world_size": world_size,
        "global_tokens": global_tokens,
        "sequence_length": sequence_length,
        "microbatch_tokens": microbatch_tokens,
        "gradient_accumulation": accumulation,
        "routing": routing,
        "recompute": recompute,
        "cpu_offload": cpu_offload,
        "warmup_steps": warmup_steps,
        "steps": steps,
        "precision": "BF16 compute; FP32 resident masters, gradients and AdamW states",
        "checkpoint_weights_loaded": False,
        "status": "prepared_not_gpu_validated",
        "changes_from_upstream": [
            "Explicit FP32 resident parameters and reduction, with BF16 compute",
            "AdamW with matched learning rate, betas, epsilon and weight decay",
            "Requested token counts, microbatch size and measured/warmup steps",
            f"Expert routing explicitly selected: {routing}",
            "Requested activation recomputation; checkpoint saving disabled",
            *(
                ["FSDP2 CPU offload of shards, gradients and optimizer state"]
                if cpu_offload
                else []
            ),
        ],
    }
    return value, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch-tokens", type=int, default=16384)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--global-tokens", type=int, default=2097152)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--routing", choices=("real", "balanced"), required=True)
    parser.add_argument(
        "--recompute", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cpu-offload", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args(argv)
    options = {key: value for key, value in vars(args).items() if key != "output"}
    value, metadata = configuration(**options)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.yaml").write_text(yaml.safe_dump(value, sort_keys=False))
    write_json(args.output / "request.json", metadata)
    print(args.output / "config.yaml")


if __name__ == "__main__":
    main()
