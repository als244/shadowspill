"""Quack expert-parallel text factory for the ordinary generic quickstart.

One invocation uses one microbatch geometry: MoonEP's communication buffer has
an exact token shape. Search the geometries independently, then select each
budget's winner across their reports before executing any full-size benchmark.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist

from shadowspill.pytorch import Distributed
from workloads.common.training import LEARNING_RATE
from workloads.full_model import throughput_spec

from .quickstart import Precision, _distributed_objective, text_metrics

_MODELS = {"olmoe": "OLMoE", "qwen3moe": "Qwen3MoE", "qwen35moe": "Qwen35MoE"}


def experiment(
    *,
    device,
    model="qwen3moe",
    sequence_length=4096,
    sequences_per_step=64,
    sequences_per_microbatch=16,
    model_dtype="bfloat16",
    master_dtype="float32",
    grad_dtype="float32",
    opt_state_dtype="float32",
    seed=0,
    preparation_timeout=3600,
    symmetric_planning=True,
    config_overrides=None,
):
    """Build an EP recipe after Runtime installation, with caller-owned groups.

    Every rank contributes distinct data. Expert weights are unique local
    parameters; nonexpert parameters are replicas across the EP group. The
    objective divides each local token-loss sum by the global token count.
    Computation weights/gradients are not transfer-sharded. Optimizer ownership
    follows each parameter's replicas, independently of expert placement.
    """
    if model not in _MODELS:
        raise ValueError(f"model must be one of {tuple(_MODELS)}")
    for name, value in (
        ("sequence_length", sequence_length),
        ("sequences_per_step", sequences_per_step),
        ("sequences_per_microbatch", sequences_per_microbatch),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if sequences_per_step % sequences_per_microbatch:
        raise ValueError("sequences_per_microbatch must divide sequences_per_step")
    if model_dtype != "bfloat16":
        raise ValueError("Quack EP requires BF16 activations and compute weights")
    if grad_dtype not in {"float32", "bfloat16"}:
        raise ValueError("Quack EP gradients must be float32 or bfloat16")
    if not dist.is_initialized() or dist.get_world_size() < 2:
        raise ValueError("launch with --distributed and at least two ranks")
    if preparation_timeout <= 0:
        raise ValueError("preparation_timeout must be positive")
    rank, world = dist.get_rank(), dist.get_world_size()
    precision = Precision(
        model_dtype=model_dtype,
        master_dtype=master_dtype,
        grad_dtype=grad_dtype,
        opt_state_dtype=opt_state_dtype,
    )
    manifest = throughput_spec(model, "mlops")
    config = replace(
        manifest.model_config,
        **{**(config_overrides or {}), "max_seq_len": sequence_length},
    )
    if config.n_experts % world:
        raise ValueError("the number of experts must divide evenly across EP ranks")
    manifest = replace(
        manifest,
        model_config=config,
        sequence_length=sequence_length,
        sequences_per_microbatch=sequences_per_step,
        accumulation_count=1,
        model_dtype=model_dtype,
        master_dtype=master_dtype or "none",
        grad_dtype=grad_dtype,
        opt_state_dtype=opt_state_dtype,
    )
    optimizer = precision.optimizer(manifest.dtypes.optimizer())
    precision.apply()
    generator = torch.Generator().manual_seed(seed * 1_000_003 + rank)
    whole = (1, sequence_length * sequences_per_step)
    tokens = torch.randint(config.vocab_size, whole, generator=generator)
    targets = torch.randint(config.vocab_size, whole, generator=generator)
    capacity = sequence_length * sequences_per_microbatch
    examples = tuple(
        (
            tokens[:, start : start + capacity].clone(),
            targets[:, start : start + capacity].clone(),
            (sequence_length,) * sequences_per_microbatch,
        )
        for start in range(0, whole[1], capacity)
    )
    group = dist.new_group(
        backend="nccl",
        device_id=torch.device(device),
        timeout=timedelta(seconds=preparation_timeout),
    )

    @contextmanager
    def resources():
        try:
            yield
        finally:
            dist.destroy_process_group(group)

    def model_factory():
        from workloads import mlops as models

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + rank)
            return getattr(models, _MODELS[model])(
                config,
                ep_group=group,
                token_capacity=capacity,
                device=device,
                parameter_device="cpu",
                dtype=torch.bfloat16,
                weight_grad_dtype=precision.gradients,
                expert_backend="quack",
            )

    def ownership(module):
        return Distributed(
            group,
            replica_overrides=[(module.expert_parameters(), None)],
            groups={"experts": group},
            timeout=preparation_timeout,
            symmetric_planning=symmetric_planning,
        )

    return {
        "model_factory": model_factory,
        "cleanup_model": lambda module: module.close(),
        "objective": partial(_distributed_objective, manifest, world),
        "optimizer": optimizer,
        "hyperparams": {"lr": LEARNING_RATE},
        "candidates": {str(sequences_per_microbatch): examples},
        "metric_reducer": text_metrics,
        "plan_options": {**precision.plan_arguments(), "shard_optimizer": True},
        "distributed": ownership,
        "context": resources,
        "units_per_step": sequence_length * sequences_per_step,
        "unit_label": "tokens",
        "metadata": {
            "model": model,
            "model_config": asdict(config),
            "world_size": world,
            "rank": rank,
            "units_scope": "per_rank",
            "sequence_length": sequence_length,
            "sequences_per_step": sequences_per_step,
            "sequences_per_microbatch": sequences_per_microbatch,
            "global_tokens_per_step": world * sequence_length * sequences_per_step,
            "expert_parallel_size": world,
            "expert_backend": "quack",
            "transfer_shard_size": None,
            "shard_gradients": False,
            "trainable": {"mode": "full"},
            "dtypes": {name: value for name, value, _ in precision.lines()},
        },
    }
