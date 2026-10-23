"""Checkpoint-shaped GLM adapter for the model-independent stage tracer."""

from __future__ import annotations

import hashlib

import torch

from benchmarking.stage_traces.api import TraceWorkload
from workloads.models.shared.glm.checkpoint import physical_state
from workloads.models.shared.glm.initialization import _roles

from .checkpoint import GLMCheckpoint
from .config import Config
from .implementations.mlops import Model


def workload(
    *,
    sequence_length,
    batch_size,
    checkpoint=None,
    tiny=False,
    tiny_expert_storage="bf16",
    lora_rank=32,
    lora_alpha=32.0,
    seed=0,
    layer_limit=None,
    head_chunk_size=1024,
):
    if tiny:
        config = Config.tiny(
            hidden_size=128,
            expert_storage=tiny_expert_storage,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_factor_dtype=torch.bfloat16,
        )
        model = Model(config, device="meta")
    else:
        if checkpoint is None:
            raise ValueError("checkpoint metadata directory is required")
        with GLMCheckpoint(checkpoint, expert_storage="fp8_block") as source:
            model = source.build_model(
                layer_limit=layer_limit,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_factor_dtype=torch.bfloat16,
            )
        config = model.config
    state = physical_state(model)
    roles = _roles(model, state)
    initializers = {}
    for name in state:
        tensor_seed = (
            int.from_bytes(
                hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little"
            )
            % 2**63
        )
        initializers[name] = {
            "callable": "workloads.models.shared.glm.initialization:fill_random_tensor",
            "kwargs": {
                "role": roles[name],
                "tensor_seed": tensor_seed,
                "std": config.initializer_range,
            },
        }
    count = sequence_length * batch_size
    generator = torch.Generator().manual_seed(seed + 100)
    tokens = torch.randint(config.vocab_size, (count,), generator=generator)
    targets = torch.randint(config.vocab_size, (count,), generator=generator)
    boundaries = tuple(range(0, count + 1, sequence_length))
    cumulative = torch.tensor(boundaries, dtype=torch.int64)
    chunks = torch.tensor(
        [
            (b, c)
            for b in range(batch_size)
            for c in range((sequence_length + 63) // 64)
        ],
        dtype=torch.int64,
    )

    def objective(module, tokens, targets, boundaries, cumulative, chunks):
        return (
            module.loss(
                tokens,
                targets,
                boundaries,
                cumulative,
                chunks,
                chunk_size=head_chunk_size,
                reduction="sum",
            )
            / count
        )

    return TraceWorkload(
        model,
        objective,
        (tokens, targets, boundaries, cumulative, chunks),
        initializers,
        metadata={
            "model": "glm53-flash",
            "initialization": "random",
            "checkpoint_payloads_read": False,
            "checkpoint": checkpoint,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "gradient_dtype": "float32",
            "expert_storage": config.expert_storage,
            "layer_limit": layer_limit,
            "tiny": tiny,
        },
    )
