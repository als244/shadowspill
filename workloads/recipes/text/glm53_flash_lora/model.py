"""Checkpoint metadata and tensor loading; no preparation or execution policy."""

import json
from contextlib import contextmanager
from pathlib import Path

import torch

from workloads.models.glm53_flash.checkpoint import GLMCheckpoint
from workloads.recipes.text.data.tokenizer import tokenizer_identity
from workloads.recipes.text.glm.initialize import Initialize


@contextmanager
def checkpoint(options):
    # None preserves the checkpoint's expert representation (block FP8 for the
    # published model), so even small BF16 fixtures need no hidden conversion.
    with GLMCheckpoint(options["checkpoint"]) as source:
        model = source.build_model(
            layer_limit=options["layer_limit"],
            lora_rank=options["lora_rank"],
            lora_alpha=options["lora_alpha"],
            lora_factor_dtype=torch.bfloat16,
            router_weight_dtype={"bf16": torch.bfloat16, "fp32": torch.float32}.get(
                options["router_weight_dtype"]
            ),
        )
        initialize = Initialize(
            model, source=source, initialization="checkpoint", seed=options["seed"]
        )
        yield model, source, initialize


def describe(options, model, source):
    dataset = json.loads((Path(options["data"]) / "meta.json").read_text())
    identity = tokenizer_identity(options["checkpoint"])
    if dataset.get("tokenizer_sha256") != identity:
        raise ValueError("corpus tokenizer identity differs from this checkpoint")
    if dataset["vocab_size"] > model.config.vocab_size:
        raise ValueError("corpus vocabulary exceeds model embeddings")
    base = {
        "architecture": "glm53_flash",
        "layers": model.config.num_hidden_layers,
        "storage": source.inventory["storage_dtype_bytes"],
        "expert_storage": model.config.expert_storage,
        "dense_gemm": "bf16",
        "lora_rank": options["lora_rank"],
        "lora_alpha": options["lora_alpha"],
    }
    return {
        "frozen_state_id": options["frozen_state_id"]
        + ":"
        + json.dumps(base, sort_keys=True),
        "inventory": source.inventory,
        "tokenizer_sha256": identity,
        "dataset": dataset,
    }
