"""GLM LoRA recipe for generic quickstart and plan_step_search.

Architecture, inputs, initialization and partitioning are supplied here. The
benchmark runner and planner need no GLM-specific behavior.
"""

from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch

from .checkpoint import GLMCheckpoint
from .config import Config
from .hf_config import from_huggingface
from .model import LanguageModel
from .partition import GLMStages


def experiment(
    *,
    device,
    checkpoint=None,
    tiny=False,
    sequence_length=1024,
    sequences_per_step=8,
    sequences_per_microbatch=(1, 2, 4),
    lora_rank=32,
    lora_alpha=32.0,
    lora_dtype="float32",
    lora_head=False,
    grad_dtype="float32",
    gemm_precision="bf16",
    layer_limit=None,
    head_chunk_size=1024,
    seed=0,
):
    """Return the ordinary quickstart factory contract.

    Supply a local HF checkpoint, or explicitly select the small random fixture
    with `tiny=True`. Checkpoint import retains compressed weight storage and
    currently requires BF16 GEMMs. The tiny fixture can also exercise FP8 GEMMs.
    This text-only recipe does not imply full-checkpoint numerical qualification.
    """
    if (checkpoint is not None) == bool(tiny):
        raise ValueError("Choose a checkpoint directory or tiny=True")
    sizes = tuple(sequences_per_microbatch)
    if sequence_length <= 0 or sequences_per_step <= 0 or not sizes:
        raise ValueError(
            "Sequence length, step size and candidate list must be positive"
        )
    if any(size <= 0 or sequences_per_step % size for size in sizes):
        raise ValueError("Each sequences_per_microbatch must divide sequences_per_step")
    if len(set(sizes)) != len(sizes):
        raise ValueError("Duplicate microbatch candidates")
    if lora_rank <= 0 or head_chunk_size <= 0:
        raise ValueError(
            "This training recipe requires positive LoRA rank and head chunk"
        )
    dtypes = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if lora_dtype not in dtypes or grad_dtype not in dtypes:
        raise ValueError("LoRA/gradient dtypes must be float32, bfloat16 or float16")
    if tiny and layer_limit is not None:
        raise ValueError(
            "layer_limit applies to checkpoint prefixes, not the tiny fixture"
        )

    checkpoint = None if tiny else Path(checkpoint).expanduser().resolve()
    source = None if tiny else GLMCheckpoint(checkpoint, gemm_precision=gemm_precision)
    lora = dict(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_factor_dtype=dtypes[lora_dtype],
        lora_head=lora_head,
    )
    config = (
        Config.tiny(
            hidden_size=128,
            expert_storage=gemm_precision,
            gemm_precision=gemm_precision,
            **lora,
        )
        if tiny
        else from_huggingface(source.metadata)
    )

    def close_source():
        nonlocal source
        if source is not None:
            source.close()
            source = None

    def model_factory():
        nonlocal source
        if tiny:
            # Only the explicitly small fixture constructs initialized state.
            # Real checkpoints instead declare meta tensors and stream into spill.
            selected = torch.device(device)
            with torch.random.fork_rng(devices=[selected]):
                torch.manual_seed(seed)
                return LanguageModel(config, device=selected).cpu()
        if source is None:
            source = GLMCheckpoint(checkpoint, gemm_precision=gemm_precision)
        return source.build_model(layer_limit=layer_limit, **lora)

    def initialize(model):
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                source.load_into(model)
        finally:
            close_source()

    @contextmanager
    def resources():
        try:
            yield
        finally:
            close_source()

    total = sequence_length * sequences_per_step
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(config.vocab_size, (total,), generator=generator)
    targets = torch.randint(config.vocab_size, (total,), generator=generator)
    candidates = {}
    for size in sizes:
        count = size * sequence_length
        boundaries = tuple(range(0, count + 1, sequence_length))
        cumulative = torch.tensor(boundaries, dtype=torch.int64)
        chunks = torch.tensor(
            [
                (sequence, chunk)
                for sequence in range(size)
                for chunk in range((sequence_length + 63) // 64)
            ],
            dtype=torch.int64,
        )
        candidates[str(size)] = tuple(
            (
                tokens[start : start + count].clone(),
                targets[start : start + count].clone(),
                boundaries,
                cumulative.clone(),
                chunks.clone(),
            )
            for start in range(0, total, count)
        )

    def objective(model, tokens, targets, boundaries, cumulative, chunks):
        return (
            model.loss(
                tokens,
                targets,
                boundaries,
                cumulative,
                chunks,
                chunk_size=head_chunk_size,
                reduction="sum",
            )
            / total
        )

    result = {
        "model_factory": model_factory,
        "objective": objective,
        "optimizer": partial(torch.optim.AdamW, lr=1e-4, foreach=False),
        "hyperparams": {"lr": 1e-4},
        "candidates": candidates,
        "plan_options": {"partition": GLMStages(), "grad_dtype": dtypes[grad_dtype]},
        "units_per_step": total,
        "unit_label": "tokens",
        "context": resources,
        "metadata": {
            "architecture": "glm53_flash",
            "checkpoint": None if tiny else str(source.directory),
            "tiny": tiny,
            "sequence_length": sequence_length,
            "sequences_per_step": sequences_per_step,
            "layer_limit": layer_limit,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dtype": lora_dtype,
            "grad_dtype": grad_dtype,
            "gemm_precision": gemm_precision,
            "partition": "GLMStages()",
        },
    }
    if not tiny:
        result["initialize"] = initialize
    return result
