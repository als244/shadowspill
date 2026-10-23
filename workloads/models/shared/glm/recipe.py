"""Shared GLM LoRA recipe for generic quickstart and plan_step_search.

Architecture, inputs and initialization are supplied here. The benchmark
runner and planner need no GLM-specific behavior; the planner partitions the
model by its own default unless ``partition="glm_stages"`` asks for
`GLMStages`.
"""

from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

import mlops
import torch
import torch.distributed as dist

from .partition import GLMStages


def experiment(
    architecture,
    config_type,
    checkpoint_type,
    model_type,
    compiler_context=nullcontext,
    /,
    *,
    device,
    checkpoint=None,
    initialization="random",
    tiny=False,
    sequence_length=1024,
    sequences_per_step=8,
    sequences_per_microbatch=(1, 2, 4),
    lora_rank=32,
    lora_alpha=32.0,
    lora_dtype="bfloat16",
    router_weight_dtype=None,
    lora_head=False,
    grad_dtype="float32",
    master_dtype=None,
    opt_state_dtype="bfloat16",
    opt_state_rounding=None,
    gemm_precision="bf16",
    expert_storage=None,
    layer_limit=None,
    head_chunk_size=1024,
    partition=None,
    seed=0,
    transfer_shard_size=None,
    shard_gradients=False,
    preparation_timeout=1800,
):
    """Return the ordinary quickstart factory contract.

    Supply a local HF checkpoint for metadata, or select the small fixture
    with `tiny=True`. By default, checkpoint-shaped state is randomly initialized
    without reading weight payloads. Use `initialization="checkpoint"` to load
    actual weights. A compressed checkpoint's experts (FP8 or NVFP4) load as
    one-copy block-scaled FP8 (``expert_storage="fp8_block"``, an FP8
    checkpoint's bytes as they are) whose GEMMs run in FP8; ``"fp8"`` is the
    two-copy container and ``"bf16"`` decodes them. Its dense projections
    need BF16 GEMMs. The tiny fixture's experts are bf16, or with
    ``gemm_precision="fp8"`` block-scaled FP8 (``expert_storage`` picks
    another form). ``partition="glm_stages"`` plans with
    `GLMStages`; by default the planner chooses. This text-only recipe does
    not imply full-checkpoint numerical qualification.
    """
    if initialization not in ("random", "checkpoint"):
        raise ValueError("initialization must be 'random' or 'checkpoint'")
    if tiny and initialization == "checkpoint":
        raise ValueError("tiny=True has no checkpoint weights to load")
    if (checkpoint is not None) == bool(tiny):
        raise ValueError("Choose a checkpoint directory or tiny=True")
    distributed = dist.is_initialized()
    if transfer_shard_size is not None and not distributed:
        raise ValueError("transfer_shard_size requires distributed initialization")
    if shard_gradients and transfer_shard_size is None:
        raise ValueError("shard_gradients requires transfer_shard_size")
    world_size = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
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
    if router_weight_dtype not in (None, "bfloat16", "float32"):
        raise ValueError("router_weight_dtype must be None, bfloat16 or float32")
    if lora_dtype not in dtypes or grad_dtype not in dtypes:
        raise ValueError("LoRA/gradient dtypes must be float32, bfloat16 or float16")
    if master_dtype is not None and master_dtype not in dtypes:
        raise ValueError("master_dtype must be None, float32, bfloat16 or float16")
    if opt_state_dtype not in (*dtypes, "parameter"):
        raise ValueError(
            "the optimizer state dtype must be float32, bfloat16, float16 or parameter"
        )
    if opt_state_rounding is None:
        # As the other mlops recipes: moments stored at BF16 keep small
        # updates in expectation, anything wider rounds to nearest.
        opt_state_rounding = (
            "stochastic" if opt_state_dtype == "bfloat16" else "nearest"
        )
    if tiny and layer_limit is not None:
        raise ValueError(
            "layer_limit applies to checkpoint prefixes, not the tiny fixture"
        )
    if tiny and gemm_precision not in ("bf16", "fp8"):
        raise ValueError("The tiny fixture's experts compute in bf16 or fp8")
    if expert_storage not in (None, "bf16", "fp8", "fp8_block"):
        raise ValueError(
            f"expert_storage must be None, 'bf16', 'fp8' or 'fp8_block', "
            f"not {expert_storage!r}"
        )
    if tiny:
        expert_storage = expert_storage or (
            "fp8_block" if gemm_precision == "fp8" else "bf16"
        )
    if partition not in (None, "glm_stages"):
        raise ValueError(f"partition must be None or 'glm_stages', not {partition!r}")

    checkpoint = None if tiny else Path(checkpoint).expanduser().resolve()
    source = (
        None
        if tiny
        else checkpoint_type(
            checkpoint, gemm_precision=gemm_precision, expert_storage=expert_storage
        )
    )
    lora = dict(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_factor_dtype=dtypes[lora_dtype],
        lora_head=lora_head,
    )
    config = (
        config_type.tiny(
            hidden_size=128,
            expert_storage=expert_storage,
            router_weight_dtype=dtypes[router_weight_dtype or "bfloat16"],
            **lora,
        )
        if tiny
        else source.config_from_metadata(source.metadata)
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
            # Checkpoint-shaped models declare meta tensors, then fill spill.
            selected = torch.device(device)
            with torch.random.fork_rng(devices=[selected]):
                torch.manual_seed(seed)
                return model_type(config, device=selected).cpu()
        if source is None:
            source = checkpoint_type(
                checkpoint, gemm_precision=gemm_precision, expert_storage=expert_storage
            )
        return source.build_model(
            layer_limit=layer_limit,
            router_weight_dtype=dtypes.get(router_weight_dtype),
            **lora,
        )

    def initialize(model, adopted=frozenset()):
        from shadowspill.pytorch import initialize_storage_

        from .initialization import InitializationProgress, initialize_random

        progress = (
            InitializationProgress(model, mode=initialization, adopted=adopted)
            if rank == 0
            else None
        )
        try:
            if initialization == "random":
                initialize_random(
                    model,
                    seed=seed,
                    device=device,
                    adopted=adopted,
                    write_tensor=initialize_storage_,
                    progress=progress,
                )
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    source.load_into(
                        model,
                        adopted=adopted,
                        write_tensor=initialize_storage_,
                        on_tensor=progress,
                    )
        finally:
            close_source()

    @contextmanager
    def resources():
        with compiler_context(), distributed_resources():
            try:
                yield
            finally:
                close_source()

    total = sequence_length * sequences_per_step
    generator = torch.Generator().manual_seed(seed + 1_000_003 * rank)
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
        return model.loss(
            tokens,
            targets,
            boundaries,
            cumulative,
            chunks,
            chunk_size=head_chunk_size,
            reduction="sum",
        ) / (total * world_size)

    specification = None
    distributed_resources = nullcontext
    if distributed:
        from workloads.recipes.text.distributed import data_parallel_groups

        specification, distributed_resources = data_parallel_groups(
            device=torch.device(device),
            shard_size=transfer_shard_size,
            shard_gradients=shard_gradients,
            timeout=preparation_timeout,
        )

    result = {
        "model_factory": model_factory,
        "objective": objective,
        # mlops's AdamW, as the other mlops recipes use: one fused update per
        # parameter, reading each gradient at the dtype the step accumulates
        # it in, so nothing rounds it on the way.
        "optimizer": partial(
            mlops.optim.AdamW,
            lr=1e-4,
            gradient_dtype=dtypes[grad_dtype],
            opt_state_dtype=dtypes.get(opt_state_dtype, opt_state_dtype),
            opt_state_rounding=opt_state_rounding,
        ),
        "hyperparams": {"lr": 1e-4},
        "candidates": candidates,
        "plan_options": {
            "grad_dtype": dtypes[grad_dtype],
            "master_dtype": dtypes.get(master_dtype),
            **({"partition": GLMStages()} if partition == "glm_stages" else {}),
        },
        "units_per_step": total,
        "unit_label": "tokens",
        "context": resources,
        "metadata": {
            "architecture": architecture,
            "checkpoint": None if tiny else str(source.directory),
            "tiny": tiny,
            "initialization": initialization,
            "initialization_device": torch.device(device).type,
            "sequence_length": sequence_length,
            "sequences_per_step": sequences_per_step,
            "layer_limit": layer_limit,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dtype": lora_dtype,
            "grad_dtype": grad_dtype,
            "master_dtype": master_dtype,
            "optimizer": "mlops.optim.AdamW",
            "opt_state_dtype": opt_state_dtype,
            "opt_state_rounding": opt_state_rounding,
            "gemm_precision": gemm_precision,
            "expert_storage": expert_storage,
            "partition": "GLMStages()" if partition == "glm_stages" else "auto",
            "world_size": world_size,
            "rank": rank,
            "units_scope": "per_rank",
            "transfer_shard_size": transfer_shard_size,
            "shard_gradients": shard_gradients,
        },
    }
    if specification is not None:
        result["distributed"] = specification
    if not tiny:
        result["initialize"] = initialize
    return result
