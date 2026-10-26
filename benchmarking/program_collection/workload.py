"""Build catalog structure and packed input geometry without model storage."""

from dataclasses import replace
from functools import partial

import torch

from shadowspill import Program, build_step
from workloads.full_model import build_model, full_model_objective, spec
from workloads.models.shared.training import LEARNING_RATE

from .matrix import ProgramRequest


def build_program(request: ProgramRequest) -> Program:
    base = spec(request.model.model, request.model.implementation)
    manifest = replace(
        base,
        sequence_length=request.sequence_length,
        sequences_per_microbatch=request.sequences_per_microbatch,
        accumulation_count=request.accumulation_rounds,
        head_scratch_bytes=base.head_scratch_bytes
        if request.model.head_scratch_bytes is None
        else request.model.head_scratch_bytes,
    )
    generator = torch.Generator().manual_seed(request.seed)
    args = (
        *(
            torch.randint(
                manifest.model_config.vocab_size,
                (1, request.tokens_per_microbatch),
                generator=generator,
            )
            for _ in range(2)
        ),
        (request.sequence_length,) * request.sequences_per_microbatch,
    )
    model = build_model(manifest)
    optimizer = manifest.dtypes.optimizer()(
        model.parameters(), lr=torch.tensor(LEARNING_RATE)
    )
    return build_step(
        model,
        objective=partial(full_model_objective, manifest),
        optimizer=optimizer,
        example_args=args,
        num_examples=request.accumulation_rounds,
        target=request.profiling.device,
        **manifest.dtypes.plan_arguments(),
    )
