"""Bounded default preparation for independent fixed stacked tensor models."""

from collections.abc import Sequence
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from math import ceil, floor
from pathlib import Path

from shadowspill.training.inputs.coverage_bounds import input_bounds
from shadowspill.training.inputs.shapes import BufferShape
from shadowspill.training.inputs.validation import Path as FieldPath
from shadowspill_client import BuildCandidates, PlanningOptions, Preparation
from shadowspill_client.protocol.definitions import DefinitionSource

from .preparation import (
    PreparationContext,
    PreparationDefinition,
    ResolvedPreparationDefinition,
)
from .samples import RepresentativePattern, RepresentativeSample

STACKED_PRESET = "shadowspill/stacked@1"


def _check(context: PreparationContext) -> int:
    if any(batch.typical for batch in context.batches):
        raise ValueError("the generic stacked preset does not declare timing hints")
    if context.model.definition.batching != "independent" or any(
        field.batching != "stack" or any(not isinstance(n, int) for n in field.shape)
        for field in context.schema.fields
    ):
        raise ValueError(
            "variable, shared or coupled inputs require an installed preparation preset"
        )
    if context.parallelism.inputs is None:
        raise ValueError("stacked preparation requires declared input placement")
    return context.parallelism.inputs.partition_count


def defaults(context: PreparationContext) -> Preparation:
    partitions = _check(context)
    fields = {tuple(field.path): field for field in context.schema.fields}
    sizes = {1}
    for batch in context.batches:
        records = input_bounds(context.schema, batch, fields).records
        size = max(1, (records + partitions - 1) // partitions)
        sizes.update((max(1, size // 2), size))
    # A small, inspectable family. Coverage is separately proved for every batch
    # option; this preset never lowers limits or adds unrequested larger shapes.
    return Preparation(
        candidates=BuildCandidates(microbatch_sizes=sorted(sizes)),
        planning=PlanningOptions(microbatch_counts=[1, 2]),
        preset=STACKED_PRESET,
    )


def buffers(
    context: PreparationContext, batch: int, paths: Sequence[FieldPath], size: int
) -> tuple[BufferShape, ...]:
    partitions = _check(context)
    fields = {tuple(field.path): field for field in context.schema.fields}
    shape = BufferShape(
        size, {path: (size, *(int(n) for n in fields[path].shape)) for path in paths}
    )
    return (shape,) * partitions


def representative(
    context: PreparationContext, batch: int, physical: BufferShape
) -> RepresentativePattern:
    _check(context)
    values = {}
    for field in context.schema.fields:
        path = tuple(field.path)
        if path not in physical.fields:
            continue
        if field.default is not None:
            value = field.default
        else:
            lower = field.minimum
            upper = field.maximum
            if "int" in field.dtype or field.dtype == "bool":
                lower = None if lower is None else ceil(lower)
                upper = None if upper is None else floor(upper)
            value = 1
            if lower is not None:
                value = max(value, lower)
            if upper is not None:
                value = min(value, upper)
            if field.dtype == "bool":
                value = bool(value)
        assert isinstance(value, (bool, int, float))
        values[path] = value
    return RepresentativePattern(
        "fixed_stacked_schema_fills",
        [RepresentativeSample(values, count=physical.max_datums)],
    )


@lru_cache(maxsize=1)
def builtin_preparation() -> ResolvedPreparationDefinition:
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for name in (
        "definitions/stacked.py",
        "definitions/samples.py",
        "inputs/coverage_bounds.py",
        "inputs/shapes.py",
    ):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return ResolvedPreparationDefinition(
        PreparationDefinition(STACKED_PRESET, defaults, buffers, representative),
        DefinitionSource("shadowspill", version("shadowspill"), digest.hexdigest()),
    )
