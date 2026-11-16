"""Packed token views and sequence boundaries shared by supplied text models."""

from math import ceil

from shadowspill.training.definitions import (
    InputAdapter,
    InputView,
    RepresentativePattern,
    RepresentativeSample,
)
from shadowspill_client import BuildCandidates, PlanningOptions, Preparation
from shadowspill_client.inputs.shapes import BufferShape
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField

TOKENS = ("inputs", "tokens")
LENGTHS = ("inputs", "sequence_lengths")


def schema(config):
    return InputSchema(
        [
            TensorField(
                ("tokens",),
                "int64",
                ("position",),
                "pack",
                0,
                minimum=0,
                maximum=config.architecture["vocab_size"] - 1,
            )
        ],
        [
            Capacity("token_capacity", "tokens", "sum", ("tokens",), 0, scheduled=True),
            Capacity("max_sequence_length", "tokens", "max", ("tokens",), 0),
            Capacity("max_sequences", "sequences", "count"),
        ],
    )


def adapter(config, layout):
    (size,) = layout.buffer.fields[TOKENS]
    if any(layout.local_shape(p) != s for p, s in layout.buffer.fields.items()):
        raise ValueError("packed attention requires complete sequences per worker")
    capacity = config.architecture["max_seq_len"]
    slots = layout.buffer.max_datums + ceil(size / capacity)

    def fill(context, arrays):
        lengths = arrays[LENGTHS]
        end = 0
        for index, region in enumerate(context.routed.owner.regions):
            (span,) = region.fields[TOKENS]
            if span.start != end or span.stop - span.start > capacity:
                raise ValueError("packed input regions violate sequence geometry")
            lengths[index] = span.stop - span.start
            end = span.stop
        # Padding is its own causal sequence(s), never a continuation of a Datum.
        # Its slots are reserved in addition to the public sequence-count bound.
        index = len(context.routed.owner.regions)
        while end < size:
            amount = min(capacity, size - end)
            lengths[index] = amount
            index += 1
            end += amount

    return InputAdapter(
        views=[
            InputView(p, p, (1, *shape)) for p, shape in layout.buffer.fields.items()
        ],
        metadata=[TensorField(LENGTHS, "int32", (slots,), minimum=0, maximum=capacity)],
        fill=fill,
        padding_safe=True,
    )


def _check(context):
    placement = context.parallelism.inputs
    if placement is None:
        raise ValueError("packed preparation requires explicit input ownership")
    capacity = context.model.config["architecture"]["max_seq_len"]
    for batch in context.batches:
        if batch.limits.keys() != {
            "token_capacity",
            "max_sequence_length",
            "max_sequences",
        }:
            raise ValueError(
                "packed text requires token, sequence-length and sequence-count limits"
            )
        if any(type(value) is not int or value < 1 for value in batch.limits.values()):
            raise ValueError("packed text capacities must be positive integers")
        if batch.limits["max_sequence_length"] > capacity:
            raise ValueError("sequence-length limit exceeds model rotary capacity")
        if batch.limits["token_capacity"] >= 2**31:
            raise ValueError("packed offsets require fewer than 2**31 tokens")
        if batch.limits["max_sequences"] >= 2**31:
            raise ValueError("sequence count exceeds int32 metadata capacity")
        if batch.typical.keys() - {"sequence_length"}:
            raise ValueError("unknown packed text timing hint")
        typical = batch.typical.get("sequence_length", 1)
        if type(typical) is not int or not 1 <= typical <= min(
            batch.limits["max_sequence_length"], batch.limits["token_capacity"]
        ):
            raise ValueError("typical sequence length must satisfy the hard limits")
    return placement.partition_count


def defaults(context, *, preset):
    partitions = _check(context)
    sizes = set()
    capacities = []
    for batch in context.batches:
        cap = batch.limits["token_capacity"]
        full = max(1, cap // partitions)
        capacities.append(full)
        unit = min(batch.limits["max_sequence_length"], full)
        sizes.update(min(unit * multiple, full) for multiple in (1, 2, 4))
    # Compare full-capacity updates at each shape, plus a single microbatch for
    # small calls. Expansion rejects cross-products exceeding the global cap;
    # packing still checks the actual complete sequences against each layout.
    counts = {
        1,
        *(full // size for full in capacities for size in sizes if size <= full),
    }
    return Preparation(
        preset=preset,
        candidates=BuildCandidates(microbatch_sizes=sorted(sizes)),
        planning=PlanningOptions(microbatch_counts=sorted(counts)),
    )


def buffers(context, batch, paths, size):
    partitions = _check(context)
    if not 0 < size < 2**31:
        raise ValueError("packed microbatch size must fit int32 offsets")
    shape = BufferShape(
        context.batches[batch].limits["max_sequences"], {p: (size,) for p in paths}
    )
    return (shape,) * partitions


def representative(context, batch, physical):
    _check(context)
    settings = context.batches[batch]
    (size,) = physical.fields[TOKENS]
    length = min(
        size,
        settings.limits["max_sequence_length"],
        settings.typical.get("sequence_length", settings.limits["max_sequence_length"]),
    )
    count = min(
        physical.max_datums, size // length, settings.limits["token_capacity"] // length
    )
    values = {}
    for field in context.schema.fields:
        path = tuple(field.path)
        if path in physical.fields:
            values[path] = (
                field.default
                if field.default is not None
                else (True if field.dtype == "bool" else 0)
            )
    return RepresentativePattern(
        "packed_sequence_length_hint",
        [RepresentativeSample(values, {"position": length}, count=count)],
    )
