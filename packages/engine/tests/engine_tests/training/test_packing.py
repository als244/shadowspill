"""Finite-call packing obeys padded capacity and preserves every atomic record."""

import itertools
import json
import random
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest

from shadowspill.parallelism import (
    GroupRef,
    InputFieldPlacement,
    InputPlacement,
    OutputPlacement,
)
from shadowspill.training.inputs.materialize import materialize_buffer
from shadowspill.training.inputs.packing import (
    PackingError,
    pack_batch,
    select_schedule,
)
from shadowspill.training.inputs.routing import route_buffer
from shadowspill.training.inputs.shapes import BufferShape, PackingSchedule
from shadowspill.training.inputs.validation import combined_schema, validate_batch
from shadowspill_client import BatchConfig, Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField
from shadowspill_client.protocol.payload import snapshot_batch


def schema():
    return combined_schema(
        InputSchema(
            [
                TensorField(("x",), "float32", ("positions",), "pack", 0),
            ],
            [
                Capacity("positions", "positions", "sum", ("x",), 0, scheduled=True),
                Capacity("length", "positions", "max", ("x",), 0),
                Capacity("records", "records", "count"),
            ],
        ),
        InputSchema(
            [
                TensorField(("target",), "float32", ("positions",), "pack", 0),
                TensorField(
                    ("valid",),
                    "bool",
                    ("positions",),
                    "pack",
                    0,
                    required=False,
                    default=True,
                    padding=False,
                ),
                TensorField(
                    ("weight",),
                    "float32",
                    ("positions",),
                    "pack",
                    0,
                    required=False,
                    default=1.0,
                ),
            ]
        ),
    )


def shape(capacity, *, slots=256, inputs=None):
    return BufferShape(
        slots, {tuple(field.path): (capacity,) for field in (inputs or schema()).fields}
    )


def schedule(name, capacity, count=1, partitions=1, slots=256):
    return PackingSchedule(
        name, [[shape(capacity, slots=slots)] * count for _ in range(partitions)]
    )


def limits(total, longest=None, records=256):
    return BatchConfig(
        limits={
            "positions": total,
            "length": total if longest is None else longest,
            "records": records,
        }
    )


@contextmanager
def batch(lengths, bound):
    datums = [
        Datum(
            {"x": np.arange(n, dtype=np.float32) + 10 * i},
            {
                "target": np.ones(n, dtype=np.float32),
                "weight": np.full(n, (-1.0) ** i, dtype=np.float32),
            },
        )
        for i, n in enumerate(lengths)
    ]
    with snapshot_batch(datums, BufferBudget(32 << 20)) as owned:
        yield validate_batch(owned, schema(), bound)


def test_128k_cap_rejects_three_64k_buffers_and_uses_only_requested_larger_shape():
    k = 1024
    bound = limits(128 * k, 64 * k)
    placement = InputPlacement.independent(1)
    forbidden = schedule("three_64k", 64 * k, 3)
    with pytest.raises(ValueError, match="including padding"):
        forbidden.validate(schema(), bound, placement)
    two = schedule("two_64k", 64 * k, 2)
    large = schedule("one_128k", 128 * k)
    with batch([40 * k] * 3, bound) as data:
        with pytest.raises(PackingError, match="no prepared schedule"):
            select_schedule(data, schema(), bound, placement, [two])
        packed = select_schedule(data, schema(), bound, placement, [two, large])
        assert packed.schedule.name == "one_128k"
        assert packed.schedule.capacities(schema())["positions"] == 128 * k
        assert sorted(packed.records[0]) == [0, 1, 2]


def test_bounded_repair_finds_two_bins_when_best_fit_fragmentation_would_overflow():
    bound, placement = limits(128, 64), InputPlacement.independent(1)
    choice = schedule("two_64", 64, 2)
    with batch([38, 32, 19, 13, 13, 13], bound) as data:
        with pytest.raises(PackingError, match="repair nodes 0"):
            pack_batch(data, schema(), bound, placement, choice, max_repair_nodes=0)
        packed = pack_batch(data, schema(), bound, placement, choice)
        assert packed.strategy == "repair" and 0 < packed.repair_nodes < 10000
        assert sorted(
            sorted([38, 32, 19, 13, 13, 13][i] for i in records)
            for records in packed.records
        ) == [[13, 13, 38], [13, 19, 32]]


def test_many_short_and_empty_examples_have_no_hidden_count_or_loss_normalization():
    bound = limits(128, 64, records=256)
    with batch([1] * 128 + [0], bound) as data:
        packed = pack_batch(
            data, schema(), bound, InputPlacement.independent(1), schedule("two", 64, 2)
        )
        assert sorted(i for records in packed.records for i in records) == list(
            range(129)
        )
        actual = 0.0
        for index in range(2):
            with materialize_buffer(
                packed, schema(), index, BufferBudget(2 << 20)
            ) as buffer:
                values = buffer.fields
                actual += (
                    (values[("inputs", "x")] - values[("loss_inputs", "target")]) ** 2
                    * values[("loss_inputs", "weight")]
                    * values[("loss_inputs", "valid")]
                ).sum()
                for region in buffer.regions:
                    original = data.fields[region.datum][("inputs", "x")].source.array()
                    np.testing.assert_array_equal(
                        values[("inputs", "x")][region.fields[("inputs", "x")]],
                        original,
                    )
        expected = sum(((10 * i - 1) ** 2) * (-1.0) ** i for i in range(128))
        assert actual == expected


def test_repair_matches_an_independent_exhaustive_oracle_for_small_batches():
    rng = random.Random(113)
    bound, placement = limits(10, 5), InputPlacement.independent(1)
    choice = schedule("two_five", 5, 2, slots=6)
    for _ in range(100):
        lengths = [rng.randint(0, 5) for _ in range(rng.randint(1, 6))]
        if sum(lengths) > 10:
            continue
        fits = any(
            all(
                sum(
                    size
                    for size, owner in zip(lengths, owners, strict=True)
                    if owner == i
                )
                <= 5
                for i in range(2)
            )
            for owners in itertools.product(range(2), repeat=len(lengths))
        )
        with batch(lengths, bound) as data:
            if fits:
                packed = pack_batch(data, schema(), bound, placement, choice)
                assert sorted(i for records in packed.records for i in records) == list(
                    range(len(lengths))
                )
            else:
                with pytest.raises(PackingError):
                    pack_batch(data, schema(), bound, placement, choice)


def test_multiple_resources_and_atomic_groups_remain_together():
    definition = InputSchema(
        [
            TensorField(("inputs", "tokens"), "int64", ("tokens",), "pack", 0),
            TensorField(("inputs", "frames"), "float32", ("frames", 3), "pack", 0),
        ],
        [
            Capacity(
                "tokens", "tokens", "sum", ("inputs", "tokens"), 0, scheduled=True
            ),
            Capacity(
                "frames", "frames", "sum", ("inputs", "frames"), 0, scheduled=True
            ),
        ],
    )
    bound = BatchConfig(limits={"tokens": 12, "frames": 8})
    physical = BufferShape(
        2, {("inputs", "tokens"): (6,), ("inputs", "frames"): (4, 3)}
    )
    choice = PackingSchedule("two_resources", [[physical, physical]])
    records = [
        Datum({"tokens": np.arange(t), "frames": np.full((f, 3), i, dtype=np.float32)})
        for i, (t, f) in enumerate(((3, 1), (1, 2), (4, 1), (2, 1)))
    ]
    with snapshot_batch(records, BufferBudget(1 << 20)) as owned:
        data = validate_batch(owned, definition, bound)
        packed = pack_batch(
            data,
            definition,
            bound,
            InputPlacement.independent(1),
            choice,
            groups=[(0, 1), (2, 3)],
        )
        assert sorted(packed.records) == [(0, 1), (2, 3)]
        with pytest.raises(PackingError):
            pack_batch(
                data,
                definition,
                bound,
                InputPlacement.independent(1),
                choice,
                groups=[(0, 2), (1, 3)],
            )
        for index in range(2):
            with materialize_buffer(
                packed, definition, index, BufferBudget(2 << 20)
            ) as buffer:
                assert len(buffer.groups) == 1
                for region in buffer.regions:
                    for path, axes in region.fields.items():
                        np.testing.assert_array_equal(
                            buffer.fields[path][axes],
                            data.fields[region.datum][path].source.array(),
                        )


@pytest.mark.parametrize("sliced", [False, True])
def test_cooperating_worker_routing_counts_logical_inputs_once_and_keeps_alignment(
    sliced,
):
    definition = schema()
    placement = InputPlacement(
        GroupRef("pairs", ((2, 0), (3, 1))),
        fields=[
            InputFieldPlacement(
                tuple(f.path), "slice" if sliced else "replicate", 0 if sliced else None
            )
            for f in definition.fields
        ],
        loss="partitioned" if sliced else "leader",
        outputs={
            "prediction": OutputPlacement("concat", 0) if sliced else OutputPlacement()
        },
    )
    bound = limits(8, 4)
    choice = schedule("two_partitions", 4, partitions=2)
    with batch([3], bound) as data:
        packed = pack_batch(data, definition, bound, placement, choice)
        assert (
            choice.capacities(definition)["positions"] == 8
        )  # Empty partition counts.
        total, real_positions = 0.0, 0
        for index in range(2):
            with materialize_buffer(
                packed, definition, index, BufferBudget(2 << 20)
            ) as buffer:
                pieces = []
                for rank in placement.partitions.groups[index]:
                    routed = route_buffer(buffer, placement, rank)
                    values, mask = (
                        routed.fields[("inputs", "x")],
                        routed.fields[("loss_inputs", "valid")],
                    )
                    assert len(values) == len(mask)
                    if routed.contributes_loss:
                        total += values[mask].sum()
                        real_positions += mask.sum()
                    if placement.owns_output("prediction", rank):
                        pieces.append(values)
                np.testing.assert_array_equal(
                    np.concatenate(pieces), buffer.fields[("inputs", "x")]
                )
                wrong = placement.partitions.groups[1 - index][0]
                with pytest.raises(ValueError, match="does not own"):
                    route_buffer(buffer, placement, wrong)
        assert total == 3 and real_positions == 3


def test_stacked_padding_shared_metadata_and_scalar_outputs():
    definition = InputSchema(
        [
            TensorField(("inputs", "x"), "float32", ("n",)),
            TensorField(("inputs", "scale"), "float32", batching="shared"),
            TensorField(("loss_inputs", "target"), "int64"),
        ],
        [Capacity("records", "records", "count", scheduled=True)],
    )
    bound = BatchConfig(limits={"records": 4})
    physical = BufferShape(
        2,
        {
            ("inputs", "x"): (2, 3),
            ("inputs", "scale"): (),
            ("loss_inputs", "target"): (2,),
        },
    )
    choice = PackingSchedule("stacked", [[physical, physical]])
    records = [
        Datum({"x": [1.0, 2.0], "scale": scale}, {"target": i})
        for i, scale in enumerate((1.0, 2.0, 1.0))
    ]
    with snapshot_batch(records, BufferBudget(1 << 20)) as owned:
        data = validate_batch(owned, definition, bound)
        packed = pack_batch(
            data, definition, bound, InputPlacement.independent(1), choice
        )
        for index in range(2):
            with materialize_buffer(
                packed, definition, index, BufferBudget(2 << 20)
            ) as buffer:
                for region in buffer.regions:
                    assert (
                        buffer.fields[("inputs", "scale")].item()
                        == records[region.datum].inputs["scale"]
                    )
                    assert (
                        buffer.fields[("loss_inputs", "target")][region.slot]
                        == region.datum
                    )
                    np.testing.assert_array_equal(
                        buffer.fields[("inputs", "x")][region.slot], [1.0, 2.0, 0.0]
                    )


def test_schedule_records_roundtrip_and_materialization_reserves_before_allocation(
    monkeypatch,
):
    choice = schedule("roundtrip", 8)
    assert PackingSchedule.from_dict(json.loads(json.dumps(choice.to_dict()))) == choice
    with batch([3], limits(8)) as data:
        packed = pack_batch(
            data, schema(), limits(8), InputPlacement.independent(1), choice
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("allocation preceded staging admission")

        monkeypatch.setattr(np, "empty", forbidden)
        with pytest.raises(ValueError, match="staging bytes"):
            materialize_buffer(packed, schema(), 0, BufferBudget(10))
    with pytest.raises(ValueError, match="physical fixed dimension"):
        BufferShape(1, {("inputs", "x"): (1, 4)}).validate(
            {("inputs", "x"): TensorField(("inputs", "x"), "float32", (3,))}
        )
    with pytest.raises(ValueError, match="partition"):
        replace(choice, partitions=[choice.partitions[0]] * 2).validate(
            schema(), limits(16), InputPlacement.independent(1)
        )
