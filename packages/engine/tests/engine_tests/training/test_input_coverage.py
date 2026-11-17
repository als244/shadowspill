"""Universal envelope proofs have constructive, search-independent packings."""

import itertools
import json
from dataclasses import replace

import numpy as np
import pytest

from engine_tests.training.test_packing import batch, limits, schedule, schema
from shadowspill.parallelism import GroupRef, InputPlacement
from shadowspill.training.inputs.materialize import materialize_buffer
from shadowspill_client import BatchConfig, Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.inputs.coverage import CoverageCertificate, certify_coverage
from shadowspill_client.inputs.coverage_bounds import CoverageError
from shadowspill_client.inputs.packing import PackingError
from shadowspill_client.inputs.shapes import BufferShape, PackingSchedule
from shadowspill_client.inputs.validation import validate_batch
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField
from shadowspill_client.protocol.payload import snapshot_batch


def test_fragmented_128k_requires_requested_larger_shape_not_count_overflow():
    bound = limits(128 * 1024, 64 * 1024)
    small = schedule("two_64k", 64 * 1024, count=2)
    large = schedule("one_128k", 128 * 1024)
    placement = InputPlacement.independent(1)
    with pytest.raises(CoverageError, match="cannot certify advertised"):
        certify_coverage(schema(), bound, placement, [small])
    with pytest.raises(ValueError, match="including padding"):
        certify_coverage(
            schema(), bound, placement, [schedule("three_64k", 64 * 1024, 3)]
        )
    proof = certify_coverage(
        schema(), bound, placement, [small, large], grouping="grouped"
    )
    assert proof.schedule == large and proof.construction == "aggregate"
    for lengths in (
        [40 * 1024] * 3,
        [40 * 1024, 28 * 1024, 20 * 1024, 16 * 1024, 12 * 1024, 12 * 1024],
    ):
        with batch(lengths, bound) as data:
            packed = proof.select(data, [small, large], max_repair_nodes=0)
            assert packed.schedule == large
            assert sorted(i for row in packed.records for i in row) == list(
                range(len(lengths))
            )


@pytest.mark.parametrize("kind", ["aggregate", "slots"])
def test_exhaustive_small_domain_constructs_without_heuristic_or_repair(kind):
    bound = limits(
        8 if kind == "aggregate" else 6, 4 if kind == "aggregate" else 2, records=3
    )
    choice = schedule(
        kind,
        8 if kind == "aggregate" else 2,
        count=1 if kind == "aggregate" else 3,
        slots=3,
    )
    proof = certify_coverage(
        schema(),
        bound,
        InputPlacement.independent(1),
        [choice],
        grouping="grouped" if kind == "aggregate" else "independent",
    )
    assert proof.construction == kind
    cases = 0
    for count in range(4):
        for lengths in itertools.product(
            range(bound.limits["length"] + 1), repeat=count
        ):
            if sum(lengths) > bound.limits["positions"]:
                continue
            with batch(lengths, bound) as data:
                groups = (
                    [tuple(range(count))] if kind == "aggregate" and count else None
                )
                result = proof.pack(data, groups=groups)
                assert result.strategy == "coverage" and result.repair_nodes == 0
                assert sorted(i for row in result.records for i in row) == list(
                    range(count)
                )
                for index in range(len(choice.buffers)):
                    with materialize_buffer(
                        result, schema(), index, BufferBudget(2 << 20)
                    ) as physical:
                        for region in physical.regions:
                            for path, axes in region.fields.items():
                                value = data.fields[region.datum][path]
                                if value.source is not None:
                                    np.testing.assert_array_equal(
                                        physical.fields[path][axes],
                                        value.source.array(),
                                    )
            cases += 1
    assert cases >= 40


def test_guaranteed_construction_survives_bounded_search_failure(monkeypatch):
    from shadowspill_client.inputs import coverage

    bound = limits(6, 2, records=3)
    choice = schedule("slots", 2, count=3, slots=3)
    proof = certify_coverage(schema(), bound, InputPlacement.independent(1), [choice])

    def failed(*args, **kwargs):
        raise PackingError("bounded search exhausted")

    monkeypatch.setattr(coverage, "select_schedule", failed)
    with batch([1, 2, 1], bound) as data:
        result = proof.select(data, [choice], max_repair_nodes=0)
        assert result.strategy == "coverage" and result.records == ((0,), (1,), (2,))
        with pytest.raises(ValueError, match="coverage witness"):
            proof.select(data, [])
        with pytest.raises(ValueError, match="coupled groups"):
            proof.pack(data, groups=[(0, 1), (2,)])
    with pytest.raises(RuntimeError, match="closed"):
        proof.pack(data)


def test_multiresource_groups_use_aggregate_bounds_and_symbolic_alignment():
    definitions = InputSchema(
        [
            TensorField(("inputs", "tokens"), "int64", ("tokens",), "pack", 0),
            TensorField(
                ("inputs", "video"), "float32", ("frames", "height", 3), "pack", 0
            ),
            TensorField(("loss_inputs", "weights"), "float32", ("tokens",), "pack", 0),
        ],
        [
            Capacity(
                "tokens", "tokens", "sum", ("inputs", "tokens"), 0, scheduled=True
            ),
            Capacity("frames", "frames", "sum", ("inputs", "video"), 0, scheduled=True),
            Capacity("height", "pixels", "max", ("inputs", "video"), 1),
            Capacity("records", "records", "count"),
        ],
    )
    bound = BatchConfig(
        limits={"tokens": 9, "frames": 5, "height": 4, "records": 3},
        typical={"tokens": 1},
    )
    physical = BufferShape(
        3,
        {
            ("inputs", "tokens"): (9,),
            ("inputs", "video"): (5, 4, 3),
            ("loss_inputs", "weights"): (9,),
        },
    )
    proof = certify_coverage(
        definitions,
        bound,
        InputPlacement.independent(1),
        [PackingSchedule("media", [[physical]])],
        grouping="grouped",
    )
    datums = [
        Datum(
            {"tokens": np.arange(t), "video": np.ones((f, h, 3), dtype=np.float32)},
            {"weights": np.ones(t, dtype=np.float32)},
        )
        for t, f, h in ((4, 1, 2), (2, 3, 4), (3, 1, 1))
    ]
    with snapshot_batch(datums, BufferBudget(1 << 20)) as owned:
        data = validate_batch(owned, definitions, bound)
        result = proof.pack(data, groups=[(0, 2), (1,)])
        assert result.records == ((0, 2, 1),) and result.groups == (((0, 2), (1,)),)
    smaller = replace(
        physical, fields={**physical.fields, ("inputs", "video"): (5, 3, 3)}
    )
    with pytest.raises(CoverageError):
        certify_coverage(
            definitions,
            bound,
            InputPlacement.independent(1),
            [PackingSchedule("short_height", [[smaller]])],
        )


def test_logical_partitions_share_copies_but_count_every_padded_buffer():
    placement = InputPlacement(GroupRef("pairs", ((0, 2), (1, 3))))
    bound = limits(8, 2, records=4)
    choice = schedule("uniform", 2, count=2, partitions=2, slots=2)
    proof = certify_coverage(schema(), bound, placement, [choice])
    assert proof.construction == "slots" and proof.slots == (1, 1, 1, 1)
    assert choice.capacities(schema())["positions"] == 8
    with batch([1], bound) as data:
        result = proof.pack(data)
        assert result.records == ((0,), (), (), ())
    with pytest.raises(CoverageError, match="indivisible group"):
        certify_coverage(schema(), bound, placement, [choice], grouping="grouped")
    asymmetric = PackingSchedule(
        "asymmetric", [[choice.buffers[0]] * 3, [choice.buffers[0]]]
    )
    assert certify_coverage(schema(), bound, placement, [asymmetric]).slots == (1,) * 4


def test_unbounded_counts_dimensions_and_inactive_capacity_fields_cannot_be_certified():
    definition = schema()
    without_count = replace(definition, capacities=definition.capacities[:-1])
    with pytest.raises(CoverageError, match="count is unbounded"):
        certify_coverage(
            without_count,
            BatchConfig(limits={"positions": 8, "length": 4}),
            InputPlacement.independent(1),
            [schedule("unbounded", 8)],
        )
    unbounded_axis = InputSchema(
        [TensorField(("inputs", "x"), "float32", ("n",))],
        [Capacity("records", "records", "count")],
    )
    with pytest.raises(CoverageError, match="unbounded dimension"):
        certify_coverage(
            unbounded_axis,
            BatchConfig(limits={"records": 2}),
            InputPlacement.independent(1),
            [
                PackingSchedule(
                    "unbounded_axis", [[BufferShape(2, {("inputs", "x"): (2, 4)})]]
                )
            ],
        )
    inactive = InputSchema(
        [
            TensorField(("inputs", "x"), "float32", ("n",)),
            TensorField(("inputs", "unused"), "float32", ("n",)),
        ],
        [
            Capacity("records", "records", "count"),
            Capacity("length", "positions", "max", ("inputs", "unused"), 0),
        ],
    )
    with pytest.raises(CoverageError, match="unbounded dimension"):
        certify_coverage(
            inactive,
            BatchConfig(limits={"records": 2, "length": 4}),
            InputPlacement.independent(1),
            [
                PackingSchedule(
                    "missing_bound_field", [[BufferShape(2, {("inputs", "x"): (2, 4)})]]
                )
            ],
        )


@pytest.mark.parametrize("constant", [False, True])
def test_shared_values_need_independent_slots_unless_proven_constant(constant):
    definition = InputSchema(
        [
            TensorField(
                ("inputs", "scale"),
                "float32",
                batching="shared",
                minimum=1 if constant else None,
                maximum=1 if constant else None,
            )
        ],
        [Capacity("records", "records", "count")],
    )
    bound = BatchConfig(limits={"records": 3})
    physical = BufferShape(3, {("inputs", "scale"): ()})
    choice = PackingSchedule("shared", [[physical] * (1 if constant else 3)])
    proof = certify_coverage(definition, bound, InputPlacement.independent(1), [choice])
    assert proof.construction == ("aggregate" if constant else "slots")
    if not constant:
        with pytest.raises(CoverageError):
            certify_coverage(
                definition,
                bound,
                InputPlacement.independent(1),
                [PackingSchedule("insufficient_shared", [[physical]])],
            )


def test_fixed_positive_axis_can_bound_record_count_but_typical_cannot():
    definition = InputSchema(
        [TensorField(("inputs", "x"), "float32", (2,))],
        [Capacity("elements", "elements", "sum", ("inputs", "x"), 0, scheduled=True)],
    )
    bound = BatchConfig(limits={"elements": 6}, typical={"elements": 2})
    choice = PackingSchedule("stacked", [[BufferShape(3, {("inputs", "x"): (3, 2)})]])
    assert (
        certify_coverage(
            definition, bound, InputPlacement.independent(1), [choice]
        ).max_datums
        == 3
    )


def test_saved_certificate_recomputes_proof_and_rejects_different_validation_context():
    bound = limits(8, 4, records=3)
    proof = certify_coverage(
        schema(), bound, InputPlacement.independent(1), [schedule("whole", 8, slots=3)]
    )
    record = json.loads(json.dumps(proof.to_dict()))
    assert CoverageCertificate.from_dict(record) == proof
    with pytest.raises(ValueError, match="derived proof"):
        CoverageCertificate.from_dict({**record, "max_datums": 100})
    changed = json.loads(json.dumps(record))
    changed["schedule"]["partitions"][0][0]["max_datums"] = 2
    with pytest.raises(CoverageError):
        CoverageCertificate.from_dict(changed)
    with (
        batch([1], limits(16, 4, records=3)) as data,
        pytest.raises(ValueError, match="different input contract"),
    ):
        proof.pack(data)
