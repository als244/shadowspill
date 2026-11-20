"""Full candidate layouts retain every requested operation and capacity option."""

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from engine_tests.fixtures.component_operations import components
from engine_tests.training.test_setup_resolution import arguments, installed
from shadowspill.parallelism import (
    GroupRef,
    InputFieldPlacement,
    InputPlacement,
    OutputPlacement,
    Parallelism,
)
from shadowspill.training.inputs.adapters import InputAdapter
from shadowspill.training.preparation.configuration.layouts import (
    LayoutCoverageError,
    prepare_layouts,
)
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_client import BatchConfig, Execution
from shadowspill_client.inputs.coverage_bounds import CoverageError
from shadowspill_client.protocol.components import OutputSet, TensorOutput
from shadowspill_client.protocol.inputs import TensorField


def test_complete_layout_family_uses_no_models_or_arrays_and_keeps_all_capabilities(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        pytest.fail("constructed tensors or models during layout admission")

    model, _, optimizer = components()
    registry = installed(
        model=replace(model.component, create=forbidden),
        optimizer=replace(optimizer.component, create=forbidden),
    )
    setup = resolve_training(
        registry,
        **{
            **arguments(),
            "batch": [
                BatchConfig(limits={"max_examples": 3}),
                BatchConfig(limits={"max_examples": 4}),
            ],
            "operations": {
                "step": ["metrics"],
                "forward_backward": ["metrics"],
                "forward": ["predictions"],
            },
        },
    )
    monkeypatch.setattr(np, "zeros", forbidden)
    monkeypatch.setattr(np, "empty", forbidden)
    result = prepare_layouts(setup)
    assert tuple(result.families) == setup.config.operations[:-1]
    assert len(result.coverage) == 6
    for key, candidates in result.families.items():
        assert {c.candidate.batch for c in candidates} == {0, 1}
        for batch_index in (0, 1):
            witness = result.coverage[key, batch_index]
            assert witness.schedule in [c.candidate.schedule for c in candidates]
        for candidate in candidates:
            assert (
                candidate.staging.staging_bytes
                <= setup.config.execution.max_staging_bytes
            )
            encoded = json.dumps(candidate.signature(), sort_keys=True)
            assert '"inputs"' in encoded
            assert len(candidate.bindings) == candidate.candidate.count
            for (buffer, rank), binding in candidate.bindings.items():
                assert rank == 0 and binding.buffer == buffer
                assert binding.outputs is candidate.outputs
                if key.outputs == "predictions":
                    assert binding.inputs.shapes.keys() == {("inputs", "pixels")}
                    assert not binding.contract.controls.fields
                else:
                    assert ("loss_inputs", "valid") in binding.inputs.shapes
    with pytest.raises(TypeError):
        result.families[setup.config.operations[0]] = ()


def metadata_adapter(expensive):
    def adapter(config, layout):
        metadata = (
            (
                TensorField(
                    ("inputs", "scratch"),
                    "int64",
                    (1 << 19,),
                    required=False,
                    default=0,
                ),
            )
            if expensive(layout.buffer.max_datums)
            else ()
        )
        return InputAdapter(metadata=metadata, padding_safe=True)

    return adapter


def test_host_infeasible_alternatives_have_reasons_and_coverage_is_recomputed():
    model, _, _ = components()
    registry = installed(
        model=replace(
            model.component, input_adapter=metadata_adapter(lambda size: size == 2)
        )
    )
    setup = resolve_training(
        registry, **arguments(), execution=Execution(max_staging_bytes=2 << 20)
    )
    result = prepare_layouts(setup)
    assert [row.stage for row in result.rejected].count("capacity") == 1
    assert [row.stage for row in result.rejected].count("staging") == 2
    assert all(
        "input staging needs" in row.reason
        for row in result.rejected
        if row.stage == "staging"
    )
    key = setup.config.operations[0]
    assert len(result.families[key]) == 3
    assert result.coverage[key, 0].schedule.buffers[0].max_datums == 4
    # Identical physical adapters are resolved once and reused across counts.
    one, two = result.families[key][:2]
    assert one.bindings[0, 0].inputs is two.bindings[0, 0].inputs
    assert two.bindings[0, 0].inputs is two.bindings[1, 0].inputs


def test_uncovered_batch_keeps_actual_candidates_and_empty_family_still_fails():
    model, _, _ = components()
    registry = installed(
        model=replace(
            model.component, input_adapter=metadata_adapter(lambda size: size > 1)
        )
    )
    setup = resolve_training(
        registry,
        **{
            **arguments(),
            "batch": [BatchConfig(limits={"max_examples": n}) for n in (1, 4)],
            "execution": Execution(max_staging_bytes=2 << 20),
        },
    )
    layouts = prepare_layouts(setup)
    key = setup.config.operations[0]
    assert (key, 1) not in layouts.coverage
    assert {row.candidate.batch for row in layouts.families[key]} == {0, 1}
    assert layouts.setup.config.batches == setup.config.batches
    registry = installed(
        model=replace(
            model.component, input_adapter=metadata_adapter(lambda size: True)
        )
    )
    with pytest.raises(LayoutCoverageError) as caught:
        prepare_layouts(
            resolve_training(
                registry,
                **{
                    **arguments(),
                    "batch": list(setup.config.batches),
                    "execution": setup.config.execution,
                },
            )
        )
    assert any(
        row.batch == 1 and row.stage == "staging" for row in caught.value.rejected
    )


def test_unexpected_component_failures_propagate_and_expansion_is_bounded():
    model, _, _ = components()
    failure = CoverageError("unexpected metadata factory failure")

    def broken(config, layout):
        raise failure

    registry = installed(model=replace(model.component, input_adapter=broken))
    setup = resolve_training(registry, **arguments())
    with pytest.raises(ValueError, match="bindings exceed"):
        prepare_layouts(setup, max_bindings=1)
    with pytest.raises(CoverageError) as caught:
        prepare_layouts(setup)
    assert caught.value is failure
    registry = installed(
        model=replace(
            model.component, input_adapter=lambda config, layout: InputAdapter()
        )
    )
    with pytest.raises(ValueError, match="padding safety"):
        prepare_layouts(resolve_training(registry, **arguments()))


def test_generic_partition_and_worker_order_controls_every_input_and_output_binding():
    base = resolve_training(installed(), **arguments())
    placement = InputPlacement(
        GroupRef("cooperators", ((2, 0), (3, 1))),
        fields=[
            InputFieldPlacement(tuple(f.path), "slice", 0)
            for f in base.config.inputs.fields
        ],
        outputs={"prediction": OutputPlacement("concat", 0)},
        loss="partitioned",
    )
    setup = resolve_training(
        installed(),
        **arguments(),
        operations={"forward": ["predictions"]},
        parallelism=Parallelism(ranks_per_replica=4, inputs=placement),
    )
    result = prepare_layouts(setup)
    for candidate in next(iter(result.families.values())):
        schedule = candidate.candidate.schedule
        assert set(candidate.bindings) == {
            (buffer, rank)
            for buffer, (partition, _) in enumerate(schedule.locations)
            for rank in placement.partitions.groups[partition]
        }
        for (buffer, rank), binding in candidate.bindings.items():
            assert binding.logical_rank == rank
            count = binding.inputs.shapes["inputs", "pixels"][0]
            assert binding.tensor_shapes["prediction"] == (count, 3)
            assert binding.contributes_loss
            assert (
                binding.outputs.metric_workers(buffer)
                == placement.partitions.groups[schedule.locations[buffer][0]]
            )


def test_global_output_reduction_is_carried_by_the_resolved_portable_contract():
    _, objective, _ = components()
    declared = replace(
        objective.component,
        outputs={
            **objective.component.outputs,
            "predictions": OutputSet(
                model_inputs=[("pixels",)],
                tensors={
                    "total": TensorOutput("float32", (3,), "global", reduction="max"),
                },
            ),
        },
    )
    setup = resolve_training(
        installed(objective=declared),
        **arguments(),
        operations={"forward": ["predictions"]},
    )
    result = prepare_layouts(setup)
    assert setup.config.outputs["predictions"].tensors["total"].reduction == "max"
    for candidate in next(iter(result.families.values())):
        assert candidate.outputs.global_reductions == {"total": "max"}
        assert candidate.bindings[0, 0].tensor_shapes["total"] == (3,)


def test_all_layouts_and_coverage_resolve_with_frameworks_blocked():
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for name in ('torch', 'triton', 'mlops', 'shadowspill_workloads'):
    sys.modules[name] = None
from engine_tests.training.test_setup_resolution import arguments, installed
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.configuration.layouts import prepare_layouts
resolved = resolve_training(installed(), **arguments(),
    operations={'forward_backward': ['metrics'], 'forward': ['predictions']})
layouts = prepare_layouts(resolved)
assert len(layouts.families) == len(layouts.coverage) == 2
assert all(len(family) == 5 for family in layouts.families.values())
""",
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
