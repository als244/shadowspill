"""One capture per physical computation, with explicit saved count/operation uses."""

import subprocess
import sys
from dataclasses import replace

import pytest

from engine_tests.offline.test_example_instances import captured
from engine_tests.training.test_setup_resolution import arguments, installed
from shadowspill.parallelism import (
    GroupRef,
    InputFieldPlacement,
    InputPlacement,
    OutputPlacement,
    Parallelism,
)
from shadowspill.training.definitions import (
    PreparationDefinition,
    RepresentativePattern,
    RepresentativeSample,
)
from shadowspill.training.inputs.shapes import BufferShape
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.templates import (
    TemplateUse,
    capture_inventory,
    instantiate_template,
)
from shadowspill_client import (
    BatchConfig,
    BuildCandidates,
    PlanningOptions,
    Preparation,
)
from shadowspill_client.protocol.operations import OperationKey


def inventory(**overrides):
    return capture_inventory(
        prepare_layouts(
            resolve_training(
                installed(),
                **{
                    **arguments(),
                    "operations": {
                        "step": ["metrics"],
                        "forward_backward": ["metrics"],
                        "forward": ["predictions"],
                    },
                    **overrides,
                },
            )
        )
    )


def test_step_backward_counts_and_masked_variants_share_exact_captures():
    result = inventory()
    assert len(result.templates) == 9  # Three sizes, step/masked/evaluation each.
    step = result.families[OperationKey("step", "metrics")]
    backward = result.families[OperationKey("forward_backward", "metrics")]
    assert len(step) == len(backward) == 5
    for complete, split in zip(step, backward, strict=True):
        a, b = complete.workers[0], split.workers[0]
        assert a.main.template == b.main.template
        assert a.masked == b.masked
        assert a.main.kind == "step" and b.main.kind == "forward_backward"
        assert a.optimizer is None and b.optimizer.kind == "optim_step"
        assert b.optimizer.template == b.main.template
        assert b.main.count == split.layout.candidate.count
        for use in (b.main, b.masked):
            template = result.templates[use.template]
            assert template.binding.outputs.schedule.name == "capture"
            assert len(template.binding.outputs.schedule.buffers) == 1
    assert step[0].workers[0].main.template == step[1].workers[0].main.template
    assert step[0].workers[0].main.count != step[1].workers[0].main.count
    # Input/output declarations remain selective, and immutable identities are
    # unaffected by edits to an inspection result.
    evaluation = result.families[OperationKey("forward", "predictions", "evaluation")]
    ref = evaluation[0].workers[0]
    assert ref.masked is ref.optimizer is None
    template = result.templates[ref.main.template]
    assert template.binding.inputs.shapes.keys() == {("inputs", "pixels")}
    signature = template.signature()
    signature["binding"]["operation"]["mode"] = "training"
    assert template.signature()["binding"]["operation"]["mode"] == "evaluation"
    assert list(inventory().templates) == list(result.templates)
    with pytest.raises(TypeError):
        result.templates["changed"] = template


def test_optimizer_free_backward_and_rank_ownership_are_not_conflated():
    result = inventory(optimizer=None, operations={"forward_backward": ["metrics"]})
    assert len(result.templates) == 6
    assert {t.kind for t in result.templates.values()} == {
        "forward_backward",
        "forward",
    }
    assert all(
        row.workers[0].optimizer is None
        for rows in result.families.values()
        for row in rows
    )
    base = result.layouts.setup
    placement = InputPlacement(
        GroupRef("cooperators", ((2, 0), (3, 1))),
        outputs={"prediction": OutputPlacement()},
        fields=[InputFieldPlacement(tuple(f.path)) for f in base.config.inputs.fields],
    )
    distributed = inventory(
        parallelism=Parallelism(ranks_per_replica=4, inputs=placement)
    )
    for rows in distributed.families.values():
        for row in rows:
            assert row.workers.keys() == {0, 1, 2, 3}
            assert len({w.main.template for w in row.workers.values()}) == 4
            for rank, worker in row.workers.items():
                template = distributed.templates[worker.main.template]
                assert template.binding.logical_rank == rank
                assert template.binding.buffer == placement.partition_index(rank)
                assert template.binding.contributes_loss == (rank in (2, 3))
                assert len(template.binding.outputs.schedule.buffers) == 2


def test_same_geometry_with_distinct_timing_scenarios_keeps_distinct_templates():
    registry = installed()
    source = next(iter(inventory().layouts.setup.config.definition_sources.values()))
    preset = PreparationDefinition(
        "external/scenarios@1",
        lambda context: Preparation(
            candidates=BuildCandidates(microbatch_sizes=[4]),
            planning=PlanningOptions(microbatch_counts=[1]),
        ),
        lambda context, batch, paths, size: [
            BufferShape(
                size,
                {
                    tuple(f.path): (size, *f.shape)
                    for f in context.schema.fields
                    if tuple(f.path) in paths
                },
            )
        ],
        lambda context, batch, physical: RepresentativePattern(
            "custom",
            [
                RepresentativeSample(
                    dict.fromkeys(physical.fields, 1), count=physical.max_datums
                )
            ],
        ),
    )
    registry.add_preparation(preset, source=source)
    result = capture_inventory(
        prepare_layouts(
            resolve_training(
                registry,
                **{
                    **arguments(),
                    "batch": [
                        BatchConfig(limits={"max_examples": 4}, typical={"pattern": n})
                        for n in (1, 2)
                    ],
                    "prepare": Preparation(preset=preset.id),
                },
            )
        )
    )
    assert len(result.templates) == 4
    assert {
        t.signature()["scenario"]["typical"]["pattern"]
        for t in result.templates.values()
    } == {1, 2}


def test_saved_template_identity_is_checked_before_cpu_derivation(tmp_path):
    result = inventory()
    use = result.families[OperationKey("forward_backward", "metrics")][1].workers[0]
    template = result.templates[use.main.template]
    original = captured()
    program = replace(
        original,
        bindings={
            **original.bindings,
            "preparation_template": template.signature(),
            "profiling_scenario": template.signature()["scenario"],
        },
    )
    backward = instantiate_template(template, use.main, program)
    optimizer = instantiate_template(template, use.optimizer, program)
    assert backward.bindings["kind"] == "forward_backward"
    assert backward.bindings["num_examples"] == 2
    assert optimizer.bindings["kind"] == "optim_step"
    assert (
        backward.bindings["gradient_geometry"]
        == optimizer.bindings["gradient_geometry"]
    )
    assert {d.profile_key for d in backward.task_definitions} <= {
        d.profile_key for d in program.task_definitions
    }
    for bad_use, bad_program in (
        (use.main, original),
        (replace(use.main, template="wrong"), program),
        (replace(use.main, count=0), program),
        (replace(use.main, count=True), program),
        (
            use.main,
            replace(
                program,
                bindings={
                    **program.bindings,
                    "component_layouts": [
                        {
                            "parameter": "weight",
                            "component_path": ["payload"],
                            "axes": [0, 1],
                            "encoding": "changed@1",
                        }
                    ],
                },
            ),
        ),
    ):
        with pytest.raises(ValueError, match="differs"):
            instantiate_template(template, bad_use, bad_program)
    with pytest.raises(ValueError, match="cannot supply"):
        instantiate_template(template, TemplateUse(template.id, "forward", 1), program)
    with pytest.raises(ValueError, match="optimizer requires"):
        instantiate_template(template, replace(use.optimizer, count=2), program)


def test_capture_inventory_is_available_without_framework_or_catalog():
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
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.templates import capture_inventory
setup = resolve_training(installed(), **arguments())
result = capture_inventory(prepare_layouts(setup))
assert len(result.templates) == 6
assert 'torch' not in [n for n, v in sys.modules.items() if v is not None]
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
