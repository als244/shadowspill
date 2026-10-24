"""Persistent exports preserve contracts, occurrence data, and gradient creators."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export.graph_signature import InputKind

from shadowspill.ir import TaskAlternativeChoice
from shadowspill.pipeline.common import PlanningTimer
from shadowspill.planner import StepDataOrdering
from shadowspill.pytorch.capture import fake
from shadowspill.pytorch.capture.retention import RetentionPolicy
from shadowspill.pytorch.graph_pairs import GraphPairStore
from shadowspill.pytorch.guards import capture_training_signatures
from shadowspill.pytorch.lowering.training import lower_partitioned_training_program
from shadowspill.pytorch.optimizer import capture_optimizer
from shadowspill.pytorch.planning.training.capture import (
    _capture_training_objectives,
    _partition_training_graphs,
)
from shadowspill.pytorch.prepared import PreparedStep
from tests.benchmarking._fixtures import _fixture
from tests.shadowspill.pytorch.lowering.test_training_lowering import (
    _measurement,
    _MultiLinearModel,
    _objective,
)


class _TiedWithBuffer(torch.nn.Module):
    def __init__(self, *, persistent=True):
        super().__init__()
        self.first = torch.nn.Linear(3, 2)
        self.second = self.first
        self.register_buffer(
            "offset", torch.arange(2, dtype=torch.float32), persistent=persistent
        )

    def forward(self, value):
        return self.first(value) + self.second(value) + self.offset


def _with_constant(model, values, targets):
    return (model(values) - targets).square().sum() + torch.tensor(0.125)


def _prepared(monkeypatch, tmp_path, *, model=None, objective=_objective):
    # Exercise genuine strict Export and AOT on CPU; no runtime or GPU launch.
    monkeypatch.setattr(fake, "accelerator_device", lambda _: torch.device("cpu"))
    import shadowspill.pytorch.lowering.catalog as catalog

    monkeypatch.setattr(catalog, "DEVICE_TYPE", "cpu")
    model = _MultiLinearModel() if model is None else model
    inputs = tuple((torch.randn(2, 3), torch.randn(2, 2)) for _ in range(4))
    signatures = capture_training_signatures(inputs)
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake.fake_device_model(model, mode)
    captures = _capture_training_objectives(
        replica,
        objective,
        inputs,
        signatures=signatures,
        fake_mode=mode,
        device_ordinal=0,
        stores=SimpleNamespace(archive_export=lambda *a, **k: "digest"),
        timer=PlanningTimer(verbose=False),
    )
    prepared = PreparedStep.from_capture(
        model,
        SimpleNamespace(captures=captures, signatures=signatures),
        options={},
    )
    prepared = replace(
        prepared,
        programs=(
            replace(
                _fixture(),
                signature_digests=prepared.signature_digests,
                data_ordering=StepDataOrdering(2, 2),
            ),
        ),
    )
    prepared.save(tmp_path)
    return model, inputs, mode, replica, captures, prepared


def test_prepared_export_round_trip_uses_fresh_inputs_without_export(
    monkeypatch, tmp_path
):
    model, inputs, _mode, _replica, original, prepared = _prepared(
        monkeypatch, tmp_path
    )
    loaded = PreparedStep.load(tmp_path)
    assert len(loaded.exports) == 1
    assert loaded.positions == (0, 0, 0, 0)
    assert loaded.programs[0].digest == prepared.programs[0].digest
    loaded.validate(model, [s.digest for s in capture_training_signatures(inputs)], {})
    monkeypatch.setattr(
        torch.export, "export", lambda *a, **k: pytest.fail("re-exported")
    )
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake.fake_device_model(model, mode)
    rebound = loaded.bind(replica, inputs, fake_mode=mode, device_ordinal=0)
    assert len({id(c.exported.exported_program) for c in rebound}) == 1
    assert len({id(c.exported.flat_inputs[-1]) for c in rebound}) == 4
    for old, new in zip(original, rebound, strict=True):
        assert old.objective_schema == new.objective_schema
        assert str(old.exported.exported_program.graph) == str(
            new.exported.exported_program.graph
        )
    with pytest.raises(Exception, match="signatures differ"):
        loaded.validate(model, ["different"] * 4, {})
    with pytest.raises(Exception, match="options differ"):
        loaded.validate(model, prepared.signature_digests, {"different": True})
    data = next(tmp_path.glob("*-export-0.pt2"))
    data.write_bytes(b"interrupted write")
    with pytest.raises(ValueError, match="checksum"):
        PreparedStep.load(tmp_path)


@pytest.mark.parametrize("persistent", [True, False])
def test_reloaded_export_preserves_tied_state_buffers_and_lifted_constants(
    monkeypatch, tmp_path, persistent
):
    model, inputs, _mode, _replica, original, _ = _prepared(
        monkeypatch,
        tmp_path,
        model=_TiedWithBuffer(persistent=persistent),
        objective=_with_constant,
    )
    loaded = PreparedStep.load(tmp_path)
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake.fake_device_model(model, mode)
    captures = loaded.bind(replica, inputs, fake_mode=mode, device_ordinal=0)
    assert replica.first.weight is replica.second.weight
    for capture in captures:
        program = capture.exported.exported_program
        for spec, value in zip(
            program.graph_signature.input_specs,
            capture.exported.flat_inputs,
            strict=True,
        ):
            if spec.kind is InputKind.BUFFER:
                assert value is replica.offset
                assert (spec.target in program.state_dict) is persistent
                assert (spec.target in program.constants) is not persistent
            elif spec.kind is InputKind.CONSTANT_TENSOR:
                assert not value.is_meta, (
                    "lifted constant lost its value during persistence"
                )
                assert value.item() == 0.125
    partitioned = _partition_training_graphs(
        model,
        captures,
        inputs,
        fake_mode=mode,
        partition="auto",
        stores=SimpleNamespace(graph_pairs=GraphPairStore()),
        timer=PlanningTimer(verbose=False),
        retention=RetentionPolicy(),
        grad_dtype=None,
        fuse_narrow_gradient_accumulation=True,
    )
    assert len(partitioned) == len(original)


def test_fresh_binding_imports_lazy_custom_operator_providers(monkeypatch, tmp_path):
    import importlib
    import subprocess
    import sys

    (tmp_path / "lazy_prepared_ops.py").write_text(
        """
import torch

@torch.library.custom_op("shadowspill_prepared_test::scale", mutates_args=())
def scale(value: torch.Tensor) -> torch.Tensor:
    return value * 2

@scale.register_fake
def scale_fake(value):
    return torch.empty_like(value)

"""
    )
    (tmp_path / "lazy_prepared_autograd.py").write_text(
        """
from lazy_prepared_ops import scale

def backward(ctx, gradient):
    return gradient * 2

scale.register_autograd(backward)
"""
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    provider = importlib.import_module("lazy_prepared_ops")
    importlib.import_module("lazy_prepared_autograd")

    def objective(model, values, target):
        return (provider.scale(model(values)) - target).square().sum()

    *_, prepared = _prepared(monkeypatch, tmp_path / "artifact", objective=objective)
    assert prepared.operator_modules == ("lazy_prepared_autograd", "lazy_prepared_ops")
    code = """
import sys, torch
from pathlib import Path
from torch._subclasses.fake_tensor import FakeTensorMode
from shadowspill.pytorch.capture import fake
from shadowspill.pytorch.prepared import PreparedStep
from tests.shadowspill.pytorch.lowering.test_training_lowering import _MultiLinearModel

sys.path.insert(0, sys.argv[1])
assert 'lazy_prepared_ops' not in sys.modules
prepared = PreparedStep.load(Path(sys.argv[1]) / 'artifact')
assert 'lazy_prepared_ops' not in sys.modules
fake.accelerator_device = lambda _: torch.device('cpu')
mode = FakeTensorMode(allow_non_fake_inputs=True)
replica = fake.fake_device_model(_MultiLinearModel(), mode)
inputs = tuple((torch.randn(2, 3), torch.randn(2, 2)) for _ in range(4))
def no_export(*args, **kwargs):
    raise AssertionError('admission exported again')
torch.export.export = no_export
captures = prepared.bind(replica, inputs, fake_mode=mode, device_ordinal=0)
assert len(captures) == 4
assert 'lazy_prepared_ops' in sys.modules
assert 'lazy_prepared_autograd' in sys.modules
torch.testing.assert_close(
    torch.ops.shadowspill_prepared_test.scale(torch.tensor([3.])), torch.tensor([6.])
)
"""
    result = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            code,
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_prepared_training_bundle_round_trips_in_a_fresh_process(monkeypatch, tmp_path):
    import json
    import subprocess
    import sys

    from shadowspill.pytorch import search_prepared_steps
    from shadowspill.training import PreparedTraining

    _, _, _, _, _, prepared = _prepared(monkeypatch, tmp_path / "step")
    search = search_prepared_steps(
        {"geometry": prepared}, budgets=[(224, 1024), (320, 1024)]
    )
    path = tmp_path / "bundle"
    PreparedTraining({"geometry": prepared}, search).save(path)
    code = """
import json, sys
from shadowspill.training import PreparedTraining
bundle = PreparedTraining.load(sys.argv[1])
print(json.dumps([plan.digest for plan in bundle.search.winner_plans.values()]))
"""
    result = subprocess.run(
        [sys.executable, *(["-S"] if sys.flags.no_site else []), "-c", code, str(path)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout) == [p.digest for p in search.winner_plans.values()]
    # Publishing a different candidate bundle leaves the original directory
    # referenced by the old manifest intact.
    old = json.loads((path / "training.json").read_text())
    changed = replace(prepared, options={"partition": "whole"})
    PreparedTraining({"geometry": changed}, search).save(path)
    previous = PreparedStep.load(path / old["candidates"]["geometry"])
    assert previous.options == prepared.options


def test_budget_load_reads_only_the_selected_plan_and_ordering(monkeypatch, tmp_path):
    import json

    from shadowspill.pytorch import search_prepared_steps
    from shadowspill.training import PreparedTraining

    *_, prepared = _prepared(monkeypatch, tmp_path / "step")
    prepared = replace(
        prepared,
        programs=(
            *prepared.programs,
            replace(prepared.programs[0], data_ordering=StepDataOrdering(4, 1)),
        ),
    )
    budgets = [(224, 1024), (320, 1024)]
    search = search_prepared_steps({"geometry": prepared}, budgets=budgets)
    path = tmp_path / "bundle"
    PreparedTraining({"geometry": prepared}, search).save(path)
    manifest = json.loads((path / "training.json").read_text())
    candidate = path / manifest["candidates"]["geometry"]
    files = json.loads((candidate / "manifest.json").read_text())["files"]
    # Damage payloads the requested budget does not use. A selective load must
    # neither read nor deserialize these; a full load must still validate them.
    assert search.winner(*budgets[0]).ordering == prepared.programs[0].data_ordering
    (candidate / files["program-1.json"]["file"]).write_bytes(b"unused program")
    unused_plan = next(p for p in manifest["plans"] if tuple(p["budget"]) == budgets[1])
    (path / unused_plan["file"]).write_bytes(b"unused plan")
    selected = PreparedTraining.load(path, budget=budgets[0])
    assert selected.search.budgets == (budgets[0],)
    assert list(selected.search.winner_plans) == [budgets[0]]
    assert len(selected.candidates["geometry"].programs) == 1
    assert selected.search.winner_plans[budgets[0]].digest == (
        search.winner_plans[budgets[0]].digest
    )
    with pytest.raises(ValueError, match="checksum"):
        PreparedTraining.load(path)
    with pytest.raises(ValueError, match="not searched"):
        PreparedTraining.load(path, budget=(999, 1024))
    (candidate / files["program-0.json"]["file"]).write_bytes(b"damaged selection")
    with pytest.raises(ValueError, match="checksum"):
        PreparedTraining.load(path, budget=budgets[0])


@pytest.mark.parametrize(
    "ordering", [StepDataOrdering(4, 1), StepDataOrdering(2, 2), StepDataOrdering(1, 4)]
)
@pytest.mark.parametrize("variant", ["save", "recompute"])
def test_reload_preserves_first_backward_creation_and_later_accumulation(
    monkeypatch, tmp_path, ordering, variant
):
    model, inputs, mode, replica, captures, _prepared_step = _prepared(
        monkeypatch, tmp_path
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=False)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer_capture = capture_optimizer(dict(model.named_parameters()), optimizer)

    def lower(captures, mode, replica):
        partitioned = _partition_training_graphs(
            model,
            captures,
            inputs,
            fake_mode=mode,
            partition="auto",
            stores=SimpleNamespace(graph_pairs=GraphPairStore()),
            timer=PlanningTimer(verbose=False),
            retention=RetentionPolicy(),
            grad_dtype=None,
            fuse_narrow_gradient_accumulation=True,
        )
        artifacts = [
            artifact
            for part in partitioned
            for stage in part.stages
            for variant in stage.graph_pairs.variants
            for artifact in (variant.pair.forward, variant.pair.backward)
        ]
        artifacts += [
            optimizer_capture.update,
            *(t.artifact for t in optimizer_capture.update_tasks),
        ]
        measurements = {a.compatibility_digest: _measurement(a) for a in artifacts}
        return lower_partitioned_training_program(
            replica,
            partitioned,
            measurements,
            optimizer_capture,
            data_ordering=ordering,
        )

    before = lower(captures, mode, replica)
    loaded = PreparedStep.load(tmp_path)
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake.fake_device_model(model, mode)
    captures = loaded.bind(replica, inputs, fake_mode=mode, device_ordinal=0)
    after = lower(captures, mode, replica)
    assert before.program.digest == after.program.digest
    gradients = {item.gradient_object_id for item in after.gradients}
    choices = tuple(
        TaskAlternativeChoice(g.group_id, variant)
        for g in after.program.task_alternative_groups
    )
    selected = after.program.selected_tasks(choices)
    created, accumulated = set(), {key: 0 for key in gradients}
    for task in selected:
        if task.phase != "backward":
            continue
        produces = gradients.intersection(task.outputs)
        adds = gradients.intersection(m.object_id for m in task.mutations)
        assert not produces.intersection(created), (
            "a later backward overwrote a gradient"
        )
        assert adds <= created, "accumulation ran before the gradient existed"
        created.update(produces)
        for key in adds:
            accumulated[key] += 1
    assert created == gradients
    assert set(accumulated.values()) == {3}
