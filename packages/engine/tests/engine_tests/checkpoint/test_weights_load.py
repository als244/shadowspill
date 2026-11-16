"""Native weights initialize exact model values and fresh ordinary training state."""

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from engine_tests.checkpoint.test_conversion import WEIGHT, adam
from engine_tests.checkpoint.test_conversion import family as layout_family
from engine_tests.checkpoint.test_state import plan
from engine_tests.checkpoint.test_weights_conversion import exported, saved
from engine_tests.execution.test_shard_initialization import fixture, host_storage
from shadowspill.checkpoint import write_weights
from shadowspill.checkpoint.load import _Reader
from shadowspill.checkpoint.state import PlanState
from shadowspill.checkpoint.weights_conversion import convert_weights
from shadowspill.checkpoint.weights_load import load_weights
from shadowspill.execution.controls import Controls
from shadowspill.execution.executor import Executor
from shadowspill.execution.tensors import visit_tensors


def family(*args, **kwargs):
    """Add the executable control defaults to the metadata-only layout fixture."""
    result = []
    for state in layout_family(*args, **kwargs):
        program = state.plan.program
        rows = [
            {
                **row,
                **({"device": "cpu"} if row["object_id"] is None else {}),
                **(
                    {"initialization": {"kind": "constant", "value": 0.04}}
                    if row["role"] == "hyperparameter"
                    else {}
                ),
            }
            for row in program.bindings["optimizer_bindings"]
        ]
        result.append(
            PlanState(
                plan(
                    replace(
                        program,
                        bindings={**program.bindings, "optimizer_bindings": rows},
                    )
                ),
                {},
                logical_rank=state.logical_rank,
            )
        )
    return result


class HostExecutor:
    """Use the actual initialization, defaults, controls and spill visitors on CPU."""

    initialize = Executor.initialize
    export_tensors = Executor.export_tensors
    tensor_chunks = Executor.tensor_chunks

    def __init__(self, state, worker=None):
        self.state = state
        self.plan, self.plans, self.logical_rank = (
            state.plan,
            state.plans,
            state.logical_rank,
        )
        storage, self.values = host_storage(state.plan.program, state.logical_rank)
        storage.plan = state.plan
        self.controls = Controls(storage)
        self._paths = [SimpleNamespace(storage=storage, controls=self.controls)]
        self._named_paths = {}
        self._distributed = worker
        self._initialized = False

    def synchronize(self):
        pass

    def snapshot(self):
        return exported(self)


@pytest.mark.parametrize("weights", ["master", "compute"])
@pytest.mark.parametrize("master", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_exact_weight_initialization_uses_fresh_moments_counters_and_defaults(
    tmp_path, monkeypatch, weights, master, partial
):
    state = family(1, 0, master=master, frozen=True)[0]
    identity = "exact-base" if partial else None
    source = saved(tmp_path, [state], weights=weights, frozen_state_id=identity)[0][1]
    target = HostExecutor(state)
    target.controls.host["optimizer.linear.weight.step"].fill_(99)
    calls = []

    def base(request, destination):
        calls.append(request.name)
        assert request.name == "base.weight"
        destination.copy_(
            WEIGHT.reshape(-1)[request.offset : request.offset + destination.numel()]
        )

    derived = []
    original = _Reader.derive

    def derive(reader, source, request, destination, chunk_bytes):
        original(reader, source, request, destination, chunk_bytes)
        if (
            request.offset + destination.numel()
            == request.region_offset + request.region_elements
        ):
            derived.append(request.name)
            assert reader.scratch is None

    monkeypatch.setattr(_Reader, "derive", derive)
    metadata = load_weights(
        target,
        source,
        chunk_bytes=17,
        frozen_state_id=identity,
        fill_tensor=base if partial else None,
    )
    actual = target.snapshot()
    selected = WEIGHT.bfloat16().float() if master and weights == "compute" else WEIGHT
    torch.testing.assert_close(
        actual["linear.weight"],
        selected.flatten().to(actual["linear.weight"].dtype),
        rtol=0,
        atol=0,
    )
    if master:
        torch.testing.assert_close(
            actual["master.linear.weight"], selected.flatten(), rtol=0, atol=0
        )
        assert derived
    assert bool(calls) == partial and target._initialized
    assert metadata == {"origin_update": 17}
    assert target.controls.host["optimizer.linear.weight.step"].item() == 0
    assert target.controls.host["optimizer_group.0.lr"].item() == pytest.approx(0.04)
    for name, value in target.values.items():
        if name in {"moment", "variance"}:
            assert not value.count_nonzero()
    # A first Adam update from imported values must use step one and zero moments,
    # independently of the source export's old training boundary (17).
    initial = (
        actual["master.linear.weight" if master else "linear.weight"]
        .float()
        .reshape(4, 12)
    )
    gradient = torch.cos(torch.arange(48).float()).reshape(4, 12)
    actual_update, state_after = adam(
        initial,
        target.values["moment"].reshape(4, 12),
        target.values["variance"].reshape(4, 12),
        gradient,
        target.controls.host["optimizer.linear.weight.step"].item(),
    )
    reference = torch.nn.Parameter(selected.clone())
    optimizer = torch.optim.Adam(
        [reference], lr=0.03, betas=(0.8, 0.95), eps=1e-7, foreach=False
    )
    reference.grad = gradient
    optimizer.step()
    torch.testing.assert_close(actual_update, reference, rtol=0, atol=0)
    assert state_after["step"].item() == 1


def test_cross_layout_conversion_then_fresh_initialization_preserves_master_precision(
    tmp_path,
):
    sources = saved(tmp_path, family(4, 0))
    states = family(1, 1)
    with convert_weights(
        sources, states, tmp_path, scratch_bytes=128, max_disk_bytes=4096
    ) as converted:
        path = tmp_path / "target.ssp"
        with path.open("wb") as stream:
            write_weights(converted.ranks[0], stream, chunk_bytes=17)
    target = HostExecutor(states[0])
    load_weights(target, path, chunk_bytes=17)
    actual = target.snapshot()
    torch.testing.assert_close(
        actual["master.linear.weight"], WEIGHT.flatten(), rtol=0, atol=0
    )
    assert not torch.equal(
        actual["master.linear.weight"], actual["linear.weight"].float()
    )
    assert target.controls.host["optimizer.linear.weight.step"].item() == 0


@pytest.mark.parametrize("provided", [False, True])
def test_nonpersistent_model_buffers_need_bounded_initialization_before_mutation(
    tmp_path, provided
):
    original = family(1, 0, frozen=True)[0]
    program = original.plan.program
    program = replace(
        program,
        bindings={
            **program.bindings,
            "registrations": [
                {**row, "parameter": False, "persistent": False}
                if row["name"].startswith("base.")
                else row
                for row in program.bindings["registrations"]
            ],
        },
    )
    state = PlanState(plan(program), {}, logical_rank=0)
    path = saved(tmp_path, [state])[0][1]
    target = HostExecutor(state)
    before = {name: value.clone() for name, value in target.values.items()}
    calls = []

    def generated(request, destination):
        assert request.name == "base.weight" and destination.nbytes <= 17
        calls.append(request.offset)
        destination.copy_(
            torch.arange(request.offset, request.offset + destination.numel()).sin()
        )

    if not provided:
        with pytest.raises(ValueError, match=r"temporary.*initializer"):
            load_weights(target, path, chunk_bytes=17)
        assert not target._initialized
        for name, value in target.values.items():
            torch.testing.assert_close(
                value, before[name], rtol=0, atol=0, equal_nan=True
            )
    else:
        load_weights(target, path, chunk_bytes=17, fill_tensor=generated)
        assert calls and target._initialized
        torch.testing.assert_close(
            target.values["base"], torch.arange(48).float().sin(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            target.snapshot()["master.linear.weight"], WEIGHT.flatten(), rtol=0, atol=0
        )
        assert target.controls.host["optimizer.linear.weight.step"].item() == 0


@pytest.mark.parametrize("fault", ["corrupt", "missing_base", "wrong_base"])
def test_preflight_failure_never_mutates_existing_state(tmp_path, fault):
    state = family(1, 0, frozen=True)[0]
    path = saved(tmp_path, [state], frozen_state_id="base")[0][1]
    target = HostExecutor(state)
    before = {name: value.clone() for name, value in target.values.items()}
    if fault == "corrupt":
        with path.open("r+b") as stream:
            stream.seek(512)
            first = stream.read(1)[0]
            stream.seek(512)
            stream.write(bytes([first ^ 255]))
    with pytest.raises(ValueError):
        load_weights(
            target,
            path,
            chunk_bytes=17,
            frozen_state_id="different" if fault == "wrong_base" else "base",
        )
    torch.testing.assert_close(target.values, before, rtol=0, atol=0)
    assert not target._initialized


@pytest.mark.parametrize("fault", ["loader", "source_mutation"])
def test_failure_during_initialization_invalidates_the_executor(tmp_path, fault):
    state = family(1, 0, frozen=True)[0]
    path = saved(tmp_path, [state], frozen_state_id="base")[0][1]
    target = HostExecutor(state)
    changed = False

    def fill(request, destination):
        nonlocal changed
        if fault == "loader":
            raise RuntimeError("base loader failed")
        destination.zero_()
        if not changed:
            stamp = path.stat()
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))
            changed = True

    with pytest.raises((ValueError, RuntimeError)), path.open("rb") as stream:
        stream.seek(37)
        try:
            load_weights(
                target, stream, chunk_bytes=17, frozen_state_id="base", fill_tensor=fill
            )
        finally:
            assert stream.tell() == 37 and not stream.closed
    assert not target._initialized


@pytest.mark.parametrize("restore", ["linear.weight", {"unknown"}, {3}, None])
def test_bad_partial_restore_selection_fails_before_tensor_callbacks(restore):
    target = HostExecutor(family(1, 0)[0])
    before = {name: value.clone() for name, value in target.values.items()}
    with pytest.raises((ValueError, TypeError)):
        target.initialize(
            lambda *args: pytest.fail("touched model values"), restore=restore
        )
    torch.testing.assert_close(target.values, before, rtol=0, atol=0)


def test_selected_host_state_uses_canonical_names_and_other_controls_reset():
    target = HostExecutor(family(1, 0)[0])
    seen = set()

    def fill(request, destination):
        seen.add(request.name)
        destination.fill_(7)

    target.initialize(fill, restore={"optimizer.linear.weight.step"}, chunk_bytes=17)
    assert target.controls.host["optimizer.linear.weight.step"].item() == 7
    assert target.controls.host["optimizer_group.0.lr"].item() == pytest.approx(0.04)
    assert "optimizer.linear.weight.step" in seen
    assert "master.linear.weight" not in seen


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("elements", [1, 15])
def test_model_only_selection_clears_accumulated_gradients_and_shard_padding(
    rank, elements
):
    storage, values = host_storage(fixture(elements, kind="forward_backward"), rank)
    calls = []

    def fill(request, destination):
        calls.append(request.name)
        destination.fill_(2)

    visit_tensors(storage, fill, 12, initialize=True, restore={"layer.weight"})
    assert set(calls) <= {"layer.weight"}
    assert not values["reduced"].count_nonzero()


def test_private_plan_control_selection_uses_its_exported_prefix():
    target = HostExecutor(family(1, 0)[0])
    seen = []

    def fill(request, destination):
        seen.append(request.name)
        destination.fill_(0.75)

    target.controls.initialize(
        fill, restore={"plans.extra.optimizer_group.0.lr"}, prefix="plans.extra."
    )
    assert seen == ["plans.extra.optimizer_group.0.lr"]
    assert target.controls.host["optimizer.linear.weight.step"].item() == 0
    assert target.controls.host["optimizer_group.0.lr"].item() == 0.75
