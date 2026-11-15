"""Saved state descriptions preserve execution ownership without opening workers."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from engine_tests.execution.test_shard_initialization import fixture, host_storage
from engine_tests.offline.test_shared_plans import registered
from engine_tests.offline.test_state_inventory import training
from shadowspill.checkpoint.state import PlanState
from shadowspill.execution.controls import Controls, control_descriptors
from shadowspill.execution.tensors import tensor_descriptors
from shadowspill.ir import TransferPhase, TransferSpec


def plan(program, *, kind="step"):
    program = replace(
        program,
        bindings={
            **program.bindings,
            "kind": kind,
            "root_inputs": [],
            "root_sources": [],
        },
    )
    return SimpleNamespace(program=program, _costed_program=program)


def test_canonical_state_owners_and_private_controls_need_no_tensor_allocation(
    monkeypatch,
):
    primary = plan(registered(), kind="forward")
    plans = {name: plan(training(prefix=name)) for name in ("update", "second")}
    monkeypatch.setattr(torch, "empty", lambda *a, **k: pytest.fail("allocated tensor"))
    monkeypatch.setattr(
        torch, "empty_strided", lambda *a, **k: pytest.fail("allocated host control")
    )
    state = PlanState(primary, plans, logical_rank=0)
    chunks = list(state.tensor_chunks())
    assert len({chunk.name for chunk in chunks}) == len(chunks)
    assert {chunk.name for chunk in chunks} == {
        "linear.weight",
        "master.linear.weight",
        "optimizer.linear.weight.exp_avg",
        "optimizer.linear.weight.step",
        "plans.update.optimizer_group.0.lr",
        "plans.second.optimizer_group.0.lr",
    }
    assert next(c for c in chunks if c.name.endswith(".step")).dtype == torch.int64


@pytest.mark.parametrize("rank", range(4))
def test_rank_layout_matches_actual_owned_export_ranges_and_padding(rank):
    source = fixture(elements=15)
    # The fixture's compute alias is padded; the costed transfer owns eight bf16
    # elements, including the tail excluded by the existing region resolver.
    bindings = {
        **source.bindings,
        "registrations": [
            {**item, "parameter": True} for item in source.bindings["registrations"]
        ],
    }
    source = replace(source, bindings=bindings)
    target = plan(source)
    target._costed_program = replace(
        source,
        alias_groups=tuple(
            replace(
                a,
                transfer=TransferSpec(
                    16,
                    (TransferPhase("copy", bytes=16),),
                    (TransferPhase("copy", bytes=16),),
                ),
            )
            if a.alias_group_id == "weight"
            else a
            for a in source.alias_groups
        ),
    )
    actual, _ = host_storage(source, rank)
    expected = list(tensor_descriptors(actual))
    state = PlanState(target, {}, logical_rank=rank)
    assert list(state.tensor_chunks()) == expected
    weight = expected[0]
    assert weight.storage_elements == 15
    assert (weight.region_offset, weight.region_elements) == (
        (8, 7) if rank % 2 == 0 else (0, 8)
    )


@pytest.mark.parametrize("shape,stride", [((), ()), ((2, 3), (1, 4)), ((0, 3), (3, 1))])
def test_control_description_matches_exported_storage_including_empty(shape, stride):
    binding = {
        "name": "count",
        "role": "state",
        "object_id": None,
        "device": "cpu",
        "shape": shape,
        "stride": stride,
        "dtype": "torch.int64",
    }
    storage = SimpleNamespace(
        plan=SimpleNamespace(
            program=SimpleNamespace(bindings={"optimizer_bindings": [binding]})
        )
    )
    controls = Controls(storage)
    exported = []
    controls.export(lambda chunk, tensor: exported.append(chunk), chunk_bytes=4096)
    assert list(control_descriptors([binding])) == exported
