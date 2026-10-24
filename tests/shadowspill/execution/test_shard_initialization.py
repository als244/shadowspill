"""Bounded original-tensor requests over padded, logically ordered host shards."""

from types import SimpleNamespace

import pytest
import torch

from shadowspill import Parallelism, Program
from shadowspill.artifacts.tensors import tensor_geometry
from shadowspill.execution.tensor_regions import tensor_regions
from shadowspill.execution.tensors import tensor_entries, visit_tensors
from shadowspill.ir import AliasGroupSpec, DeviceSpec, ObjectSpec


def fixture(elements=15, kind="step"):
    capacity = (elements + 1) // 2
    objects = [ObjectSpec("weight", "weight", 0, elements * 2)]
    aliases = [AliasGroupSpec("weight", "device", capacity * 4)]
    layout = {"dtype": "torch.bfloat16", "shape": (elements,), "stride": (1,)}
    groups = ((1, 0), (3, 2))
    bindings = {
        "kind": kind,
        "entrypoints": [],
        "task_inputs": {},
        "root_inputs": [],
        "root_sources": [],
        "registrations": [{"name": "layer.weight", "object_id": "weight"}],
        "model_tensor_geometry": {"weight": layout},
        "parameter_ownership": [
            {
                "names": ("layer.weight",),
                "local_shape": (elements,),
                "optimizer": {"elements": elements, "groups": groups},
                "gradients": {"elements": elements, "groups": groups},
            }
        ],
        "weight_shards": {
            "weight": {
                "groups": groups,
                "shard_bytes": capacity * 2,
                "logical_bytes": elements * 2,
            }
        },
    }
    if kind == "step":
        rows = []
        for value, name, role, dtype, initializer in (
            (
                "master",
                "layer.weight",
                "parameter",
                "torch.float32",
                {"kind": "parameter", "name": "layer.weight"},
            ),
            (
                "moment",
                "optimizer.layer.weight.exp_avg",
                "state",
                "torch.float32",
                {"kind": "constant", "value": 0},
            ),
            ("compute", "compute.layer.weight", "compute_copy", "torch.bfloat16", None),
        ):
            width = 2 if value == "compute" else 4
            alias = "weight" if value == "compute" else value
            objects.append(ObjectSpec(value, alias, 0, capacity * width))
            if value != "compute":
                aliases.append(AliasGroupSpec(alias, "device", capacity * width))
            rows.append(
                {
                    "name": name,
                    "parameter_name": "layer.weight",
                    "role": role,
                    "object_id": value,
                    "shape": (capacity,),
                    "stride": (1,),
                    "dtype": dtype,
                    "initialization": initializer,
                }
            )
        bindings["optimizer_bindings"] = rows
    else:
        aliases.append(AliasGroupSpec("reduced", "device", capacity * 4))
        objects.append(ObjectSpec("reduced", "reduced", 0, capacity * 4))
        bindings["gradients"] = [
            {
                "parameter_name": "layer.weight",
                "gradient_object_id": "local_unused",
            }
        ]
        bindings["persistent_gradients"] = [
            {
                "parameter_name": "layer.weight",
                "gradient_object_id": "reduced",
                "shape": (capacity,),
                "dtype": "torch.float32",
            }
        ]
        bindings["model_tensor_geometry"]["reduced"] = {
            "shape": (capacity,),
            "stride": (1,),
            "dtype": "torch.float32",
        }
    return Program(
        devices=(DeviceSpec("device", "worker", "cuda", 0),),
        alias_groups=tuple(aliases),
        objects=tuple(objects),
        tasks=(),
        task_definitions=(),
        parallelism=Parallelism.dp(ranks_per_replica=2, num_replicas=2),
        bindings=bindings,
    )


def host_storage(program, rank):
    state = SimpleNamespace(
        plan=SimpleNamespace(program=program),
        geometry=tensor_geometry(program),
        regions=tensor_regions(program, rank),
        aliases={item.object_id: item.alias_group_id for item in program.objects},
        sizes={item.alias_group_id: item.size_bytes for item in program.alias_groups},
        shared_aliases={},
    )
    values = {}
    for _, value, _ in tensor_entries(state):
        alias = state.aliases[value]
        region = state.regions[value]
        dtype = getattr(torch, region.geometry["dtype"].removeprefix("torch."))
        values.setdefault(alias, torch.full((region.capacity,), -9, dtype=dtype))
    state.initial = set(values)
    state.host = lambda alias: values[alias].view(torch.uint8)
    return state, values


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("elements", [1, 15])
def test_original_offsets_bounded_masters_and_padding(rank, elements):
    state, values = host_storage(fixture(elements), rank)
    region = state.regions["weight"]
    calls = []

    def fill(request, chunk):
        assert chunk.numel() * chunk.element_size() <= 12
        assert request.shape == (elements,)
        assert request.storage_elements == elements
        calls.append((request.name, request.offset, chunk.numel()))
        chunk.copy_(torch.arange(request.offset, request.offset + chunk.numel()) + 1)

    visit_tensors(state, fill, 12, initialize=True)
    assert {item[0] for item in calls} == ({"layer.weight"} if region.length else set())
    assert sum(item[2] for item in calls) == region.length
    if calls:
        assert calls[0][1] == region.offset
    expected = torch.zeros(region.capacity, dtype=torch.bfloat16)
    expected[: region.length] = (
        torch.arange(region.offset, region.offset + region.length) + 1
    )
    torch.testing.assert_close(values["weight"], expected)
    torch.testing.assert_close(values["master"], expected.float())
    assert not values["moment"].count_nonzero()

    calls.clear()
    visit_tensors(state, fill, 12, initialize=True, restore=True)
    names = {"layer.weight", "master.layer.weight", "optimizer.layer.weight.exp_avg"}
    assert {item[0] for item in calls} == (names if region.length else set())
    assert sum(item[2] for item in calls) == 3 * region.length
    exported = []
    visit_tensors(
        state,
        lambda request, chunk: exported.append((request, chunk.clone())),
        12,
        initialize=False,
    )
    assert sum(chunk.numel() for _, chunk in exported) == 3 * region.length
    assert all(
        request.offset + chunk.numel() <= elements for request, chunk in exported
    )


def test_backward_initializes_and_exports_reduced_shard_not_local_contribution():
    state, values = host_storage(fixture(kind="forward_backward"), 0)
    visit_tensors(state, lambda request, chunk: chunk.fill_(2), 12, initialize=True)
    assert not values["reduced"].count_nonzero()
    names = []
    visit_tensors(
        state, lambda request, _: names.append(request.name), 12, initialize=False
    )
    assert set(names) == {"layer.weight", "gradient.layer.weight"}


def test_wrong_host_capacity_rejected_before_loading():
    state, _ = host_storage(fixture(), 0)
    state.host = lambda _: torch.empty(32, dtype=torch.uint8)
    with pytest.raises(ValueError, match="declared shard"):
        visit_tensors(
            state, lambda *_: pytest.fail("must reject first"), 12, initialize=True
        )
