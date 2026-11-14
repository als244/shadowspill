"""Shared-plan admission works with portable names and byte counts, without CUDA."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from shadowspill.artifacts.tensors import shared_model_aliases
from shadowspill.ir import AliasGroupSpec, MemoryLocation, ObjectSpec, ResidencySpec
from shadowspill.planning.shared import _spill_bytes

from ._example import program


def registered(*, prefix="", tied=True, dtype="torch.float32"):
    names = ("linear.weight", "head.weight")
    aliases = tuple(
        AliasGroupSpec(prefix + str(i), "rank0", 64) for i in range(1 if tied else 2)
    )
    objects = tuple(
        ObjectSpec(prefix + name, aliases[i % len(aliases)].alias_group_id, 0, 64)
        for i, name in enumerate(names)
    )
    return replace(
        program(),
        alias_groups=aliases,
        objects=objects,
        tasks=(),
        initial_residency=(),
        final_residency=(),
        bindings={
            "entrypoints": [],
            "registrations": [
                {"name": name, "object_id": obj.object_id, "parameter": True}
                for name, obj in zip(names, objects, strict=True)
            ],
            "model_tensor_geometry": {
                obj.object_id: {
                    "dtype": dtype,
                    "shape": [4, 4],
                    "stride": [4, 1],
                    "storage_offset": 0,
                }
                for obj in objects
            },
        },
    )


def test_matches_different_ids_and_preserves_ties():
    assert shared_model_aliases(registered(), registered(prefix="eval-")) == {
        "eval-0": "0"
    }


@pytest.mark.parametrize("source_tied,target_tied", [(True, False), (False, True)])
def test_rejects_different_aliasing(source_tied, target_tied):
    with pytest.raises(ValueError, match="aliasing differs"):
        shared_model_aliases(registered(tied=source_tied), registered(tied=target_tied))


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "torch.bfloat16"),
        ("shape", [2, 8]),
        ("stride", [1, 4]),
        ("storage_offset", 1),
    ],
)
def test_rejects_incompatible_geometry(field, value):
    other = registered(prefix="eval-")
    bindings = dict(other.bindings)
    bindings["model_tensor_geometry"] = {
        key: {**geometry, field: value}
        for key, geometry in bindings["model_tensor_geometry"].items()
    }
    with pytest.raises(ValueError, match="layout differs"):
        shared_model_aliases(registered(), replace(other, bindings=bindings))


def test_rejects_missing_tensor_and_missing_geometry():
    other = registered()
    bindings = dict(other.bindings)
    bindings["registrations"] = [
        {**item, "name": "absent"} for item in bindings["registrations"]
    ]
    with pytest.raises(ValueError, match="absent from default"):
        shared_model_aliases(registered(), replace(other, bindings=bindings))
    bindings = {**other.bindings, "model_tensor_geometry": {}}
    with pytest.raises(ValueError, match="no portable layout"):
        shared_model_aliases(registered(), replace(other, bindings=bindings))


def test_joint_spill_keeps_private_inputs_and_deduplicates_shared_model():
    def plan(model, private, size, peak, *, retained):
        aliases = (
            AliasGroupSpec(model, "rank0", 64, retain_spill_copy=retained),
            AliasGroupSpec(private, "rank0", size, retain_spill_copy=True),
        )
        entries = tuple(
            ResidencySpec(a.alias_group_id, MemoryLocation.SPILL) for a in aliases
        )
        return SimpleNamespace(
            program=SimpleNamespace(alias_groups=aliases, final_residency=entries),
            schedule=SimpleNamespace(initial_residency=entries),
            timeline=SimpleNamespace(spill_peak_bytes=peak),
        )

    training = plan("weights", "inputs", 32, 128, retained=True)
    evaluation = plan("eval-weights", "inputs", 128, 224, retained=True)
    paths = [("", training), ("evaluation", evaluation)]
    maps = {"": {}, "evaluation": {"eval-weights": "weights"}}
    # 64 shared weights + both private input sets + 32 bytes active scratch.
    assert _spill_bytes(paths, maps) == 256
    # Unretained initial leases may disappear during the peak: don't subtract
    # them just because they are initial. Conservative extra capacity is needed.
    unretained = plan("weights", "inputs", 32, 128, retained=False)
    assert _spill_bytes([("", unretained), paths[1]], maps) == 320
