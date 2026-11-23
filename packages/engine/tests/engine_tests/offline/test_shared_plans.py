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


def test_joint_spill_counts_persistent_state_once_beside_the_largest_transient_peak():
    def plan(model, private, size, peak, *, retained=True, end=0, extra=()):
        aliases = (
            AliasGroupSpec(model, "rank0", 64, retain_spill_copy=retained),
            AliasGroupSpec(private, "rank0", size, retain_spill_copy=True),
            *extra,
        )
        entries = tuple(
            ResidencySpec(a.alias_group_id, MemoryLocation.SPILL) for a in aliases[:2]
        )
        return SimpleNamespace(
            program=SimpleNamespace(alias_groups=aliases, final_residency=entries),
            schedule=SimpleNamespace(initial_residency=entries),
            timeline=SimpleNamespace(
                spill_transient_peak_bytes=peak, spill_transient_end_bytes=end
            ),
        )

    maps = {"": {}, "evaluation": {"eval-weights": ("", "weights")}}
    evaluation = plan("eval-weights", "inputs", 128, 16)
    # 64 shared weights once + both private input sets + the larger
    # transient peak, 32 bytes of training scratch.
    training = plan("weights", "inputs", 32, 32)
    assert _spill_bytes([("", training), ("evaluation", evaluation)], maps) == 256
    # Residency alone makes the weights persistent, retained or not.
    unretained = plan("weights", "inputs", 32, 32, retained=False)
    assert _spill_bytes([("", unretained), ("evaluation", evaluation)], maps) == 256
    # A retained copy outside the residency is persistent too.
    staged = plan(
        "weights",
        "inputs",
        32,
        32,
        extra=(AliasGroupSpec("staging", "rank0", 8, retain_spill_copy=True),),
    )
    assert _spill_bytes([("", staged), ("evaluation", evaluation)], maps) == 264
    # Transient spill a Plan leaves at its end stays held from then on.
    leaving = plan("weights", "inputs", 32, 32, end=16)
    assert _spill_bytes([("", leaving), ("evaluation", evaluation)], maps) == 272
