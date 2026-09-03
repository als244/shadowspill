from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from shadowspill.errors import (
    PlanningError,
)
from shadowspill.pytorch import (
    ObjectConsistency,
    SharedInput,
    SharedOutput,
    TensorRef,
    shared_input,
    shared_output,
)
from shadowspill.pytorch.sharing import (
    StateRef,
    format_path,
    resolve_path,
    resolve_shared_inputs,
    resolve_shared_outputs,
)
from shadowspill.runtime import ObjectRef


class _Owner:
    def __init__(self) -> None:
        self.released: list[ObjectRef] = []

    def _release_object_reference(self, reference: ObjectRef) -> None:
        self.released.append(reference)


def _object(size_bytes: int = 64) -> ObjectRef:
    return ObjectRef(_Owner(), object_id=7, size_bytes=size_bytes, handle=11)


def test_tensor_reference_captures_only_view_metadata() -> None:
    tensor = torch.empty_strided((2, 3), (4, 1), dtype=torch.float32)
    reference = TensorRef.from_tensor(_object(32), tensor)

    assert reference.dtype is torch.float32
    assert reference.shape == (2, 3)
    assert reference.stride == (4, 1)
    assert reference.storage_offset == 0


def test_tensor_reference_rejects_view_outside_object() -> None:
    with pytest.raises(ValueError, match="tensor view exceeds"):
        TensorRef(
            object=_object(8),
            dtype=torch.float32,
            shape=(3,),
            stride=(1,),
        )


def test_state_reference_is_immutable_mapping() -> None:
    tensor = TensorRef(
        object=_object(),
        dtype=torch.float32,
        shape=(4,),
        stride=(1,),
    )
    source = {"key": tensor}
    state = StateRef(source)
    source.clear()

    assert tuple(state) == ("key",)
    assert state["key"] is tensor


def test_tensor_reference_close_releases_its_runtime_root_once() -> None:
    owner = _Owner()
    object_reference = ObjectRef(owner, object_id=9, size_bytes=16, handle=13)
    tensor = TensorRef(
        object=object_reference,
        dtype=torch.float32,
        shape=(4,),
        stride=(1,),
    )

    tensor.close()
    tensor.close()

    assert tensor.closed
    assert owner.released == [object_reference]


def test_state_reference_closes_aliased_storage_once() -> None:
    owner = _Owner()
    object_reference = ObjectRef(owner, object_id=9, size_bytes=16, handle=13)
    first = TensorRef(
        object=object_reference,
        dtype=torch.float32,
        shape=(4,),
        stride=(1,),
    )
    second = TensorRef(
        object=object_reference,
        dtype=torch.float32,
        shape=(2,),
        stride=(1,),
        storage_offset=2,
    )

    with StateRef({"first": first, "second": second}) as state:
        assert not state.closed

    assert state.closed
    assert owner.released == [object_reference]


def test_shared_output_normalizes_one_or_many_retained_pools() -> None:
    single = SharedOutput(path=("cache", 0), retain_in="execution")
    multiple = shared_output("cache", 0, retain_in=("execution", "spill"))

    assert single.retain_in == ("execution",)
    assert multiple.retain_in == ("execution", "spill")
    with pytest.raises(ValueError, match="unique"):
        SharedOutput(path=(), retain_in=("execution", "execution"))


def test_shared_input_defaults_to_causal_consistency() -> None:
    tensor = TensorRef(
        object=_object(),
        dtype=torch.float16,
        shape=(8,),
        stride=(1,),
    )

    declaration = shared_input(tensor, require_in="execution")
    unordered = SharedInput(
        tensor,
        require_in="execution",
        consistency=ObjectConsistency.UNORDERED,
    )

    assert declaration.consistency is ObjectConsistency.CAUSAL
    assert unordered.consistency is ObjectConsistency.UNORDERED


def test_shared_inputs_preserve_alias_geometry_and_use_one_cpu_owner() -> None:
    owner = _Owner()
    root = ObjectRef(owner, object_id=9, size_bytes=32, handle=13)
    first = TensorRef(
        object=root,
        dtype=torch.float32,
        shape=(4,),
        stride=(1,),
        retained_pools=("execution",),
    )
    second = TensorRef(
        object=root,
        dtype=torch.float32,
        shape=(2,),
        stride=(1,),
        storage_offset=2,
        retained_pools=("execution",),
    )

    values, resolved = resolve_shared_inputs(
        [
            shared_input(first, require_in="execution"),
            shared_input(second, require_in="execution"),
        ],
        pool_names=("execution", "spill"),
        runtime=owner,
    )

    assert isinstance(values, list)
    assert values[0].untyped_storage()._cdata == values[1].untyped_storage()._cdata
    assert values[1].storage_offset() == 2
    assert tuple(item.public_leaf_index for item in resolved) == (0, 1)


def test_shared_input_requires_guaranteed_pool_and_authentic_control_value() -> None:
    owner = _Owner()
    root = ObjectRef(owner, object_id=9, size_bytes=8, handle=13)
    floating = TensorRef(
        object=root,
        dtype=torch.float32,
        shape=(2,),
        stride=(1,),
        retained_pools=("spill",),
    )
    with pytest.raises(PlanningError, match="does not guarantee"):
        resolve_shared_inputs(
            [shared_input(floating, require_in="execution")],
            pool_names=("execution", "spill"),
            runtime=owner,
        )

    control = TensorRef(
        object=root,
        dtype=torch.int32,
        shape=(2,),
        stride=(1,),
        retained_pools=("spill",),
    )
    with pytest.raises(PlanningError, match="provide profiling_value"):
        resolve_shared_inputs(
            [shared_input(control, require_in="spill")],
            pool_names=("execution", "spill"),
            runtime=owner,
        )
    values, _ = resolve_shared_inputs(
        [
            shared_input(
                control,
                require_in="spill",
                profiling_value=torch.tensor([3, 5], dtype=torch.int32),
            )
        ],
        pool_names=("execution", "spill"),
        runtime=owner,
    )
    torch.testing.assert_close(values[0], torch.tensor([3, 5], dtype=torch.int32))


@dataclass
class _Output:
    cache: tuple[torch.Tensor, ...]


def test_pytree_path_resolution_has_stable_diagnostics() -> None:
    tensor = torch.ones(1)
    output = {"result": _Output(cache=(tensor,))}
    path = ("result", "cache", 0)

    assert resolve_path(output, path) is tensor
    assert format_path(path) == "$.result.cache[0]"
    with pytest.raises(KeyError, match=r"\$\.result\.missing"):
        resolve_path(output, ("result", "missing"))


def test_shared_outputs_resolve_repeated_tensor_leaves_by_path() -> None:
    repeated = torch.ones(2)
    output = {"values": (repeated, repeated), "count": 2}

    resolved = resolve_shared_outputs(
        output,
        (
            shared_output("values", 1, retain_in="spill"),
            shared_output("values", 0, retain_in="execution"),
        ),
        pool_names=("execution", "spill"),
    )

    assert tuple(item.public_leaf_index for item in resolved) == (0, 1)
    assert resolved[0].path == ("values", 0)
    assert resolved[1].path == ("values", 1)


def test_shared_output_resolution_rejects_nonleaf_and_unknown_pool() -> None:
    output = {"values": (torch.ones(1),)}

    with pytest.raises(ValueError, match="must identify one public pytree leaf"):
        resolve_shared_outputs(
            output,
            (shared_output("values", retain_in="spill"),),
            pool_names=("execution", "spill"),
        )
    with pytest.raises(ValueError, match="unknown runtime pools"):
        resolve_shared_outputs(
            output,
            (shared_output("values", 0, retain_in="archive"),),
            pool_names=("execution", "spill"),
        )
