from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from shadowspill.pytorch import InputGuardError, PlanningError, TensorSpec
from shadowspill.pytorch.guards import (
    capture_input_signature,
    capture_training_signatures,
    validate_training_inputs,
)


def test_position_specific_training_guards_preserve_static_metadata() -> None:
    examples = [
        [TensorSpec((2, 4), torch.float32), {"sequence_length": 4}],
        [TensorSpec((3, 7), torch.float32), {"sequence_length": 7}],
    ]
    signatures = capture_training_signatures(examples)
    runtime = [
        [torch.zeros(2, 4), {"sequence_length": 4}],
        [torch.zeros(3, 7), {"sequence_length": 7}],
    ]
    validate_training_inputs(runtime, signatures)
    assert signatures[0].digest != signatures[1].digest

    runtime[1][1]["sequence_length"] = 6
    try:
        validate_training_inputs(runtime, signatures)
    except InputGuardError as error:
        assert "microbatch 1" in str(error)
        assert "static value" in str(error)
    else:
        raise AssertionError("changed metadata escaped its guard")


def test_tensor_guard_rejects_geometry_before_execution() -> None:
    signature = capture_input_signature([torch.zeros(2, 3).t()])
    signature.validate([torch.zeros(2, 3).t()])
    try:
        signature.validate([torch.zeros(3, 2)])
    except InputGuardError as error:
        assert "stride" in str(error)
    else:
        raise AssertionError("changed stride escaped its guard")


def test_tensor_guard_preserves_storage_extent_offset_and_aliases() -> None:
    base = torch.arange(12, dtype=torch.float32)
    signature = capture_input_signature([base[:8], base[2:10]])
    runtime = torch.zeros(12)
    signature.validate([runtime[:8], runtime[2:10]])

    separate = torch.zeros(12)
    with pytest.raises(InputGuardError, match="alias relationship"):
        signature.validate([runtime[:8], separate[2:10]])
    with pytest.raises(InputGuardError, match="storage_nbytes"):
        capture_input_signature([torch.zeros(8)]).validate([runtime[:8]])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TensorSpec((-1,), torch.float32),
        lambda: TensorSpec((1,), "float32"),
        lambda: TensorSpec((1,), torch.float32, layout=torch.sparse_coo),
        lambda: TensorSpec((2, 3), torch.float32, stride=(1,)),
    ],
)
def test_tensor_spec_rejects_invalid_geometry(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


def test_tensor_spec_reports_minimum_strided_storage_extent() -> None:
    assert TensorSpec((3, 2), torch.bfloat16, stride=(1, 3)).storage_nbytes == 12
    assert TensorSpec((0, 4), torch.float32).storage_nbytes == 0
    assert TensorSpec((), torch.float32).storage_nbytes == 4


def test_guard_reports_structure_type_count_and_static_errors() -> None:
    signature = capture_input_signature([torch.zeros(2), {"mode": "x"}])
    with pytest.raises(InputGuardError, match="structure"):
        signature.validate((torch.zeros(2), {"mode": "x"}))
    with pytest.raises(InputGuardError, match="must be a tensor"):
        signature.validate([3, {"mode": "x"}])
    with pytest.raises(InputGuardError, match="static value"):
        signature.validate([torch.zeros(2), {"mode": "y"}])

    signatures = (signature,)
    with pytest.raises(InputGuardError, match="outer"):
        validate_training_inputs(3, signatures)  # type: ignore[arg-type]
    with pytest.raises(InputGuardError, match="count"):
        validate_training_inputs([], signatures)
    with pytest.raises(InputGuardError, match="list or tuple"):
        validate_training_inputs([3], signatures)  # type: ignore[list-item]


@dataclass
class _Uncopyable:
    def __deepcopy__(self, memo: object) -> object:
        raise RuntimeError("cannot copy")


class _UnstableEquality:
    def __deepcopy__(self, memo: object) -> _UnstableEquality:
        return _UnstableEquality()

    def __eq__(self, other: object) -> bool:
        return False


def test_capture_rejects_invalid_templates() -> None:
    with pytest.raises(PlanningError, match="non-empty"):
        capture_training_signatures([])
    with pytest.raises(PlanningError, match="list or tuple"):
        capture_training_signatures([3])  # type: ignore[list-item]
    with pytest.raises(PlanningError, match="strided"):
        capture_input_signature([torch.sparse_coo_tensor([[0]], [1.0], (2,))])
    with pytest.raises(PlanningError, match="cannot be preserved"):
        capture_input_signature([_Uncopyable()])
    with pytest.raises(PlanningError, match="stable equality"):
        capture_input_signature([_UnstableEquality()])
