"""CPU readiness and reference precision for the actual-FP16 distributed fixture."""

import json

import pytest
import torch

from engine_tests.execution.narrow_regions.fixture import CASES
from engine_tests.execution.narrow_regions.test_training import run_phase


@pytest.mark.parametrize("name", CASES)
def test_actual_narrow_region_vjps(tmp_path, name):
    (tmp_path / "case.json").write_text(json.dumps(CASES[name]) + "\n")
    run_phase(tmp_path, "preflight", cpu=True)


@pytest.mark.parametrize("name", CASES)
def test_narrow_reference_preserves_storage_and_accumulation(tmp_path, name):
    case = CASES[name]
    (tmp_path / "case.json").write_text(json.dumps(case) + "\n")
    run_phase(tmp_path, "prepare", cpu=True)
    run_phase(tmp_path, "reference_cpu", cpu=True)
    reference = torch.load(tmp_path / "reference.pt", weights_only=True)
    assert reference["reference_contract"] == "cpu_control"
    assert {v.dtype for v in reference["initial"].values()} == {torch.float16}
    for step, expected in enumerate(reference["expected"]):
        assert all(g.dtype == torch.float16 for g in expected["gradients"].values())
        for name, fresh in expected["new_gradients"].items():
            left, right = expected["rank_gradients"]
            assert torch.equal(fresh, left[name] + right[name])
            value = expected["gradients"][name]
            if expected["accumulate"]:
                assert step == 1 and case["kind"] == "forward_backward"
                assert torch.equal(
                    value, fresh + reference["expected"][0]["gradients"][name]
                )
            else:
                assert torch.equal(value, fresh)
        if case["kind"] == "step":
            masters = [
                v for n, v in expected["state"].items() if n.startswith("master.")
            ]
            assert masters and all(value.dtype == torch.float32 for value in masters)
        assert all(
            torch.equal(expected["state"][name], reference["initial"][name])
            for name in reference["frozen"]
        )
