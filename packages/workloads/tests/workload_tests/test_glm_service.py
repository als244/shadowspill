"""Small pretrained GLM LoRA through the public client and saved-Plan restart."""

import json
import subprocess
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from qualification.numerical.metrics import tensor_metrics
from qualification.numerical.tolerances import (
    LOSS_ABSOLUTE_TOLERANCE,
    LOSS_RELATIVE_TOLERANCE,
    meets_tensor_tolerance,
)
from workload_tests.test_glm53_checkpoint import _write
from workload_tests.test_glm_definitions import mark_family


def compare_state(reference, actual, *, steps, include_gradients=True):
    """Use the existing numerical gate's state bounds, preserving storage dtypes."""
    records = {}
    consumed_gradients = []
    with tarfile.open(reference) as left, tarfile.open(actual) as right:
        expected, observed = (
            json.loads(archive.extractfile("manifest.json").read())
            for archive in (left, right)
        )
        assert observed["state"] == expected["state"]
        entries = {row["name"]: row for row in observed["tensors"]}
        assert set(entries) == {row["name"] for row in expected["tensors"]}
        for before in expected["tensors"]:
            name = before["name"]
            after = entries[name]
            for key in ("dtype", "shape", "stride", "storage_offset"):
                assert before[key] == after[key], (name, key)
            if name.startswith("gradient.") and not include_gradients:
                # Both operations have consumed their gradients. A fused step
                # need not write the persistent accumulation bank used by
                # forward_backward + optim_step. Its inactive bytes are not
                # part of whole/split equivalence; masters and moments are.
                consumed_gradients.append(name)
                continue
            values = [
                torch.frombuffer(
                    bytearray(archive.extractfile(row["member"]).read()),
                    dtype=getattr(torch, row["dtype"].removeprefix("torch.")),
                ).as_strided(row["shape"], row["stride"], row["storage_offset"])
                for archive, row in ((left, before), (right, after))
            ]
            exact = before["blake3"] == after["blake3"]
            if (
                not values[0].is_floating_point()
                or name.startswith("optimizer_group.")
                or name.endswith(".step")
            ):
                records[name] = {"passed": exact, "bitwise_equal": exact}
            else:
                metric = tensor_metrics(*values)
                category = "optimizer" if name.startswith("optimizer.") else "model"
                records[name] = {
                    "passed": meets_tensor_tolerance(
                        metric,
                        key=f"state/{category}/{name}",
                        step_size=0.001,
                        steps=steps,
                    ),
                    "bitwise_equal": exact,
                    **asdict(metric),
                }
    return {"tensors": records, "consumed_gradients": consumed_gradients}


@pytest.mark.cuda
@pytest.mark.slow
def test_glm_checkpoint_step_split_evaluation_and_resume(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    _write(checkpoint, fp8_experts=False)
    mark_family(checkpoint, "glm53_flash")
    original = {path.name: path.read_bytes() for path in checkpoint.iterdir()}
    options = {
        "checkpoint": str(checkpoint),
        "state_directory": str(tmp_path / "states"),
        "client_script": "_glm_service_client.py",
        "client_timeout": 1000,
        "fresh_saved_plans": True,
        "cold_workers": True,
        "model_description": "tiny GLM Flash, BF16 base and LoRA, FP32 state",
        "oracle_description": "whole/split updates and optimizer-state continuation",
    }
    (tmp_path / "setup.json").write_text(json.dumps(options))
    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_catalog_service_server.py")),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=1200,
    )
    (tmp_path / "server.stdout").write_text(child.stdout)
    (tmp_path / "server.stderr").write_text(child.stderr)
    assert child.returncode == 0, child.stdout + child.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert (
        receipt["all_server_claims_released"] and receipt["all_client_claims_released"]
    )
    comparisons = {}
    for before, after, steps in (
        ("whole", "split", 1),
        ("after-four", "resumed-four", 4),
    ):
        comparisons[after] = compare_state(
            tmp_path / "states" / before / "ranks/rank-00000.ssp",
            tmp_path / "states" / after / "ranks/rank-00000.ssp",
            steps=steps,
            include_gradients=after != "split",
        )
    (tmp_path / "state-comparisons.json").write_text(json.dumps(comparisons, indent=2))
    assert not {
        f"{case}/{name}": row
        for case, comparison in comparisons.items()
        for name, row in comparison["tensors"].items()
        if not row["passed"]
    }
    for expected, actual in receipt["loss_comparisons"].values():
        assert actual == pytest.approx(
            expected, rel=LOSS_RELATIVE_TOLERANCE, abs=LOSS_ABSOLUTE_TOLERANCE
        )
    assert receipt["evaluation_isolation"] and receipt["saved_plan_restart"]
    assert {path.name: path.read_bytes() for path in checkpoint.iterdir()} == original
