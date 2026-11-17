"""Public exported-weights startup with independent fresh-optimizer trajectories."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from workload_tests.service.test_llama_service import options


def initialized_options(policy):
    value = options(policy)
    source = value["oracle"][1]["weights"]
    fresh = options(policy, initial_weights=source)
    value["weights_initialization"] = fresh
    value["exact_resume"] = True
    value["model_description"] = "exported packed Llama weights into fresh training" + (
        ", LoRA policy" if policy else ", full CE"
    )
    return value


@pytest.mark.parametrize("policy", [False, True])
def test_weights_startup_oracle_keeps_exact_masters_and_uses_a_fresh_optimizer(policy):
    value = initialized_options(policy)
    fresh = value["weights_initialization"]
    assert fresh["oracle"][0]["weights"] == value["oracle"][1]["weights"]
    assert any(
        fresh["oracle"][1]["weights"][name] != value["oracle"][2]["weights"][name]
        for name in value["trainable"]
    )
    assert json.loads(json.dumps(value)) == value


@pytest.mark.cuda
@pytest.mark.parametrize("policy", [False, True])
def test_public_exported_weights_step_split_fresh_optimizer_and_checkpoint_resume(
    tmp_path, policy
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    (tmp_path / "setup.json").write_text(json.dumps(initialized_options(policy)))
    server = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "helpers/_catalog_service_server.py"),
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        timeout=900,
    )
    (tmp_path / "server.stdout").write_text(server.stdout)
    (tmp_path / "server.stderr").write_text(server.stderr)
    assert server.returncode == 0, server.stdout + server.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert (
        receipt["all_client_claims_released"] and receipt["all_server_claims_released"]
    )
    assert receipt["whole_split_equivalence"] and receipt["evaluation_isolation"]
    assert receipt["weights_initialization"] and receipt["fresh_saved_plans"]
    assert receipt["adamw_resume"] and receipt["exact_resume"]
    assert len(receipt["runs"]) == 4
