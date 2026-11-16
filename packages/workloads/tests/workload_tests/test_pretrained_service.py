"""Real HTTP pretrained loading plus the public NumPy client training oracle."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from shadowspill_client import Model
from shadowspill_workloads.models.llama3.config import Config
from shadowspill_workloads.models.llama3.implementations.pytorch import (
    Model as Reference,
)
from workload_tests._hub_server import serve_hub
from workload_tests.test_llama_service import options
from workload_tests.test_pretrained_initialization import checkpoint
from workload_tests.test_pretrained_metadata import REVISION


@pytest.mark.cuda
@pytest.mark.parametrize("policy", [False, True])
def test_public_pretrained_step_split_and_checkpoint_resume(
    tmp_path, monkeypatch, policy
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    # Referenced initialization uses seed zero for fresh LoRA factors,
    # independently of checkpoint chunking/rank ownership.
    value = options(policy, seed=0)
    requested = Model.from_dict(value["model"])
    base = Reference(Config(**requested.config["architecture"])).bfloat16()
    with torch.no_grad():
        for name, parameter in base.named_parameters():
            parameter.copy_(torch.tensor(value["oracle"][0]["weights"][name]))
    source = tmp_path / "hub-source"
    source.mkdir()
    checkpoint(source, "llama3", base)
    value["model"] = Model.from_pretrained(
        "fixture/llama",
        revision=REVISION,
        lora=requested.lora,
        precision=requested.precision,
        config=requested.config,
    ).to_dict()
    value["exact_resume"] = True
    value["model_description"] = "pretrained packed Llama over bounded HTTP" + (
        ", LoRA policy" if policy else ", full CE"
    )
    (tmp_path / "setup.json").write_text(json.dumps(value))
    with serve_hub(source) as (endpoint, requests):
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
        monkeypatch.setenv("HF_HOME", str(tmp_path / "isolated-hub-cache"))
        server = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("_catalog_service_server.py")),
                str(tmp_path),
            ],
            text=True,
            capture_output=True,
            timeout=900,
        )
    (tmp_path / "server.stdout").write_text(server.stdout)
    (tmp_path / "server.stderr").write_text(server.stderr)
    (tmp_path / "hub-requests.json").write_text(json.dumps(requests))
    assert server.returncode == 0, server.stdout + server.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert (
        receipt["all_client_claims_released"] and receipt["all_server_claims_released"]
    )
    assert receipt["whole_split_equivalence"] and receipt["evaluation_isolation"]
    assert receipt["adamw_resume"] and receipt["exact_resume"]
    assert any(
        method == "GET" and name.endswith(".safetensors") for method, name in requests
    )
    assert not list((tmp_path / "cache").glob("initialization-*"))
