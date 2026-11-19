"""Real HTTP pretrained loading plus the public NumPy client training oracle."""

import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from shadowspill_client import Model
from shadowspill_workloads.models.llama3.config import Config
from shadowspill_workloads.models.llama3.implementations.pytorch import (
    Model as Reference,
)
from workload_tests.initialization._hub_server import serve_hub
from workload_tests.initialization.test_pretrained_initialization import checkpoint
from workload_tests.initialization.test_pretrained_metadata import REVISION
from workload_tests.service.test_llama_service import options


@pytest.mark.cuda
@pytest.mark.parametrize(
    "source_kind,policy",
    # Every setting's values run in the default suite; the remaining
    # combinations run with `-m slow`.
    [
        ("hub", False),
        pytest.param("hub", True, marks=pytest.mark.slow),
        pytest.param("local", False, marks=pytest.mark.slow),
        ("local", True),
    ],
)
def test_public_pretrained_step_split_and_checkpoint_resume(
    tmp_path, monkeypatch, policy, source_kind
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
        "fixture/llama" if source_kind == "hub" else str(source),
        revision=REVISION if source_kind == "hub" else None,
        lora=requested.lora,
        precision=requested.precision,
        config=requested.config,
    ).to_dict()
    value["exact_resume"] = True
    value["fresh_saved_plans"] = options(policy, seed=29)
    value["model_description"] = f"pretrained packed Llama from {source_kind}" + (
        ", LoRA policy" if policy else ", full CE"
    )
    (tmp_path / "setup.json").write_text(json.dumps(value))
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    connection = serve_hub(source) if source_kind == "hub" else nullcontext((None, []))
    with connection as (endpoint, requests):
        if endpoint is not None:
            monkeypatch.setenv("HF_ENDPOINT", endpoint)
        monkeypatch.setenv("HF_HOME", str(tmp_path / "isolated-hub-cache"))
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
    (tmp_path / "hub-requests.json").write_text(json.dumps(requests))
    assert server.returncode == 0, server.stdout + server.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert (
        receipt["all_client_claims_released"] and receipt["all_server_claims_released"]
    )
    assert receipt["whole_split_equivalence"] and receipt["evaluation_isolation"]
    assert receipt["adamw_resume"] and receipt["exact_resume"]
    assert receipt["fresh_saved_plans"] and len(receipt["runs"]) == 4
    if source_kind == "hub":
        assert any(
            method == "GET" and name.endswith(".safetensors")
            for method, name in requests
        )
    else:
        assert not requests
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original
    assert not list((tmp_path / "cache").glob("initialization-*"))
