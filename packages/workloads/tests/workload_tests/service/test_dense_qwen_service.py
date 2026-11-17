"""Dense pretrained public training against independent compiled references."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from workload_tests.service.oracles._dense_qwen_oracle import options


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize(
    "family,lora,override_decay",
    [
        ("qwen3", False, False),
        ("qwen3", True, False),
        ("qwen35", True, False),
        pytest.param("qwen35", True, True, id="qwen35-lora-dtype-override"),
    ],
)
def test_pretrained_dense_step_split_and_resume_matches_compiled_reference(
    tmp_path, monkeypatch, family, lora, override_decay
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    # Match the compiled reference's FP32 products. As in the MoE service gate,
    # FLA's default TF32 recurrence is tested separately from this strict oracle.
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    # Different parameterizations share the reference forward code object. Each
    # has several packed/evaluation layouts; do not count earlier models' graphs
    # against this model's Dynamo specialization limit.
    torch.compiler.reset()
    value = options(tmp_path, family, lora=lora, override_decay=override_decay)
    torch.cuda.empty_cache()
    (tmp_path / "setup.json").write_text(json.dumps(value))
    source = tmp_path / "pretrained"
    original = {path.name: path.read_bytes() for path in source.iterdir()}
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
    assert receipt["adamw_resume"] and receipt["exact_resume"]
    assert receipt["parameter_storage_dtypes"] == value["parameter_storage_dtypes"]
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original
