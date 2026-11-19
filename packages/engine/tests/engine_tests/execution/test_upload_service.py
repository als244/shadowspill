"""Checkpoint round trip through a laptop and fresh GPU service storage."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from engine_tests.execution.test_rng_service import configured
from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.training.definitions import InitializationRuntime
from shadowspill.training.worker.initialization import prepare_initialization


@pytest.mark.cuda
@pytest.mark.parametrize(
    "frozen",
    [
        pytest.param(False, id="full", marks=pytest.mark.slow),
        pytest.param(True, id="frozen"),
    ],
)
def test_public_uploaded_checkpoint_continues_step_and_split_training(
    tmp_path, monkeypatch, frozen
):
    setup = configured(tmp_path, monkeypatch, trainable=["head.*"] if frozen else None)
    model = setup.model.create()
    with (
        prepare_initialization(setup)[0].description.open(
            InitializationRuntime(
                tmp_path, lambda _: pytest.fail("unexpected weight artifact")
            )
        ) as fill,
        torch.no_grad(),
    ):
        for name, value in model.named_parameters():
            fill(
                TensorChunk(name, tuple(value.shape), value.dtype, 0, value.numel()),
                value.reshape(-1),
            )
    (tmp_path / "setup.json").write_text(
        json.dumps(
            {
                "config": setup.config.to_dict(),
                "initial": {
                    name: value.detach().tolist()
                    for name, value in model.named_parameters()
                },
            }
        )
    )
    server = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_upload_service_server.py")),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=780,
    )
    (tmp_path / "server.stdout").write_text(server.stdout)
    (tmp_path / "server.stderr").write_text(server.stderr)
    assert server.returncode == 0, server.stdout + server.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert (
        receipt["all_server_claims_released"]
        and receipt["target_storage_started_empty"]
    )
    assert all(row["independent_step_and_split_oracle"] for row in receipt["runs"])
    assert receipt["runs"][1]["exact_training_and_evaluation_rng"]
    assert receipt["runs"][0]["checkpoint"] == receipt["runs"][1]["checkpoint"]
    assert all(row["frozen_base_reconstructed"] == frozen for row in receipt["runs"])
    assert all(
        row["exported_inference_matches_independent_oracle"] for row in receipt["runs"]
    )
    assert all(
        [item["representation"] for item in row["inference_exports"]]
        == ["master", "compute"]
        for row in receipt["runs"]
    )
