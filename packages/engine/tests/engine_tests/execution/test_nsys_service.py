"""Real native report inspection after public whole and split optimizer updates."""

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from shutil import which

import pytest
import torch

from engine_tests.execution.test_rng_service import configured
from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.training.components import InitializationRuntime
from shadowspill.training.worker.initialization import prepare_initialization


@pytest.mark.cuda
def test_public_nsys_report_contains_only_selected_complete_updates(
    tmp_path, monkeypatch
):
    nsys = which("nsys")
    if nsys is None:
        pytest.skip("requires an installed Nsight collector")
    setup = configured(tmp_path, monkeypatch)
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
            str(Path(__file__).with_name("_nsys_service_server.py")),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=660,
    )
    (tmp_path / "server.stdout").write_text(server.stdout)
    (tmp_path / "server.stderr").write_text(server.stderr)
    assert server.returncode == 0, server.stdout + server.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert (
        receipt["all_client_claims_released"] and receipt["all_server_claims_released"]
    )
    assert receipt["exact_training_and_eval_rng"]
    assert len(receipt["runs"]) == 2
    traced = receipt["runs"][1]
    report = tmp_path / "rank-0.nsys-rep"
    with report.open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest()
            == traced["manifest"]["report"]["sha256"]
        )
    exported = subprocess.run(
        [
            nsys,
            "export",
            "--type=sqlite",
            "--output",
            str(tmp_path / "rank-0.sqlite"),
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (tmp_path / "export.stdout").write_text(exported.stdout)
    (tmp_path / "export.stderr").write_text(exported.stderr)
    assert exported.returncode == 0, exported.stdout + exported.stderr
    with sqlite3.connect(tmp_path / "rank-0.sqlite") as database:
        kernels = database.execute(
            "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0]
        rows = database.execute(
            'SELECT coalesce(n.text, s.value), n.start, n."end" '
            "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
            "WHERE coalesce(n.text, s.value) LIKE 'train/update=%' ORDER BY n.start"
        ).fetchall()
    assert kernels > 0
    assert [row[0] for row in rows] == traced["expected_labels"]
    assert all(start < end for _, start, end in rows)
    (tmp_path / "report-inspection.json").write_text(
        json.dumps(
            {
                "kernels": kernels,
                "request_ranges": rows,
                "usable_native_report": True,
                "complete_whole_and_split_window": True,
                "unselected_updates_absent": True,
            },
            indent=2,
        )
        + "\n"
    )
