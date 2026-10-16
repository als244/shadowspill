"""Separate-device end-to-end coverage, skipped when fewer devices are visible."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import warnings

import pytest
import torch

pytestmark = [pytest.mark.fresh_process, pytest.mark.cuda]


def _nvml_lists_every_gpu():
    """Each rank's runtime identifies its GPU through NVML before allocating.
    One GPU in a failed state makes NVML's listing fail for the whole node."""
    if torch.version.hip:
        return True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.cuda._raw_device_uuid_nvml() is not None


@pytest.mark.parametrize(
    "arguments",
    [
        ["--variant", "save", "--parameter-metrics", "--symmetric-planning"],
        ["--variant", "recompute", "--no-sharded"],
        ["--variant", "save", "--no-sharded", "--optimizer", "matrix"],
        [
            "--variant",
            "recompute",
            "--precision",
            "fp16",
            "--masters",
            "--optimizer",
            "mlops",
            "--diagnostics",
            "--symmetric-planning",
        ],
    ],
)
def test_distributed_training_matches_combined_reference(tmp_path, arguments):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two separate visible devices")
    if not _nvml_lists_every_gpu():
        pytest.skip(
            "a GPU on this node is in a failed state: NVML cannot list the GPUs"
        )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = {
        **os.environ,
        "GLOO_SOCKET_IFNAME": "lo",
        "NCCL_SOCKET_IFNAME": "lo",
        "OMP_NUM_THREADS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            "--nproc-per-node=2",
            "-m",
            "tests.shadowspill.pytorch.distributed._training_case",
            "--out",
            str(tmp_path),
            *arguments,
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=360,
    )
    assert result.returncode == 0, result.stdout
    records = json.loads((tmp_path / "result.json").read_text())
    assert len(records) == 2 and all(item["passed"] for item in records)
    for rank in range(2):
        report = json.loads((tmp_path / f"rank-{rank:05d}" / "plan.json").read_text())
        assert report
