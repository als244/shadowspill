from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from benchmarking.rank_device import isolate_device


@pytest.mark.parametrize(
    "mask,rank,expected",
    [
        (None, 3, "3"),
        ("2,4,6,7", 2, "6"),
        ("GPU-first,GPU-second", 1, "GPU-second"),
        ("MIG-GPU-first/1/0", 3, "MIG-GPU-first/1/0"),
    ],
)
def test_device_isolation_preserves_distributed_rank_identity(mask, rank, expected):
    environment = {"RANK": "11", "WORLD_SIZE": "16", "LOCAL_RANK": str(rank)}
    if mask is not None:
        environment["CUDA_VISIBLE_DEVICES"] = mask
    assert isolate_device(environment) == expected
    assert environment["CUDA_VISIBLE_DEVICES"] == expected
    assert environment["RANK"] == "11"
    assert environment["LOCAL_RANK"] == str(rank)
    assert environment["WORLD_SIZE"] == "16"


@pytest.mark.parametrize("mask,rank", [("", 0), ("-1", 0), ("0,,2", 1), ("0,2", 2)])
def test_invalid_gpu_masks_are_rejected(mask, rank):
    with pytest.raises(ValueError):
        isolate_device({"CUDA_VISIBLE_DEVICES": mask, "LOCAL_RANK": str(rank)})


def test_entrypoint_sees_isolation_before_importing_torch(tmp_path):
    script = tmp_path / "entry.py"
    script.write_text(
        "import json, os, sys\n"
        "print(json.dumps([os.environ['CUDA_VISIBLE_DEVICES'], "
        "os.environ['RANK'], 'torch' in sys.modules, sys.argv[1:]]))\n"
    )
    output = subprocess.check_output(
        [sys.executable, "-m", "benchmarking.rank_device", str(script), "--test", "x"],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "2,4,6,7",
            "LOCAL_RANK": "2",
            "RANK": "10",
        },
        text=True,
    )
    assert json.loads(output) == ["6", "10", False, ["--test", "x"]]
