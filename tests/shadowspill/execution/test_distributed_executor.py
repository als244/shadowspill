"""Public phase pipeline with distinct NCCL devices and fresh execution workers."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["forward", "forward_backward", "step"])
def test_distributed_saved_executor(tmp_path, kind):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two distinct GPUs")
    directory = Path(__file__).parent

    def run(phase, distributed=False, *extra):
        command = [sys.executable, "-S"]
        if distributed:
            command += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "--start-method=spawn",
                "--no-python",
                sys.executable,
                "-S",
            ]
        script = (
            "_distributed_execute.py"
            if phase == "execute"
            else "_distributed_phases.py"
        )
        command += [str(directory / script)]
        if phase != "execute":
            command += [phase]
        command += [str(tmp_path), *extra]
        environment = {**os.environ, "OMP_NUM_THREADS": "1"}
        with (
            (tmp_path / f"{phase}.log").open("w") as log,
            subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            ) as process,
        ):
            try:
                status = process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                pytest.fail(f"{kind}: {phase} timed out; see {log.name}")
        assert status == 0, (tmp_path / f"{phase}.log").read_text()
        print(f"{kind}: {phase} passed", flush=True)

    run("build", False, kind)
    run("profile")
    run("communicate", True)
    run("plan")
    run("execute", True)
