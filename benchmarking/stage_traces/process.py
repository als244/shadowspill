"""Host-only device selection and bounded subprocess lifetime."""

import datetime as dt
import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path

from benchmarking._serialization import atomic_json


def stamp():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def visible_devices(selection):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if selection != ["all"]:
        if visible is None:
            return selection
        available = visible.split(",")
        return [available[int(value)] for value in selection]
    if visible:
        return visible.split(",")
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
    ).split()


def worker_environment(gpu, cpu_threads):
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": str(cpu_threads),
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "MLOPS_QUACK_TUNE_DISTRIBUTED": "0",
        "PYTHONUNBUFFERED": "1",
    }


def signal_group(pid, signal_number=signal.SIGTERM):
    with suppress(ProcessLookupError):
        os.killpg(pid, signal_number)


def launch(command, log_path, *, env, timeout, pid_path=None):
    with Path(log_path).open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            if pid_path is not None:
                atomic_json(
                    pid_path, {"pid": process.pid, "process_group": process.pid}
                )
            return process.wait(timeout=timeout), False
        except BaseException as error:
            signal_group(process.pid)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                signal_group(process.pid, signal.SIGKILL)
                process.wait()
            if not isinstance(error, subprocess.TimeoutExpired):
                raise
            return process.returncode, True
