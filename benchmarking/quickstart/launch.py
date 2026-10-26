"""Launch a phase with the worker count declared by the saved Programs."""

import os
import subprocess

from benchmarking.processes import worker_command

from .phases.common import read


def launch(phase, request_file, selection=None):
    request = read(request_file)
    ranks = 1
    if phase == "execute" or (
        phase == "communicate" and not request.get("communication_info")
    ):
        index = read(request_file.parent / "programs/index.json")
        ranks = index.get("world_size", 1)
    if ranks > 1 and request["device"] != "cuda:0":
        raise ValueError(
            "distributed quickstart builds for cuda:0; "
            "select GPUs with CUDA_VISIBLE_DEVICES"
        )
    suffix = "" if selection is None else f"-{selection:04d}"
    log_path = request_file.parent / "logs" / (phase + suffix + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    arguments = [phase, str(request_file)]
    if selection is not None:
        arguments += ["--selection", str(selection)]
    command = worker_command(
        "benchmarking.quickstart.phases.worker", arguments, ranks=ranks
    )
    environment = dict(os.environ)
    if phase == "plan":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    if ranks > 1:
        environment.setdefault("OMP_NUM_THREADS", "1")
    print(f"{phase}{suffix}: starting; log {log_path}", flush=True)
    with (
        log_path.open("w") as log,
        subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as worker,
    ):
        try:
            for line in worker.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = worker.wait()
        except BaseException:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
            raise
    if code:
        raise RuntimeError(f"quickstart {phase} failed ({code}); see {log_path}")
