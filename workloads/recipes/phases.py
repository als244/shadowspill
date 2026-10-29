"""Small subprocess driver and saved settings for standalone recipe phases."""

import json
import os
import subprocess
from pathlib import Path

from benchmarking.processes import worker_command


def settings(directory):
    root = Path(directory)
    return root, json.loads((root / "config.json").read_text())


def run_phases(package, root, selected, *, ranks=1):
    phases = ["build", "profile"]
    if ranks > 1:
        phases.append("communicate")
    for phase in [*phases, "plan", "execute"]:
        if selected not in {"all", phase}:
            continue
        log_path = root / f"logs/{phase}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"{phase}: starting; log {log_path}", flush=True)
        environment = dict(os.environ)
        if phase == "plan":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        workers = ranks if phase in {"communicate", "execute"} else 1
        command = worker_command(f"{package}.{phase}", [root], ranks=workers)
        if workers > 1:
            environment.setdefault("OMP_NUM_THREADS", "1")
        with log_path.open("w") as log:
            result = subprocess.run(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            # The phase ran in a worker whose output is its log; name the
            # failure where the caller is looking.
            tail = "".join(log_path.read_text().splitlines(keepends=True)[-20:])
            raise RuntimeError(
                f"{phase} failed with status {result.returncode}; "
                f"the last lines of {log_path}:\n{tail}"
            )
        print(f"{phase}: complete", flush=True)
