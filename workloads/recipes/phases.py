"""Small subprocess driver and saved settings for standalone recipe phases."""

import json
import os
import subprocess
import sys
from pathlib import Path


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
        python = [sys.executable, *(["-S"] if sys.flags.no_site else [])]
        command = [*python, "-u", "-m", f"{package}.{phase}", str(root)]
        if ranks > 1 and phase in {"communicate", "execute"}:
            environment.setdefault("OMP_NUM_THREADS", "1")
            command = [
                *python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={ranks}",
                "--no-python",
                *python,
                "-u",
                "-m",
                "workloads.recipes.logical_device_worker",
                f"{package}.{phase}",
                str(root),
            ]
        with log_path.open("w") as log:
            subprocess.run(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(f"{phase}: complete", flush=True)
