"""Small subprocess driver and saved settings for standalone text recipe phases."""

import json
import os
import subprocess
import sys
from pathlib import Path


def settings(directory):
    root = Path(directory)
    return root, json.loads((root / "config.json").read_text())


def run_phases(package, root, selected):
    for phase in ("build", "profile", "plan", "execute"):
        if selected not in {"all", phase}:
            continue
        log_path = root / f"logs/{phase}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"{phase}: starting; log {log_path}", flush=True)
        environment = dict(os.environ)
        if phase == "plan":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        with log_path.open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    *(["-S"] if sys.flags.no_site else []),
                    "-u",
                    "-m",
                    f"{package}.{phase}",
                    str(root),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(f"{phase}: complete", flush=True)
