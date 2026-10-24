"""Small process boundary: inherit the interpreter, record phase time, fail loudly."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def python_command(module: str, *arguments: object) -> list[str]:
    return [
        sys.executable,
        *(["-S"] if sys.flags.no_site else []),
        "-u",
        "-m",
        module,
        *(str(argument) for argument in arguments),
    ]


def run_phase(
    directory: Path, phase: str, module: str, *arguments: object, cpu: bool = False
) -> None:
    print(f"{phase}: starting", flush=True)
    environment = dict(os.environ)
    if cpu:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    started = time.perf_counter()
    status = "failed"
    try:
        subprocess.run(python_command(module, *arguments), env=environment, check=True)
        status = "completed"
    finally:
        path = directory / "phases.json"
        phases = json.loads(path.read_text()) if path.exists() else {}
        phases[phase] = time.perf_counter() - started
        path.write_text(json.dumps(phases, indent=2) + "\n")
        print(f"{phase}: {status} in {phases[phase]:.3f}s", flush=True)


def profile_and_plan(directory: Path) -> None:
    run_phase(directory, "profile", "qualification.pipeline.profile", directory)
    run_phase(directory, "plan", "qualification.pipeline.plan", directory, cpu=True)
