"""Small subprocess driver and saved settings for standalone recipe phases."""

import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from shadowspill.orchestration.launch.phases import (
    PHASES,
    PhaseInvocation,
    PhaseResult,
)
from shadowspill.orchestration.launch.processes import worker_command
from shadowspill.orchestration.monitoring.lifecycle import Lifecycle


def settings(directory: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(directory)
    return root, json.loads((root / "config.json").read_text())


def _subprocess(invocation: PhaseInvocation) -> PhaseResult:
    with invocation.log_path.open("w") as log:
        result = subprocess.run(
            invocation.command,
            env=invocation.environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return PhaseResult(result.returncode, (invocation.log_path,))


def _tail(path: Path) -> str:
    # Keep failure reporting bounded even for legacy standalone phase logs.
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - (16 << 10)))
        return "".join(
            stream.read(16 << 10).decode(errors="replace").splitlines(True)[-20:]
        )


def run_phases(
    package: str,
    root: Path,
    selected: str,
    *,
    ranks: int = 1,
    preparation_ranks: int = 1,
    phases: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
    launch: Callable[[PhaseInvocation], PhaseResult] | None = None,
) -> None:
    order = PHASES
    sequence = (
        tuple(phases)
        if phases is not None
        else tuple(phase for phase in order if phase != "communicate" or ranks > 1)
    )
    if (
        not sequence
        or len(set(sequence)) != len(sequence)
        or any(phase not in order for phase in sequence)
        or sequence != tuple(sorted(sequence, key=order.index))
        or selected not in {"all", *sequence}
    ):
        raise ValueError(
            "select an ordered, distinct sequence of known training phases"
        )
    root = root.resolve()
    base_environment = dict(os.environ if environment is None else environment)
    launcher = _subprocess if launch is None else launch
    for phase in sequence:
        if selected not in {"all", phase}:
            continue
        log_path = root / f"logs/{phase}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"{phase}: starting", flush=True)
        phase_environment = dict(base_environment)
        if phase == "plan":
            phase_environment["CUDA_VISIBLE_DEVICES"] = ""
        workers = (
            ranks
            if phase in {"communicate", "execute"}
            else preparation_ranks
            if phase in {"build", "profile"}
            else 1
        )
        command = worker_command(f"{package}.{phase}", [root], ranks=workers)
        if workers > 1:
            phase_environment.setdefault("OMP_NUM_THREADS", "1")
        with (
            Lifecycle(root / f"logs/{phase}-launcher.jsonl") as timing,
            timing.phase("workers"),
        ):
            result = launcher(
                PhaseInvocation(
                    phase, tuple(command), phase_environment, workers, root, log_path
                )
            )
            timing.event(
                "workers.returned",
                returncode=result.returncode,
                logs=[str(path) for path in result.logs],
                log_statuses=[
                    {
                        "truncated": status.truncated,
                        "bytes_seen": status.bytes_seen,
                        "bytes_written": status.bytes_written,
                        "error": status.error,
                    }
                    for status in result.log_statuses
                ],
            )
        if result.returncode:
            # The phase ran in a worker whose output is its log; name the
            # failure where the caller is looking.
            tail = "\n".join(f"{path}:\n{_tail(path)}" for path in result.logs)
            raise RuntimeError(
                f"{phase} failed with status {result.returncode}; "
                f"the last log lines:\n{tail}"
            )
        print(f"{phase}: complete; logs {', '.join(map(str, result.logs))}", flush=True)
