"""Bounded fresh workers with isolated caches and immutable saved artifacts."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path


def prerequisites():
    """Check optional kernels in a child, leaving the test runner CUDA-free."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.util, json
missing = [name for name in ('mlops', 'quack', 'sonicmoe', 'cutlass')
           if importlib.util.find_spec(name) is None]
if missing:
    reason = 'missing optional dependencies: ' + ', '.join(missing)
    print(json.dumps({'reason': reason}))
else:
    import torch
    if not torch.cuda.is_available():
        print(json.dumps({'reason': 'CUDA is unavailable'}))
    else:
        capability = torch.cuda.get_device_capability()
        from mlops.providers.sonicmoe.moe import CAPABILITIES
        print(json.dumps({'capability': capability, 'reason': None if capability[0] in
            CAPABILITIES else 'SonicMoE dispatch supports compute capability 9.x and 10.x'}))
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(result.stdout)


def _digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run_phase(directory, phase, *, timeout, cpu=False):
    cache = directory / (phase + "-cache")
    cache.mkdir()
    environment = dict(os.environ)
    if cpu:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    environment.update(
        {
            name: str(cache / child)
            for name, child in (
                ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                ("TRITON_CACHE_DIR", "triton"),
                ("QUACK_CACHE_DIR", "quack"),
                ("CUTE_DSL_CACHE_DIR", "cute"),
                ("CUDA_CACHE_PATH", "cuda"),
                ("XDG_CACHE_HOME", "xdg"),
            )
        }
    )
    if phase == "build" and os.environ.get("SHADOWSPILL_SONIC_BUILD_CACHE"):
        # An explicitly supplied ordinary Quack build cache is optional. Cold
        # profile/execute/resume still start with unrelated empty caches and
        # reject all external compilation/tuning. Never reuse a saved Program.
        environment["QUACK_CACHE_DIR"] = os.environ["SHADOWSPILL_SONIC_BUILD_CACHE"]
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    assert not tuple(cache.iterdir())
    command = [
        sys.executable,
        "-u",
        "-m",
        "tests.workloads.prepared_sonic.worker",
        phase,
        str(directory),
    ]
    started = time.monotonic()
    row = dict(
        phase=phase,
        command=command,
        timeout_seconds=timeout,
        cpu=cpu,
        empty_phase_cache=True,
        quack_cache=environment["QUACK_CACHE_DIR"],
    )
    with (directory / (phase + ".log")).open("x") as output:
        child = subprocess.Popen(
            command,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            status = child.wait(timeout=timeout)
        except BaseException:
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            row.update(
                status="INTERRUPTED",
                returncode=child.returncode,
                seconds=time.monotonic() - started,
            )
            (directory / (phase + ".json")).write_text(json.dumps(row, indent=2) + "\n")
            raise
    row.update(
        status="PASS" if status == 0 else "FAIL",
        returncode=status,
        seconds=time.monotonic() - started,
    )
    (directory / (phase + ".json")).write_text(json.dumps(row, indent=2) + "\n")
    if status:
        tail = (
            (directory / (phase + ".log"))
            .read_text(errors="replace")
            .splitlines()[-50:]
        )
        raise AssertionError(
            f"{phase} failed ({status}); {directory / (phase + '.log')}\n"
            + "\n".join(tail)
        )
    return row


def run_case(directory, case):
    directory = Path(directory).resolve()
    directory.mkdir()
    (directory / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    artifacts, rows = {}, []
    for phase, timeout, cpu in (
        ("prepare", 60, True),
        ("build", 600, False),
        ("profile", 180, False),
        ("plan", 60, True),
        ("execute", 180, False),
        ("resume", 180, False),
        ("inspect", 60, True),
    ):
        assert all(
            _digest(directory / name) == wanted for name, wanted in artifacts.items()
        )
        rows.append(run_phase(directory, phase, timeout=timeout, cpu=cpu))
        assert all(
            _digest(directory / name) == wanted for name, wanted in artifacts.items()
        )
        for name in (
            "reference.pt",
            "program.ssp",
            "profiles.ssp",
            "profiled.ssp",
            "communication.ssp",
            "plan.ssp",
        ):
            if (directory / name).is_file():
                artifacts[name] = _digest(directory / name)
    (directory / "complete.json").write_text(
        json.dumps(
            dict(
                status="PASS_PREPARED_SONIC_SAVED_PROGRAM",
                case=case,
                phases=rows,
                artifacts=artifacts,
                scope=(
                    "Small BF16 actual-provider fixture; "
                    "not full-model or distributed qualification."
                ),
            ),
            indent=2,
        )
        + "\n"
    )
