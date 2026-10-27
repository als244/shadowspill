"""Build freshness and the host environment recorded for each gate."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


def _default_run() -> str:
    """Name a run for the commit it measured."""

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        modified = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return time.strftime("%m%d_%H%M")
    return f"{revision}{'_dirty' if modified else ''}_{time.strftime('%m%d_%H%M')}"


def _host_state() -> str:
    """One line saying what else the host was doing: load and GPU tenants."""
    try:
        one, five, fifteen = os.getloadavg()
        load = f"load {one:.2f} {five:.2f} {fifteen:.2f}"
    except OSError:
        load = "load unavailable"
    if shutil.which("nvidia-smi") is None:
        return (
            f"[{time.strftime('%H:%M:%S')}] host: {load}; "
            "gpu tenants unknown (no nvidia-smi)"
        )
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        tenants = [line.strip() for line in query.stdout.splitlines() if line.strip()]
        gpu = f"gpu tenants {'; '.join(tenants) if tenants else 'none'}"
    except Exception:
        gpu = "gpu tenants unavailable"
    return f"[{time.strftime('%H:%M:%S')}] host: {load}; {gpu}"


def _newest(root: Path, suffixes: set[str] | None = None) -> float:
    return max(
        (
            path.stat().st_mtime
            for path in root.rglob("*")
            if path.is_file() and (suffixes is None or path.suffix in suffixes)
        ),
        default=0.0,
    )


def _refuse_a_stale_library() -> str | None:
    """Why the gates must not run, if the build is older than its sources."""

    try:
        from shadowspill.libraries import resolve_library
    except Exception:  # pragma: no cover - the suite gate reports this better
        return None
    try:
        library = resolve_library("libshadowspill.so")
    except Exception:
        return None
    sources = Path("csrc")
    if library is None or not sources.is_dir() or not library.is_file():
        return None
    built_from = {".c", ".h", ".cpp", ".hpp", ".txt", ".cmake"}
    if _newest(library.parent) >= _newest(sources, built_from):
        return None
    return (
        f"the build the gates will load is older than csrc/:\n"
        f"    {library.parent}\n"
        f"Rebuild the tree the editable install reads, not build/dev:\n"
        f"    cmake --build {library.parent} --parallel\n"
        f"Pass --allow-stale-library to measure the build as it stands."
    )


def _cpu_environment(cpu_threads: int | None = None) -> tuple[int, dict[str, str]]:
    """Use the process's available CPUs, even if its shell limits OpenMP."""
    if cpu_threads is None:
        try:
            cpu_threads = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cpu_threads = os.cpu_count() or 1
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = str(cpu_threads)
    return cpu_threads, environment
