"""What a row records about where it came from: the sources that ran, the
packages they ran on, the hardware, and the kernels mlops dispatched.

A result that cannot name these cannot be compared with another, so the
sweep refuses to resume a manifest whose sources or environment changed and
every row carries them.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def source_identity(paths: Iterable[str]) -> dict[str, Any]:
    """The git head, whether the tree is dirty, and a digest of every Python
    file under the given repository-relative paths."""

    files: dict[str, str] = {}
    for relative in paths:
        path = ROOT / relative
        candidates = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for candidate in candidates:
            if "__pycache__" in candidate.parts:
                continue
            files[str(candidate.relative_to(ROOT))] = hashlib.sha256(
                candidate.read_bytes()
            ).hexdigest()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True
    )
    status = subprocess.run(
        ["git", "status", "--short", "--", *paths],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return {
        "git_head": head.stdout.strip() or None,
        "git_dirty": bool(status.stdout.strip()),
        "file_sha256": files,
    }


def environment_identity(python: str | Path) -> dict[str, Any]:
    """Probe the interpreter a sweep launches without importing PyTorch or
    initializing a device: its version, every installed package, and a
    digest of the mlops sources it would import."""

    probe = """
import hashlib, importlib.metadata, importlib.util, json, pathlib, sys
packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
spec = importlib.util.find_spec('mlops')
files = {}
if spec is not None and spec.origin:
    root = pathlib.Path(spec.origin).parent
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*.py'))}
print(json.dumps({'python': sys.version, 'packages': packages, 'mlops_source': files}))
"""
    result = subprocess.run(
        [str(python), "-c", probe], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)


def hardware_identity() -> dict[str, Any]:
    """The host and the device a harness process runs on; imports PyTorch,
    so a launcher calls it only in the process that measures."""

    import torch

    device: dict[str, Any] = {"count": 0}
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        device = {
            "name": properties.name,
            "count": torch.cuda.device_count(),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "memory_bytes": properties.total_memory,
            "driver": torch.version.cuda,
        }
    memory_bytes = None
    with contextlib.suppress(ValueError, OSError):
        memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {
        "host": platform.node(),
        "cpus": os.cpu_count(),
        "host_memory_bytes": memory_bytes,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
    }


def dispatch_record(
    trace: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    """The kernels mlops dispatched, by operation, with how often each
    implementation ran: what ran, recorded as counted, so a row whose
    operation exercised two implementations says so rather than failing."""

    return {
        str(operation): {str(name): int(count) for name, count in counts.items()}
        for operation, counts in trace.items()
    }


__all__ = [
    "ROOT",
    "dispatch_record",
    "environment_identity",
    "hardware_identity",
    "source_identity",
]
