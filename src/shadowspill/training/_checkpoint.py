"""Atomic training checkpoints, including optional source/schedule progress."""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

import torch

from shadowspill.pytorch.distributed import BoundDistributed

from ._types import StepExecution


def rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        state["numpy"] = numpy.random.get_state()
    if device.type == "cuda" and cast(Callable[[], bool], torch.cuda.is_initialized)():
        state["accelerator"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        import numpy

        numpy.random.set_state(state["numpy"])
    if "accelerator" in state:
        torch.cuda.set_rng_state(state["accelerator"], device)


def save(
    path: str | Path,
    execution: StepExecution,
    loop: Mapping[str, Any],
    *,
    distributed: BoundDistributed | None = None,
    weights: Literal["master", "compute"] = "master",
) -> Path:
    if weights not in {"master", "compute"}:
        raise ValueError("weights must be 'master' or 'compute'")
    if distributed is not None:
        distributed.control.agree("checkpoint/weight_representation", weights)
        from shadowspill.pytorch.distributed._checkpoint import save_collective

        def write_rank(root: Path) -> None:
            execution.synchronize()
            execution.save(root / "state.pt", weights=weights)
            torch.save(loop, root / "loop.pt")

        return save_collective(
            path,
            distributed.control,
            write_rank,
            step=loop["step"],
            layout=distributed.checkpoint_layout(),
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    temporary = Path(tempfile.mkdtemp(prefix="." + path.name + "-", dir=path.parent))
    try:
        execution.synchronize()
        execution.save(temporary / "state.pt", weights=weights)
        torch.save(loop, temporary / "loop.pt")
        (temporary / "manifest.json").write_text(
            json.dumps({"version": 1, "step": loop["step"]}, indent=2) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return path


def load(
    path: str | Path, *, distributed: BoundDistributed | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path)
    if distributed is not None:
        from shadowspill.pytorch.distributed._checkpoint import load_collective

        path, step = load_collective(
            path, distributed.control, layout=distributed.checkpoint_layout()
        )
    else:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("version") != 1:
            raise ValueError("unsupported training checkpoint format")
        if "members" in manifest:
            raise ValueError(
                "distributed checkpoint requires the matching distributed configuration"
            )
        step = manifest["step"]
    state = torch.load(
        path / "state.pt", map_location="cpu", mmap=True, weights_only=True
    )
    # Source and schedule capabilities can return ordinary Python state.
    loop = torch.load(path / "loop.pt", map_location="cpu", weights_only=False)
    if state["step"] != loop["step"] or loop["step"] != step:
        raise ValueError("checkpoint model and loop update counts differ")
    return state, loop
