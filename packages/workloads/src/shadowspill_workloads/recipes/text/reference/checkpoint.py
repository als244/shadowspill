"""Atomic PyTorch reference checkpoints and resumable packing/schedule state."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

from shadowspill.pytorch._rng import restore_rng, rng_state
from shadowspill.training import checkpoints


def save(path, engine, source, step, elapsed, options):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    temporary = Path(tempfile.mkdtemp(prefix="." + path.name + "-", dir=path.parent))
    try:
        engine.synchronize()
        checkpoints.save(
            temporary / "state.pt",
            engine.model,
            engine.optimizer,
            step,
            engine.weights.masters,
            weights=options.get("checkpoint_weights", "master"),
            include_optimizer=options.get("checkpoint_include_optimizer", True),
        )
        torch.save(
            {
                "step": step,
                "elapsed_seconds": elapsed,
                "selected_candidate": source.name,
                "schedules": {
                    name: schedule.state_dict()
                    for name, schedule in engine.schedules.items()
                    if callable(getattr(schedule, "state_dict", None))
                },
                "source": source.state_dict(),
                "rng": rng_state(engine.device),
            },
            temporary / "loop.pt",
        )
        (temporary / "manifest.json").write_text(
            json.dumps({"version": 1, "step": step})
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load(path, engine, source):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("version") != 1 or "members" in manifest:
        raise ValueError("reference requires a local PyTorch training checkpoint")
    state = torch.load(
        path / "state.pt", map_location="cpu", mmap=True, weights_only=True
    )
    loop = torch.load(path / "loop.pt", map_location="cpu", weights_only=False)
    if state["step"] != loop["step"] or loop["step"] != manifest["step"]:
        raise ValueError("checkpoint model and loop update counts differ")
    if loop["selected_candidate"] != source.name:
        raise ValueError("checkpoint microbatch candidate differs")
    for name, values in loop["schedules"].items():
        restore = getattr(engine.schedules.get(name), "load_state_dict", None)
        if not callable(restore):
            raise ValueError(f"schedule {name!r} cannot restore saved state")
        restore(values)
    checkpoints.load(state, engine.model, engine.optimizer, engine.weights.masters)
    source.load_state_dict(loop["source"])
    restore_rng(loop["rng"], engine.device)
    return loop["step"], loop["elapsed_seconds"]


def prune(directory, keep_last):
    saved = sorted(
        path
        for path in directory.glob("step_*")
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    for path in saved[:-keep_last]:
        shutil.rmtree(path)
