"""Bounded preservation of mutable buffers, controls and RNG during diagnostics."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryFile
from typing import TYPE_CHECKING, Any

import torch

from shadowspill.pytorch._rng import preserve_rng

if TYPE_CHECKING:
    from shadowspill.execution import Executor
    from shadowspill.execution.paths import ExecutionPath


def learning_rates(path: ExecutionPath) -> tuple[str, ...]:
    bindings = path.plan.program.bindings
    if bindings.get("optimizer_zero_lr_preserves_state") is not True:
        raise ValueError(
            "startup diagnostics require zero_lr_preserves_state=True on the "
            "optimizer used by build_step; ordinary AdamW changes moments at zero LR"
        )
    rates = tuple(bindings.get("optimizer_learning_rates", ()))
    if not rates or any(
        name not in path.controls.host
        or path.controls.bindings[name]["role"] != "hyperparameter"
        or path.controls.host[name].numel() != 1
        for name in rates
    ):
        raise ValueError(
            "startup diagnostics require captured host-tensor learning rates"
        )
    return rates


def buffer_aliases(path: ExecutionPath, controls: Mapping[str, Any]) -> set[str]:
    storage = path.storage
    # Publishing unchanged optimizer-owned shards only restores their model
    # representation. It is covered by the optimizer's explicit zero-LR promise.
    publications = {
        (item["profile_key"], item["destination"])
        for component in path.plan.program.bindings.get("optimizer_templates", ())
        for item in component.get("publications", ())
    }
    mutated = {
        storage.aliases[mutation.object_id]
        for bound in path.tasks
        if bound.task.phase != "optimizer"
        for mutation in bound.task.mutations
        if not (
            bound.task.phase == "weight_communication"
            and (bound.task.profile_id, mutation.object_id) in publications
        )
    }
    registrations = path.plan.program.bindings["registrations"]
    parameters = {
        storage.aliases[item["object_id"]]
        for item in registrations
        if item["parameter"]
    }
    if mutated & parameters:
        raise ValueError(
            "startup diagnostics cannot preserve parameters mutated "
            "outside optimizer tasks"
        )
    return {
        storage.aliases[item["object_id"]]
        for item in registrations
        if not item["parameter"]
        and (storage.aliases[item["object_id"]] in mutated or item["name"] in controls)
    }


@contextmanager
def preserve(
    executor: Executor,
    path: ExecutionPath,
    aliases: set[str],
    *,
    directory: Path,
    chunk_bytes: int = 64 << 20,
) -> Iterator[int]:
    """Keep only affected model buffers on disk, using bounded spill reads/writes.

    Parameters, masters, moments and optimizer counters rely on the explicit
    optimizer guarantee. Gradients are recreated by the next training invocation.
    The temporary buffer file is removed even when execution fails.
    """
    host = {
        name: value.clone()
        for name, value in path.controls.host.items()
        if path.controls.bindings[name]["role"] == "hyperparameter"
    }
    storage = path.storage
    with preserve_rng(executor.device), TemporaryFile(dir=directory) as snapshot:
        sizes = []
        for alias in sorted(aliases):
            sizes.append((alias, storage.spill.sizes[alias]))
            for _, value in storage.spill.chunks(alias, chunk_bytes=chunk_bytes):
                snapshot.write(memoryview(value.numpy()))
        try:
            yield sum(size for _, size in sizes)
        finally:
            primary = sys.exception()
            try:
                executor.synchronize()
                # Restoration uses host storage only; never materialize a model copy.
                snapshot.seek(0)
                for alias, _size in sizes:
                    for _, value in storage.spill.chunks(
                        alias, chunk_bytes=chunk_bytes, read=False, write=True
                    ):
                        target = memoryview(value.numpy())
                        if snapshot.readinto(target) != len(target):
                            raise RuntimeError("incomplete diagnostic buffer snapshot")
                with torch.no_grad():
                    for name, value in host.items():
                        path.controls.host[name].copy_(value)
            except Exception as error:
                if primary is None:
                    raise
                primary.add_note(f"Diagnostic state restoration also failed: {error}")
