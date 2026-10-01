"""Atomic, same-topology distributed checkpoint publication on shared storage."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from ._control import Control

if TYPE_CHECKING:
    from . import BoundDistributed


def save_collective(
    path: str | Path,
    control: Control,
    write_rank: Callable[[Path], None],
    *,
    step: int,
    layout: Mapping[str, Any],
) -> Path:
    """All ranks finish their independent files before the manifest is published.

    Failed partial writes remain in a hidden directory for diagnosis. They never
    have a complete checkpoint path or readable manifest. No model tensors are
    gathered to a leader and no compute-device collectives run here.
    """
    path = Path(path).resolve()
    control.agree("checkpoint/request", {"path": str(path), "step": step, "version": 1})

    def create() -> str | None:
        if control.rank != control.members[0]:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"checkpoint already exists: {path}")
        return tempfile.mkdtemp(prefix="." + path.name + "-", dir=path.parent)

    temporary = control.run("checkpoint/create", create)
    locations = control.exchange("checkpoint/location", temporary)
    root = Path(locations[0])

    def write() -> dict[str, Any]:
        if not root.is_dir():
            raise ValueError(
                "checkpoint root must be visible on every participant (shared storage)"
            )
        rank_dir = root / f"rank-{control.rank:05d}"
        rank_dir.mkdir()
        write_rank(rank_dir)
        files = {
            str(item.relative_to(rank_dir)): item.stat().st_size
            for item in rank_dir.rglob("*")
            if item.is_file()
        }
        if not files:
            raise ValueError("checkpoint writer produced no files")
        return {"rank": control.rank, "layout": layout, "files": files}

    local = control.run("checkpoint/write", write)
    records = control.exchange("checkpoint/inventory", local)

    def publish() -> None:
        if control.rank != control.members[0]:
            return
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "step": step,
                    "members": list(control.members),
                    "ranks": records,
                },
                indent=2,
            )
            + "\n"
        )
        # Recheck so a preexisting experiment checkpoint cannot be replaced.
        if path.exists():
            raise FileExistsError(f"checkpoint appeared before publication: {path}")
        os.rename(root, path)

    control.run("checkpoint/publish", publish)
    return path


def load_collective(
    path: str | Path,
    control: Control,
    *,
    layout: Mapping[str, Any],
) -> tuple[Path, int]:
    """Validate every rank's files/topology before any state is restored."""
    path = Path(path).resolve()

    def read() -> tuple[Path, int]:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("version") != 1 or manifest.get("members") != list(
            control.members
        ):
            raise ValueError("checkpoint topology or version differs")
        records = manifest["ranks"]
        if [record["rank"] for record in records] != list(control.members):
            raise ValueError("checkpoint rank inventory is incomplete")
        record = records[control.members.index(control.rank)]
        if record["layout"] != layout:
            raise ValueError(
                f"checkpoint parameter ownership differs on rank {control.rank}"
            )
        root = path / f"rank-{control.rank:05d}"
        for name, size in record["files"].items():
            file = (root / name).resolve()
            if (
                not file.is_relative_to(root)
                or not file.is_file()
                or file.stat().st_size != size
            ):
                raise ValueError(f"checkpoint file missing or truncated: {file}")
        return root, manifest["step"]

    root, step = control.run("checkpoint/validate", read)
    control.agree("checkpoint/step", step)
    return root, step


def master_aliases(state: Mapping[str, Any], bound: BoundDistributed) -> set[str]:
    """Registered compute names represented by a saved master instead."""
    names = set(state.get("masters", {}))
    return {
        alias
        for parameter in bound.parameters
        if parameter.name in names
        for alias in parameter.aliases
    }


def restore_compute_weights(
    destinations: Mapping[str, torch.Tensor],
    masters: Mapping[str, torch.Tensor],
    bound: BoundDistributed,
    *,
    chunk_bytes: int = 16 << 20,
) -> None:
    """Rebuild compute weights from saved masters using bounded CPU transfers.

    Checkpoints store each trained weight once. Owners broadcast their saved
    slices through the CPU control group. Replicas cast directly into existing
    model storage; no full master is gathered and no device task runs.
    """
    control = bound.control
    parameters = {item.name: item for item in bound.parameters}

    def describe() -> list[dict[str, Any]]:
        if chunk_bytes < 1:
            raise ValueError("checkpoint transfer chunk_bytes must be positive")
        result = []
        for name, value in masters.items():
            if name not in parameters or not isinstance(value, torch.Tensor):
                raise ValueError(f"invalid checkpoint master {name!r}")
            parameter = parameters[name]
            owners = parameter.replicas if bound.shard_optimizer else (control.rank,)
            capacity = math.ceil(math.prod(parameter.shape) / len(owners))
            expected = (capacity,) if len(owners) > 1 else parameter.shape
            if tuple(value.shape) != expected or value.device.type != "cpu":
                raise ValueError(f"checkpoint master geometry differs for {name!r}")
            destination = destinations.get(name)
            if (
                not isinstance(destination, torch.Tensor)
                or destination.device.type != "cpu"
                or tuple(destination.shape) != parameter.shape
                or str(destination.dtype) != parameter.dtype
            ):
                raise ValueError(f"checkpoint compute destination differs for {name!r}")
            result.append(
                {
                    "name": name,
                    "owners": owners,
                    "replicas": parameter.replicas,
                    "shape": parameter.shape,
                    "dtype": str(value.dtype),
                }
            )
        return result

    local = control.run("checkpoint/master_geometry", describe)
    control.agree("checkpoint/transfer_chunk", chunk_bytes)
    records = control.exchange("checkpoint/master_inventory", local)
    by_rank = {
        rank: {item["name"]: item for item in entries}
        for rank, entries in zip(control.members, records, strict=True)
    }
    unique = {}
    for entries in by_rank.values():
        for name, item in entries.items():
            for peer in item["replicas"]:
                other = by_rank[peer].get(name)
                if other is None or any(
                    other[key] != item[key] for key in ("shape", "dtype", "replicas")
                ):
                    raise ValueError(
                        f"checkpoint replicas disagree about master {name!r}"
                    )
            unique[(tuple(item["owners"]), name)] = item

    def restore_local() -> None:
        with torch.no_grad():
            for name, master in masters.items():
                if not bound.shard_optimizer or len(parameters[name].replicas) == 1:
                    destinations[name].copy_(master)

    control.run("checkpoint/local_weights", restore_local)
    with torch.no_grad():
        for (owners, name), item in sorted(unique.items()):
            if len(owners) == 1:
                continue
            dtype = getattr(torch, item["dtype"].removeprefix("torch."))
            count = math.prod(item["shape"])
            capacity = math.ceil(count / len(owners))
            elements = max(
                1, chunk_bytes // torch.empty((), dtype=dtype).element_size()
            )
            destination = (
                destinations[name] if control.rank in item["replicas"] else None
            )
            for index, owner in enumerate(owners):
                start = index * capacity
                length = max(0, min(capacity, count - start))
                for offset in range(0, length, elements):
                    extent = min(elements, length - offset)

                    def stage_tile(
                        *,
                        name: str = name,
                        owner: int = owner,
                        offset: int = offset,
                        extent: int = extent,
                        dtype: torch.dtype = dtype,
                    ) -> torch.Tensor:
                        tile = torch.empty(extent, dtype=dtype)
                        if control.rank == owner:
                            tile.copy_(
                                masters[name].reshape(-1).narrow(0, offset, extent)
                            )
                        return tile

                    tile = control.run("checkpoint/stage_master_tile", stage_tile)
                    # Staging succeeded everywhere before any rank enters Gloo.
                    dist.broadcast(
                        tile.view(torch.uint8), src=owner, group=control.group
                    )

                    def install_tile(
                        *,
                        destination: torch.Tensor | None = destination,
                        tile: torch.Tensor = tile,
                        begin: int = start + offset,
                    ) -> None:
                        if destination is None:
                            return
                        if destination.is_contiguous():
                            destination.view(-1).narrow(0, begin, tile.numel()).copy_(
                                tile
                            )
                        else:
                            indices = torch.arange(begin, begin + tile.numel())
                            destination[
                                torch.unravel_index(indices, destination.shape)
                            ] = tile.to(destination.dtype)

                    control.run("checkpoint/install_compute_tile", install_tile)
    control.exchange("checkpoint/weights_restored", True)


def masters_from_compute(
    weights: Mapping[str, torch.Tensor],
    names: set[str],
    bound: BoundDistributed,
) -> dict[str, torch.Tensor]:
    """Select owned compute slices; OptimizerState.load casts into master storage.

    Only each rank's slice is materialized, still at the saved compute precision.
    Unsharded weights remain views of the checkpoint. No full master is built.
    """
    from ._optimizer import UpdateLayout
    from ._shards import fill_parameter

    result = {}
    for parameter in bound.parameters:
        if parameter.name not in names:
            continue
        value = weights[parameter.name]
        if tuple(value.shape) != parameter.shape or value.device.type != "cpu":
            raise ValueError(
                f"checkpoint compute geometry differs for {parameter.name!r}"
            )
        size = len(parameter.replicas) if bound.shard_optimizer else 1
        if size == 1:
            result[parameter.name] = value
            continue
        layout = UpdateLayout(
            parameter.shape,
            value.dtype,
            None,
            size,
            parameter.replicas.index(bound.control.rank),
        )
        owned = torch.empty(layout.capacity, dtype=value.dtype)
        fill_parameter(owned, value, layout)
        result[parameter.name] = owned
    return result
