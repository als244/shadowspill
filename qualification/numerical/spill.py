"""Spill selection shared by the numerical matrix and standalone cases."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from shadowspill.memory import SpillPool
from shadowspill.network import RemotePool
from shadowspill.ssd import SSDPool, ssd

from .tolerances import SPILL_BUDGET


def add_spill_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--spill-pool",
        choices=("host", "ssd"),
        default="host",
        help="planned-arm storage; references always use ordinary PyTorch",
    )
    parser.add_argument(
        "--ssd-directory",
        type=Path,
        default=os.environ.get("SHADOWSPILL_SSD_DIRECTORY"),
        help="existing local SSD directory (or SHADOWSPILL_SSD_DIRECTORY)",
    )
    parser.add_argument(
        "--ssd-staging-mib",
        type=int,
        default=256,
        help="host payload staging cap across SSD imports and lanes (default: 256 MiB)",
    )
    parser.add_argument(
        "--ssd-chunk-mib",
        type=int,
        default=2,
        help="direct-I/O pipeline chunk size (default: 2 MiB)",
    )
    parser.add_argument(
        "--ssd-queue-depth",
        type=int,
        default=16,
        help="maximum in-flight chunks per transfer direction (default: 16)",
    )
    parser.add_argument(
        "--remote-spill",
        metavar="HOST:PORT:BYTES",
        help="planned-arm storage on a memory daemon instead of pinned host",
    )


def configured_spill(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> SpillPool | None:
    """Validate configuration without acquiring a pool or loading its library."""
    try:
        if arguments.remote_spill is not None:
            if arguments.spill_pool != "host":
                raise ValueError(
                    "--remote-spill cannot be combined with --spill-pool ssd"
                )
            host, _, rest = arguments.remote_spill.partition(":")
            port, _, size = rest.partition(":")
            if not host or not port.isdigit() or not size.isdigit():
                raise ValueError("--remote-spill must read HOST:PORT:BYTES")
            return RemotePool(capacity=int(size), host=host, port=int(port))
        if arguments.spill_pool == "host":
            return None
        if arguments.ssd_directory is None:
            raise ValueError(
                "SSD spill requires --ssd-directory or SHADOWSPILL_SSD_DIRECTORY"
            )
        return ssd(
            capacity=SPILL_BUDGET,
            directory=arguments.ssd_directory,
            staging_bytes=arguments.ssd_staging_mib << 20,
            chunk_bytes=arguments.ssd_chunk_mib << 20,
            queue_depth=arguments.ssd_queue_depth,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))


def spill_arguments(pool: SpillPool | None) -> list[str]:
    """Serialize only the planned arm's pool; reference identity is unchanged."""
    if pool is None:
        return []
    if isinstance(pool, RemotePool):
        return ["--remote-spill", f"{pool.host}:{pool.port}:{pool.capacity}"]
    if isinstance(pool, SSDPool):
        return [
            "--spill-pool",
            "ssd",
            "--ssd-directory",
            str(pool.directory),
            "--ssd-staging-mib",
            str(pool.staging_bytes >> 20),
            "--ssd-chunk-mib",
            str(pool.chunk_bytes >> 20),
            "--ssd-queue-depth",
            str(pool.queue_depth),
        ]
    raise TypeError(f"unsupported numerical spill pool: {type(pool).__name__}")


def spill_description(pool: SpillPool | None) -> dict[str, object]:
    """Record the storage configuration alongside the numerical verdict."""
    result: dict[str, object] = {
        "kind": "pinned_host" if pool is None else pool.kind_name,
        "capacity_bytes": SPILL_BUDGET if pool is None else pool.capacity,
    }
    if isinstance(pool, SSDPool):
        result.update(
            directory=str(pool.directory),
            staging_bytes=pool.staging_bytes,
            chunk_bytes=pool.chunk_bytes,
            queue_depth=pool.queue_depth,
        )
    elif isinstance(pool, RemotePool):
        result.update(host=pool.host, port=pool.port)
    return result
