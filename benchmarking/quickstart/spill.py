"""Persist pool configuration; resolve live backends only in communication/execution."""

import argparse
import importlib
import json
from dataclasses import fields
from pathlib import Path

from shadowspill.memory import SpillPool


def record(pool):
    if pool is None or isinstance(pool, dict):
        return pool
    if not isinstance(pool, SpillPool):
        raise TypeError("spill must be a SpillPool configuration")
    arguments = {
        field.name: getattr(pool, field.name)
        for field in fields(pool)
        if field.init and not field.name.startswith("_")
    }
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in arguments.items()}
    result = {
        "type": f"{type(pool).__module__}:{type(pool).__qualname__}",
        "arguments": arguments,
    }
    json.dumps(result)  # Reject process-local objects before launching any workers.
    return result


def resolve(request):
    value = request.get("spill")
    if value is None:
        return None
    module, name = value["type"].split(":", 1)
    factory = importlib.import_module(module)
    for part in name.split("."):
        factory = getattr(factory, part)
    if not isinstance(factory, type) or not issubclass(factory, SpillPool):
        raise TypeError("saved spill type must be a SpillPool configuration")
    return factory(**value["arguments"])


def add_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ssd-spill", type=Path)
    group.add_argument("--remote-spill", metavar="HOST:PORT")
    parser.add_argument("--ssd-staging-mib", type=int, default=256)
    parser.add_argument("--ssd-chunk-mib", type=int, default=2)
    parser.add_argument("--ssd-queue-depth", type=int, default=16)


def from_arguments(args, capacity):
    """No filesystem/backend import here: even planning may run on another host."""
    arguments = {"capacity": capacity}
    if args.ssd_spill is not None:
        arguments.update(
            directory=str(args.ssd_spill.expanduser().resolve()),
            staging_bytes=args.ssd_staging_mib << 20,
            chunk_bytes=args.ssd_chunk_mib << 20,
            queue_depth=args.ssd_queue_depth,
        )
        return {"type": "shadowspill.ssd:SSDPool", "arguments": arguments}
    if args.remote_spill is not None:
        host, colon, port = args.remote_spill.rpartition(":")
        if not host or not colon or not port.isdigit() or not 1 <= int(port) <= 65535:
            raise argparse.ArgumentTypeError("--remote-spill must be HOST:PORT")
        arguments.update(host=host, port=int(port))
        return {"type": "shadowspill.network:RemotePool", "arguments": arguments}
    return None
