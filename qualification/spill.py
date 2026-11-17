"""Portable spill configurations shared by benchmark and qualification workers."""

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


def parse_remote(value):
    """Read HOST:PORT:BYTES without connecting to the backend."""
    from shadowspill.network import RemotePool

    try:
        host, port, size = value.rsplit(":", 2)
        if not host or not port.isdigit() or not size.isdigit():
            raise ValueError
    except (AttributeError, ValueError) as error:
        raise ValueError("--remote-spill must read HOST:PORT:BYTES") from error
    return RemotePool(capacity=int(size), host=host, port=int(port))
