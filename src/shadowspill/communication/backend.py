"""Describe backend settings separately from compute build identity."""

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

_CONFIG_FIELDS = (
    "blocking",
    "cga_cluster_size",
    "collnet_enable",
    "cta_policy",
    "max_ctas",
    "min_ctas",
    "net_name",
    "nvls_ctas",
    "split_share",
    "traffic_class",
)
_OUTPUT_PATHS = frozenset(
    {"NCCL_DEBUG_FILE", "NCCL_TOPO_DUMP_FILE", "NCCL_GRAPH_DUMP_FILE"}
)


def backend_setup(
    groups: Mapping[str, dist.ProcessGroup], *, device_ordinal: int | None = None
) -> dict[str, Any]:
    """Inspect bound group options; do not construct groups or run collectives."""
    descriptions: dict[str, Any] = {}
    nccl = False
    for name, group in sorted(groups.items()):
        kind = str(dist.get_backend(group)).lower()
        record: dict[str, Any] = {"backend": kind}
        if kind == "nccl":
            nccl = True
            ordinal = (
                torch.cuda.current_device()
                if device_ordinal is None
                else device_ordinal
            )
            options: Any = group._get_backend(torch.device("cuda", ordinal)).options
            record["high_priority_stream"] = options.is_high_priority_stream
            record["config"] = {
                field: getattr(options.config, field)
                for field in _CONFIG_FIELDS
                if hasattr(options.config, field)
            }
        descriptions[name] = record
    result = {
        "groups": descriptions,
        "nccl_version": list(torch.cuda.nccl.version()) if nccl else None,  # type: ignore[no-untyped-call]
        "environment": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith(("NCCL_", "TORCH_NCCL_")) and name not in _OUTPUT_PATHS
        }
        if nccl
        else {},
    }
    if nccl:
        # A filename is not sufficient: a config/topology file can change in place.
        paths = [Path("/etc/nccl.conf"), Path.home() / ".nccl.conf"]
        paths.extend(
            Path(os.environ[key])
            for key in ("NCCL_CONF_FILE", "NCCL_TOPO_FILE", "NCCL_GRAPH_FILE")
            if os.environ.get(key)
        )
        result["config_files"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else None
            for path in sorted(set(paths))
        }
    return result
