"""Communication settings that invalidate measured costs, not ownership.

Group names and membership alone do not describe collective performance. A
caller can keep those fixed while changing NCCL's GPU resource use or stream
priority. Preserve that distinction in profile and planning cache identities.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from . import BoundDistributed


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


def communication_timing_identity(
    bound: BoundDistributed, *, device_ordinal: int | None = None
) -> dict[str, Any]:
    """Describe borrowed groups without constructing or mutating any group.

    Options are read from the instantiated NCCL backend, so caller-supplied
    ProcessGroupNCCL.Options and environment overrides are both represented.
    Configuration-file paths are retained; moving a debug-output file is not
    a change to the measured workload.
    """
    groups: dict[str, Any] = {}
    nccl = False
    for name, group in sorted(bound.groups.groups.items()):
        backend = str(dist.get_backend(group)).lower()
        record: dict[str, Any] = {"backend": backend}
        if backend == "nccl":
            nccl = True
            ordinal = (
                torch.cuda.current_device()
                if device_ordinal is None
                else device_ordinal
            )
            # The base Backend stub exposes only generic Options; the selected
            # NCCL backend supplies the communicator fields below.
            options: Any = group._get_backend(torch.device("cuda", ordinal)).options
            record["high_priority_stream"] = options.is_high_priority_stream
            record["config"] = {
                field: getattr(options.config, field)
                for field in _CONFIG_FIELDS
                if hasattr(options.config, field)
            }
        groups[name] = record
    return {
        "groups": groups,
        "nccl_version": list(torch.cuda.nccl.version()) if nccl else None,  # type: ignore[no-untyped-call]
        "environment": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith(("NCCL_", "TORCH_NCCL_")) and name not in _OUTPUT_PATHS
        }
        if nccl
        else {},
    }


def communication_timing_digest(bound: BoundDistributed, *, device_ordinal: int) -> str:
    record = communication_timing_identity(bound, device_ordinal=device_ordinal)
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
