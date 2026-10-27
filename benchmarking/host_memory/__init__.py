"""Host-memory and physical-device preflight before benchmark pool creation."""

import torch.distributed as dist

from .facts import local_resources
from .limits import validate_resources


def preflight(device, spill_bytes, staging_bytes, *, group=None):
    """Check shared host limits, reporting local probe failures to every worker."""
    if group is None:
        validate_resources((local_resources(device, spill_bytes, staging_bytes),))
        return
    if str(dist.get_backend(group)).lower() != "gloo":
        raise ValueError("host-memory preflight requires a Gloo group")
    record, error = None, None
    try:
        record = local_resources(device, spill_bytes, staging_bytes)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    observed = [None] * dist.get_world_size(group)
    dist.all_gather_object(observed, (record, error), group=group)
    failures = [
        f"worker {rank}: {failure}"
        for rank, (_, failure) in enumerate(observed)
        if failure
    ]
    if failures:
        raise ValueError("host-memory preflight failed: " + "; ".join(failures))
    validate_resources([record for record, _ in observed])
