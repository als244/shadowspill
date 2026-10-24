"""Communication settings that invalidate measured costs, not ownership.

Group names and membership alone do not describe collective performance. A
caller can keep those fixed while changing NCCL's GPU resource use or stream
priority. Preserve that distinction in profile and planning cache identities.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from shadowspill.communication.backend import backend_setup

if TYPE_CHECKING:
    from . import BoundDistributed


def communication_timing_identity(
    bound: BoundDistributed, *, device_ordinal: int | None = None
) -> dict[str, Any]:
    """Describe borrowed groups without constructing or mutating any group.

    Options are read from the instantiated NCCL backend, so caller-supplied
    ProcessGroupNCCL.Options and environment overrides are both represented.
    Configuration-file paths are retained; moving a debug-output file is not
    a change to the measured workload.
    """
    return backend_setup(bound.groups.groups, device_ordinal=device_ordinal)


def communication_timing_digest(bound: BoundDistributed, *, device_ordinal: int) -> str:
    record = communication_timing_identity(bound, device_ordinal=device_ordinal)
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
