"""The ids and versions the adapter header defines, mirrored here."""

from __future__ import annotations

from typing import Final

ADAPTER_ABI_VERSION: Final = 3

#: Ids the frontend synthesises for work that is not a planned task, copied
#: from the adapter header, which decodes them in its failure report.
PROFILING_SCOPE_BASE: Final = 1 << 62

INITIAL_ACTIONS_TASK_ID: Final = 1 << 60

#: The scope a runtime-owned object's storage is attributed to: no plan owns it,
#: and any number of plans may bind the object it backs. Mirrors
#: `SHADOWSPILL_RUNTIME_OBJECT_SCOPE_ID`.
RUNTIME_OBJECT_SCOPE_ID: Final = (1 << 64) - 2
