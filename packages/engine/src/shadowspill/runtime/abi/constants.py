"""The ids and versions the adapter header defines, mirrored here."""

from __future__ import annotations

from typing import Final

ADAPTER_ABI_VERSION: Final = 3

#: Two id ranges for work that is not a planned task, mirrored from the adapter
#: header, whose failure report decodes them: allocation scope ids at or above
#: `PROFILING_SCOPE_BASE`, action-batch task ids at or above
#: `INITIAL_ACTIONS_TASK_ID`.
PROFILING_SCOPE_BASE: Final = 1 << 62

INITIAL_ACTIONS_TASK_ID: Final = 1 << 60

#: The scope a runtime-owned object's storage is attributed to: no plan owns it,
#: and any number of plans may bind the object it backs. Mirrors
#: `SHADOWSPILL_RUNTIME_OBJECT_SCOPE_ID`.
RUNTIME_OBJECT_SCOPE_ID: Final = (1 << 64) - 2
