"""The ids and versions the adapter header defines, mirrored here."""

from __future__ import annotations

from typing import Final

ADAPTER_ABI_VERSION: Final = 1

#: The id range for work that is not a planned task, mirrored from the adapter
#: header, whose failure report decodes it: allocation scope ids at or above
#: `PROFILING_SCOPE_BASE`.
PROFILING_SCOPE_BASE: Final = 1 << 62

#: The scope a runtime-owned object's storage is attributed to: no plan owns it,
#: and any number of plans may bind the object it backs. Mirrors
#: `SHADOWSPILL_RUNTIME_OBJECT_SCOPE_ID`.
RUNTIME_OBJECT_SCOPE_ID: Final = (1 << 64) - 2
