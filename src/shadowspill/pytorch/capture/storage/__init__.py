"""One task's semantic storage contract: its roots, and how it is captured.

The records themselves live in ``records``, capturing one from a task that has run in
``capture``, and building one by replaying a task symbolically in
``symbolic``.
"""

from __future__ import annotations

from .capture import ExplicitMutation, capture_task_storage_contract
from .records import (
    MutationBinding,
    OutputView,
    StorageRoot,
    StorageRootKind,
    TaskStorageContract,
)
from .symbolic import make_storage_contract

__all__ = [
    "ExplicitMutation",
    "MutationBinding",
    "OutputView",
    "StorageRoot",
    "StorageRootKind",
    "TaskStorageContract",
    "capture_task_storage_contract",
    "make_storage_contract",
]
