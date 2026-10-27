"""Spill leases use shard bytes while registrations keep execution capacity."""

import ctypes
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from shadowspill.errors import PlanningError
from shadowspill.ir import AliasGroupSpec, TransferPhase, TransferSpec
from shadowspill.runtime.abi import ObjectLocationSnapshot
from shadowspill.runtime.failures import RuntimeExecutionError
from shadowspill.runtime.plan import objects as module
from tests.shadowspill.ir._examples import representative_program


class Library:
    host_bytes = 32

    def shadowspill_object_location_snapshot(self, runtime, object_id, pool_id, output):
        record = ctypes.cast(output, ctypes.POINTER(ObjectLocationSnapshot)).contents
        record.pointer = 4096
        record.size_bytes = self.host_bytes
        record.has_lease = record.current = 1
        return 0


def objects(monkeypatch):
    library, registrations = Library(), []
    runtime = SimpleNamespace(
        _installed=SimpleNamespace(library=library),
        _lock=threading.RLock(),
        _next_persistent_object_id=1,
        _require_open=lambda: None,
        _runtime_handle=1,
        pools={"spill": SimpleNamespace(pool_id=1, addressable=True)},
    )
    source = representative_program()
    source = replace(
        source,
        alias_groups=(
            *source.alias_groups,
            AliasGroupSpec(
                "alias_000099",
                "cuda_0",
                256,
                transfer=TransferSpec(
                    32,
                    (TransferPhase("copy", bytes=32),),
                    (TransferPhase("copy", bytes=32),),
                ),
            ),
        ),
    )
    result = module.PlanObjects(runtime, library, source, 1, spill_pool_id=1)
    monkeypatch.setattr(result, "_bind_plan_object", lambda alias: None)

    def register(runtime, object_id, size_bytes, **options):
        registrations.append((size_bytes, options))
        return 0

    monkeypatch.setattr(module, "register_object", register)
    return result, library, registrations


def test_reserve_and_host_views_keep_separate_physical_sizes(monkeypatch):
    state, library, registrations = objects(monkeypatch)
    state.reserve_spill("alias_000099", retain_spill_copy=True, transfer_address=123)
    assert registrations == [
        (
            256,
            {
                "pool_id": 1,
                "retain_spill_copy": True,
                "initially_resident": True,
                "transfer_address": 123,
            },
        )
    ]
    assert state.spill_location("alias_000099") == (4096, 32)
    library.host_bytes = 256
    with pytest.raises(RuntimeExecutionError, match="physical spill capacity"):
        state.spill_location("alias_000099")


def test_sharded_registration_requires_transport_before_reserving(monkeypatch):
    state, _, registrations = objects(monkeypatch)
    with pytest.raises(PlanningError, match="transfer descriptor"):
        state.reserve_spill("alias_000099", retain_spill_copy=True)
    with pytest.raises(PlanningError, match="transfer descriptor"):
        state.register_placeholder("alias_000099")
    assert not registrations
