"""Reuse a manifest across processes, keyed by the graph Inductor compiled."""

from __future__ import annotations

from shadowspill.errors import CaptureError, CompilationError
from shadowspill.pytorch.capture.storage import (
    TaskStorageContract,
)
from shadowspill.pytorch.compilation.inductor_manifest import (
    CachedTaskManifest,
    load_task_manifest,
    store_task_manifest,
)

from .manifest import ExecutableRootAllocation, ExecutableTaskManifest, _make_manifest


def _fx_graph_cache_key(compiled: object) -> str | None:
    value = getattr(compiled, "_fx_graph_cache_key", None)
    return value if isinstance(value, str) and value else None


def _load_cached_manifest(
    cache_key: str,
    semantic_contract: TaskStorageContract,
    *,
    optimized_contract: TaskStorageContract | None,
    capture_ns: int,
) -> ExecutableTaskManifest | None:
    cached = load_task_manifest(cache_key, semantic_contract.compatibility_digest)
    if cached is None:
        return None
    if (
        optimized_contract is not None
        and optimized_contract.compatibility_digest
        != cached.optimized_storage_contract.compatibility_digest
    ):
        return None
    try:
        manifest = _make_manifest(
            semantic_contract,
            cached.optimized_storage_contract,
            cached.storage_contract,
            tuple(
                ExecutableRootAllocation(root_id, requested_bytes)
                for root_id, requested_bytes in enumerate(cached.root_allocation_bytes)
            ),
            capture_ns=capture_ns,
        )
    except (CaptureError, CompilationError, ValueError):
        return None
    return (
        manifest
        if manifest.compatibility_digest == cached.compatibility_digest
        else None
    )


def _store_cached_manifest(
    cache_key: str,
    manifest: ExecutableTaskManifest,
) -> None:
    optimized = manifest.optimized_storage_contract
    if optimized is None:
        raise AssertionError("compiled task manifest omitted its optimized contract")
    try:
        store_task_manifest(
            cache_key,
            manifest.semantic_contract_digest,
            CachedTaskManifest(
                optimized,
                manifest.storage_contract,
                tuple(
                    allocation.requested_bytes
                    for allocation in manifest.root_allocations
                ),
                manifest.compatibility_digest,
            ),
        )
    except OSError:
        # The compiler cache is an optimization. The current process already
        # owns a complete manifest and remains correct if its cache directory
        # is read-only or disappears concurrently.
        return
