"""Deterministic sidecars for executable storage contracts cached by Inductor."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import torch
from torch._inductor.runtime.cache_dir_utils import cache_dir

from shadowspill.pytorch.capture.storage import TaskStorageContract

_SCHEMA = "shadowspill.inductor_task_manifest/v2"


@dataclass(frozen=True, slots=True)
class CachedTaskManifest:
    """Storage contracts tied to one exact Inductor FX-cache entry."""

    optimized_storage_contract: TaskStorageContract
    storage_contract: TaskStorageContract
    root_allocation_bytes: tuple[int, ...]
    compatibility_digest: str


def load_task_manifest(
    fx_graph_cache_key: str,
    semantic_contract_digest: str,
) -> CachedTaskManifest | None:
    """Return a validated sidecar, or ``None`` when it must be regenerated."""

    path = _manifest_path(fx_graph_cache_key, semantic_contract_digest)
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            return None
        expected = {
            "schema",
            "fx_graph_cache_key",
            "semantic_contract_digest",
            "torch_version",
            "accelerator_runtime_version",
            "optimized_storage_contract",
            "storage_contract",
            "root_allocation_bytes",
            "compatibility_digest",
        }
        if set(payload) != expected:
            return None
        if (
            payload["schema"] != _SCHEMA
            or payload["fx_graph_cache_key"] != fx_graph_cache_key
            or payload["semantic_contract_digest"] != semantic_contract_digest
            or payload["torch_version"] != torch.__version__
            or payload["accelerator_runtime_version"] != _accelerator_runtime_version()
        ):
            return None
        optimized = payload["optimized_storage_contract"]
        executable = payload["storage_contract"]
        digest = payload["compatibility_digest"]
        raw_allocation_bytes = payload["root_allocation_bytes"]
        if not isinstance(optimized, dict) or not isinstance(executable, dict):
            return None
        if not isinstance(digest, str) or len(digest) != 64:
            return None
        if not isinstance(raw_allocation_bytes, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in raw_allocation_bytes
        ):
            return None
        return CachedTaskManifest(
            TaskStorageContract.from_dict(optimized),
            TaskStorageContract.from_dict(executable),
            tuple(raw_allocation_bytes),
            digest,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def store_task_manifest(
    fx_graph_cache_key: str,
    semantic_contract_digest: str,
    manifest: CachedTaskManifest,
) -> None:
    """Atomically publish a sidecar beside Inductor's content-addressed cache."""

    path = _manifest_path(fx_graph_cache_key, semantic_contract_digest)
    payload = {
        "schema": _SCHEMA,
        "fx_graph_cache_key": fx_graph_cache_key,
        "semantic_contract_digest": semantic_contract_digest,
        "torch_version": torch.__version__,
        "accelerator_runtime_version": _accelerator_runtime_version(),
        "optimized_storage_contract": manifest.optimized_storage_contract.to_dict(),
        "storage_contract": manifest.storage_contract.to_dict(),
        "root_allocation_bytes": list(manifest.root_allocation_bytes),
        "compatibility_digest": manifest.compatibility_digest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w") as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)


def _manifest_path(
    fx_graph_cache_key: str,
    semantic_contract_digest: str,
) -> Path:
    identity = hashlib.sha256(
        f"{fx_graph_cache_key}:{semantic_contract_digest}".encode()
    ).hexdigest()
    return (
        Path(cache_dir())
        / "shadowspill"
        / "task_manifests"
        / "v2"
        / identity[:2]
        / f"{identity}.json"
    )


def _accelerator_runtime_version() -> str | None:
    return torch.version.cuda or torch.version.hip


__all__ = [
    "CachedTaskManifest",
    "load_task_manifest",
    "store_task_manifest",
]
