"""Persistent repository of structural AOT graph pairs."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import torch

from shadowspill.errors import CaptureError
from shadowspill.planner.artifact_store import digest_directory
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.profiling import PlanningArtifactRecorder
from shadowspill.schema import artifact_schema

from ..partition.artifacts import StageExample
from .artifacts import TaskGraphPairs
from .build import build_default_graph_pairs
from .rebind import rebind_task_graph_pairs
from .serialization import (
    CachedAotGraphPair,
    atomic_json,
    restore_cached_variant,
    valid_cached_variant,
)

_GRAPH_PAIR_CACHE_SCHEMA = artifact_schema("aot_graph_pair")


class GraphPairStore:
    """Reuse AOT graph pairs while rebinding occurrence-specific values."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        read_enabled: bool = True,
        write_enabled: bool = True,
        overwrite: bool = False,
        artifact_recorder: PlanningArtifactRecorder | None = None,
    ) -> None:
        self._pairs: dict[tuple[str, tuple[int, ...], bool], TaskGraphPairs] = {}
        self._root = None if root is None else Path(root).expanduser()
        self._read_enabled = read_enabled
        self._write_enabled = write_enabled
        self._overwrite = overwrite
        self._artifact_recorder = artifact_recorder
        self._keys_seen: set[tuple[str, tuple[int, ...], bool]] = set()
        self.hits = 0
        self.misses = 0

    @property
    def unique_keys(self) -> int:
        return len(self._keys_seen)

    def resolve(
        self,
        example: StageExample,
        roots: tuple[int, ...],
        *,
        specialize_unit_tangents: bool,
        accumulating: bool = False,
    ) -> TaskGraphPairs:
        stage_contract = GraphArtifact.input_compatibility_digest(
            graph_module=example.stage.graph_module,
            example_inputs=example.inputs,
            explicit_mutations=example.stage.mutations,
            input_provenance=example.stage.input_provenance,
        )
        key = (stage_contract, roots, specialize_unit_tangents)
        self._keys_seen.add(key)
        existing = self._pairs.get(key)
        if existing is None:
            existing = self._read(key)
            if existing is not None:
                self._pairs[key] = existing
                self.hits += 1
                return rebind_task_graph_pairs(
                    self._with_accumulating(key, existing, accumulating), example
                )
            existing = build_default_graph_pairs(
                example,
                roots,
                specialize_unit_tangents=specialize_unit_tangents,
            )
            self._pairs[key] = existing
            self._write(key, existing)
            self.misses += 1
            return self._with_accumulating(key, existing, accumulating)
        self.hits += 1
        return rebind_task_graph_pairs(
            self._with_accumulating(key, existing, accumulating), example
        )

    def _with_accumulating(
        self,
        key: tuple[str, tuple[int, ...], bool],
        pairs: TaskGraphPairs,
        accumulating: bool,
    ) -> TaskGraphPairs:
        """Give this contract its accumulating form the first time one is asked for.

        Deriving it costs a graph capture, so it belongs with the structural
        entry rather than with each occurrence that rebinds from it. A step
        whose microbatches never accumulate never asks, and never pays.
        """

        if not accumulating or any(item.accumulates for item in pairs.variants):
            return pairs
        grown = replace(
            pairs,
            variants=(*pairs.variants, *pairs.accumulating_variants()),
        )
        self._pairs[key] = grown
        return grown

    def _path(self, key: tuple[str, tuple[int, ...], bool]) -> Path | None:
        if self._root is None:
            return None
        payload = {
            "schema": _GRAPH_PAIR_CACHE_SCHEMA,
            "structural_task_contract": key[0],
            "roots": key[1],
            "specialize_unit_tangents": key[2],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        # The contract and the differentiation options together are the key:
        # one entry, one digest, like every other artifact.
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        return digest_directory(self._root, digest) / "graph_pairs.pt"

    def _manifest_path(self, key: tuple[str, tuple[int, ...], bool]) -> Path | None:
        path = self._path(key)
        return None if path is None else path.with_name("manifest.json")

    def _read(self, key: tuple[str, tuple[int, ...], bool]) -> TaskGraphPairs | None:
        path = self._path(key)
        if path is None or not self._read_enabled:
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            return None
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
        ) as exc:
            raise CaptureError(f"AOT graph-pair store entry {path} is invalid") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != _GRAPH_PAIR_CACHE_SCHEMA
            or payload.get("key") != key
        ):
            raise CaptureError(f"AOT graph-pair store entry {path} has the wrong key")
        reference_option_id = payload.get("reference_option_id")
        if not isinstance(reference_option_id, str) or not reference_option_id:
            raise CaptureError(
                f"AOT graph-pair store entry {path} has no reference option"
            )
        variants = payload.get("variants")
        if (
            not isinstance(variants, tuple)
            or not variants
            or any(not valid_cached_variant(item) for item in variants)
        ):
            raise CaptureError(f"AOT graph-pair store entry {path} has invalid data")
        result = TaskGraphPairs(
            structural_contract=key[0],
            root_output_indices=key[1],
            variants=tuple(restore_cached_variant(item) for item in variants),
            reference_option_id=reference_option_id,
        )
        self._record(key, path, "read", result)
        manifest_path = self._manifest_path(key)
        if manifest_path is not None and manifest_path.exists():
            self._record(key, manifest_path, "read", result, kind="graph_pair_manifest")
        return result

    def _write(
        self,
        key: tuple[str, tuple[int, ...], bool],
        pairs: TaskGraphPairs,
    ) -> None:
        path = self._path(key)
        if path is None or not self._write_enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        cached = tuple(
            (
                item.option_id,
                item.memory_budget,
                CachedAotGraphPair.capture(item.pair),
            )
            for item in pairs.variants
        )
        if path.exists() and not self._overwrite:
            self._record(key, path, "matched", pairs)
            return
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                torch.save(
                    {
                        "schema": _GRAPH_PAIR_CACHE_SCHEMA,
                        "key": key,
                        "reference_option_id": pairs.reference_option_id,
                        "variants": cached,
                    },
                    output,
                )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)
        manifest_path = self._manifest_path(key)
        assert manifest_path is not None
        manifest = {
            "schema": _GRAPH_PAIR_CACHE_SCHEMA,
            "structural_task_contract": key[0],
            "differentiable_root_positions": list(key[1]),
            "specialize_unit_tangents": key[2],
            "reference_option_id": pairs.reference_option_id,
            "variants": [
                {
                    "option_id": item.option_id,
                    "memory_budget": item.memory_budget,
                    "forward": item.pair.forward.compatibility_digest,
                    "backward": item.pair.backward.compatibility_digest,
                }
                for item in pairs.variants
            ],
        }
        atomic_json(manifest_path, manifest)
        self._record(key, path, "write", pairs)
        self._record(
            key,
            manifest_path,
            "write",
            pairs,
            kind="graph_pair_manifest",
        )

    def _record(
        self,
        key: tuple[str, tuple[int, ...], bool],
        path: Path,
        access: str,
        pairs: TaskGraphPairs,
        *,
        kind: str = "aot_graph_pairs",
    ) -> None:
        if self._artifact_recorder is None:
            return
        digest = path.parent.name
        dependencies = (
            key[0],
            *(
                digest
                for item in pairs.variants
                for digest in (
                    item.pair.forward.compatibility_digest,
                    item.pair.backward.compatibility_digest,
                )
            ),
        )
        self._artifact_recorder(
            category="graphpairs",
            kind=kind,
            digest=digest,
            path=path,
            access=access,
            schema=_GRAPH_PAIR_CACHE_SCHEMA,
            dependencies=dependencies,
        )


__all__ = ["GraphPairStore"]
