"""Persistent repository of structural AOT graph-pair portfolios."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from contextlib import suppress
from pathlib import Path

import torch

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.compilation.profiling import PlanningArtifactRecorder

from ..contracts import CaptureError
from ..partition.artifacts import StageExample
from .artifacts import GraphPairPortfolio
from .build import build_default_portfolio
from .rebind import rebind_graph_pair_portfolio
from .serialization import (
    CachedAotGraphPair,
    atomic_json,
    restore_cached_variant,
    valid_cached_variant,
)

_GRAPH_PAIR_CACHE_SCHEMA = "shadowspill.aot_graph_pair/v4"


class GraphPairRepository:
    """Reuse AOT portfolios while rebinding occurrence-specific values."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        read_enabled: bool = True,
        write_enabled: bool = True,
        overwrite: bool = False,
        artifact_recorder: PlanningArtifactRecorder | None = None,
    ) -> None:
        self._pairs: dict[tuple[str, tuple[int, ...], bool], GraphPairPortfolio] = {}
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
    ) -> GraphPairPortfolio:
        stage_abi = GraphArtifact.input_compatibility_digest(
            graph_module=example.stage.graph_module,
            example_inputs=example.inputs,
            explicit_mutations=example.stage.mutations,
            input_provenance=example.stage.input_provenance,
        )
        key = (stage_abi, roots, specialize_unit_tangents)
        self._keys_seen.add(key)
        existing = self._pairs.get(key)
        if existing is None:
            existing = self._read(key)
            if existing is not None:
                self._pairs[key] = existing
                self.hits += 1
                return rebind_graph_pair_portfolio(existing, example)
            existing = build_default_portfolio(
                example,
                roots,
                specialize_unit_tangents=specialize_unit_tangents,
            )
            self._pairs[key] = existing
            self._write(key, existing)
            self.misses += 1
            return existing
        self.hits += 1
        return rebind_graph_pair_portfolio(existing, example)

    def _path(self, key: tuple[str, tuple[int, ...], bool]) -> Path | None:
        if self._root is None:
            return None
        payload = {
            "schema": _GRAPH_PAIR_CACHE_SCHEMA,
            "structural_task_abi": key[0],
            "roots": key[1],
            "specialize_unit_tangents": key[2],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        selection = hashlib.sha256(encoded.encode()).hexdigest()
        return self._root / "v4" / key[0][:2] / key[0] / selection / "graph_pairs.pt"

    def _manifest_path(self, key: tuple[str, tuple[int, ...], bool]) -> Path | None:
        path = self._path(key)
        return None if path is None else path.with_name("manifest.json")

    def _read(
        self, key: tuple[str, tuple[int, ...], bool]
    ) -> GraphPairPortfolio | None:
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
            raise CaptureError(f"AOT graph-pair cache entry {path} is invalid") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != _GRAPH_PAIR_CACHE_SCHEMA
            or payload.get("key") != key
        ):
            raise CaptureError(f"AOT graph-pair cache entry {path} has the wrong key")
        reference_option_id = payload.get("reference_option_id")
        if not isinstance(reference_option_id, str) or not reference_option_id:
            raise CaptureError(
                f"AOT graph-pair cache entry {path} has no reference option"
            )
        variants = payload.get("variants")
        if (
            not isinstance(variants, tuple)
            or not variants
            or any(not valid_cached_variant(item) for item in variants)
        ):
            raise CaptureError(f"AOT graph-pair cache entry {path} has invalid data")
        result = GraphPairPortfolio(
            structural_abi=key[0],
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
        pairs: GraphPairPortfolio,
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
            "structural_task_abi": key[0],
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
        pairs: GraphPairPortfolio,
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


__all__ = ["GraphPairRepository"]
