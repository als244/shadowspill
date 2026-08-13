"""Content-addressed persistence for complete PressureFit selections."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from shadowspill.ir import (
    MemorySchedule,
    Program,
    RecomputationSelection,
    ResidencySpec,
)
from shadowspill.simulator import SimulationConfig, simulate

from .model import (
    CandidateDiagnostic,
    PressureFitDiagnostics,
    PressureFitOptions,
    PressureFitResult,
)
from .pressurefit import pressurefit

_SCHEMA = "shadowspill.pressurefit_selection/v3"


class _ArtifactRecorder(Protocol):
    def __call__(
        self,
        *,
        category: str,
        kind: str,
        digest: str | None,
        path: str | Path,
        access: str,
        schema: str | None,
        dependencies: tuple[str, ...] = (),
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class CachedPressureFitResult:
    """One selected result plus whether it came from persistent storage."""

    result: PressureFitResult
    cache_hit: bool


class PressureFitCache:
    """Atomic cache whose key excludes non-semantic worker concurrency."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        read_enabled: bool = True,
        write_enabled: bool = True,
        overwrite: bool = False,
        artifact_recorder: _ArtifactRecorder | None = None,
    ) -> None:
        configured = os.environ.get("SHADOWSPILL_RECOMPUTATION_CACHE")
        selected = root if root is not None else configured
        self.root = (
            Path(selected).expanduser()
            if selected is not None
            else Path.home() / ".cache" / "shadowspill" / "recomputation"
        )
        self.read_enabled = read_enabled
        self.write_enabled = write_enabled
        self.overwrite = overwrite
        self.artifact_recorder = artifact_recorder

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def resolve(
        self,
        program: Program,
        *,
        initial_residency: tuple[ResidencySpec, ...],
        final_residency: tuple[ResidencySpec, ...],
        config: SimulationConfig,
        options: PressureFitOptions | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> CachedPressureFitResult:
        """Return a validated cached selection or run unmodified PressureFit."""

        selected_options = options or PressureFitOptions()
        key = _key(
            program,
            initial_residency,
            final_residency,
            config,
            selected_options,
        )
        cached = (
            self._read(
                key,
                program,
                initial_residency,
                final_residency,
                config,
                selected_options,
            )
            if self.read_enabled
            else None
        )
        if cached is not None:
            return CachedPressureFitResult(cached, True)
        result = pressurefit(
            program,
            initial_residency=initial_residency,
            final_residency=final_residency,
            config=config,
            options=selected_options,
            progress=progress,
        )
        self._write(key, result)
        return CachedPressureFitResult(result, False)

    def _read(
        self,
        key: str,
        program: Program,
        initial_residency: tuple[ResidencySpec, ...],
        final_residency: tuple[ResidencySpec, ...],
        config: SimulationConfig,
        options: PressureFitOptions,
    ) -> PressureFitResult | None:
        path = self.path(key)
        try:
            value = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"PressureFit cache entry {path} cannot be read") from exc
        if not isinstance(value, dict) or value.get("schema") != _SCHEMA:
            raise ValueError(f"PressureFit cache entry {path} has an invalid schema")
        if value.get("key_digest") != key:
            raise ValueError(f"PressureFit cache entry {path} has the wrong identity")
        if value.get("program_digest") != program.digest:
            raise ValueError(f"PressureFit cache entry {path} has the wrong Program")
        expected_boundary = {
            "initial_residency": [item.to_dict() for item in initial_residency],
            "final_residency": [item.to_dict() for item in final_residency],
            "simulation": {
                "devices": [asdict(item) for item in config.devices],
                "host_capacity_bytes": config.host_capacity_bytes,
            },
            "options": _options_identity(options),
        }
        normalized_boundary = json.loads(
            json.dumps(expected_boundary, sort_keys=True, separators=(",", ":"))
        )
        for field, expected in normalized_boundary.items():
            if value.get(field) != expected:
                raise ValueError(
                    f"PressureFit cache entry {path} has stale {field} evidence"
                )
        schedule = MemorySchedule.from_dict(value.get("schedule"))
        raw_selections = value.get("selections")
        if not isinstance(raw_selections, list):
            raise ValueError(f"PressureFit cache entry {path} has invalid selections")
        selections = tuple(
            RecomputationSelection.from_value(item, f"cache.selections[{index}]")
            for index, item in enumerate(raw_selections)
        )
        schedule.validate(program, selections)
        simulation = simulate(
            program,
            schedule,
            selections=selections,
            config=config,
        )
        diagnostics = _diagnostics_from_value(value.get("diagnostics"), path)
        if diagnostics.selected_makespan_ns != simulation.makespan_ns:
            raise ValueError(
                f"PressureFit cache entry {path} has stale simulator evidence"
            )
        result = PressureFitResult(
            program=program,
            options=options,
            initial_residency=initial_residency,
            final_residency=final_residency,
            simulation_config=config,
            schedule=schedule,
            selections=selections,
            simulation=simulation,
            diagnostics=diagnostics,
        )
        self._record(key, program.digest, path, "read")
        return result

    def _write(self, key: str, result: PressureFitResult) -> None:
        if not self.write_enabled:
            return
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": _SCHEMA,
            "key_digest": key,
            "program_digest": result.program.digest,
            "initial_residency": [item.to_dict() for item in result.initial_residency],
            "final_residency": [item.to_dict() for item in result.final_residency],
            "simulation": {
                "devices": [asdict(item) for item in result.simulation_config.devices],
                "host_capacity_bytes": result.simulation_config.host_capacity_bytes,
            },
            "options": _options_identity(result.options),
            "schedule": result.schedule.to_dict(),
            "selections": [item.to_dict() for item in result.selections],
            "diagnostics": asdict(result.diagnostics),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if path.exists() and not self.overwrite:
            try:
                existing = path.read_text()
            except OSError as exc:
                raise ValueError(
                    f"PressureFit cache entry {path} cannot be read"
                ) from exc
            if existing != encoded:
                raise ValueError(
                    "fresh PressureFit output differs from an existing cache entry; "
                    "use overwrite_plan=True or a new implementation_revision: "
                    f"{path}"
                )
            self._record(key, result.program.digest, path, "matched")
            return
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{key}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)
        self._record(key, result.program.digest, path, "write")

    def _record(
        self,
        key: str,
        program_digest: str,
        path: Path,
        access: str,
    ) -> None:
        if self.artifact_recorder is None:
            return
        self.artifact_recorder(
            category="pressurefit",
            kind="selection",
            digest=key,
            path=path,
            access=access,
            schema=_SCHEMA,
            dependencies=(program_digest,),
        )


def _key(
    program: Program,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    options: PressureFitOptions,
) -> str:
    payload = {
        "schema": _SCHEMA,
        "program_digest": program.digest,
        "initial_residency": [item.to_dict() for item in initial_residency],
        "final_residency": [item.to_dict() for item in final_residency],
        "simulation": {
            "devices": [asdict(device) for device in config.devices],
            "host_capacity_bytes": config.host_capacity_bytes,
        },
        "options": _options_identity(options),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _options_identity(options: PressureFitOptions) -> dict[str, object]:
    return {
        "initial_placement": options.initial_placement.value,
        "residency_strategies": options.residency_strategies,
        "prefetch_rules": options.prefetch_rules,
        "evaluate_coalesced": options.evaluate_coalesced,
        "max_repair_attempts": options.max_repair_attempts,
    }


def _diagnostics_from_value(value: object, path: Path) -> PressureFitDiagnostics:
    if not isinstance(value, dict):
        raise ValueError(f"PressureFit cache entry {path} has invalid diagnostics")
    raw_candidates = value.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError(f"PressureFit cache entry {path} has invalid candidates")
    try:
        candidates = tuple(CandidateDiagnostic(**item) for item in raw_candidates)
        return PressureFitDiagnostics(
            selected_candidate_id=str(value["selected_candidate_id"]),
            selected_selection_id=str(value["selected_selection_id"]),
            candidate_count=int(value["candidate_count"]),
            valid_candidate_count=int(value["valid_candidate_count"]),
            selected_makespan_ns=int(value["selected_makespan_ns"]),
            candidates=candidates,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"PressureFit cache entry {path} has invalid diagnostics"
        ) from exc


__all__ = ["CachedPressureFitResult", "PressureFitCache"]
