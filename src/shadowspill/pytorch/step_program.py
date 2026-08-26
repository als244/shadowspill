"""Serializable result of capture, profiling, and canonical step lowering."""

from __future__ import annotations

import json
from dataclasses import dataclass

from shadowspill.planner.serialization import (
    _canonical_json,
    _digest,
    _integer,
    _list,
    _mapping,
    _optional_string,
    _pair,
    _string,
)

from .diagnostics import PlanCacheArtifact, PlanProfilingMetadata
from .program_inputs import PressureFitProgram

_STEP_PROGRAM_SCHEMA = "shadowspill.step_program/v1"


@dataclass(frozen=True, slots=True)
class StepProgram:
    """Public result of capture, profiling, and canonical step lowering."""

    recurrent: PressureFitProgram
    initial: PressureFitProgram | None
    optimizer_ordering: str
    signature_digests: tuple[str, ...]
    profiling_metadata: tuple[PlanProfilingMetadata, ...]
    phase_timings_ns: tuple[tuple[str, int], ...]
    cache_directories: tuple[tuple[str, str], ...]
    cache_artifacts: tuple[PlanCacheArtifact, ...]
    transfer_capabilities_json: str
    unique_profile_count: int
    captured_stage_count: int

    def __post_init__(self) -> None:
        if self.recurrent.role != "recurrent":
            raise ValueError("StepProgram.recurrent has the wrong role")
        if self.initial is not None and self.initial.role != "initial":
            raise ValueError("StepProgram.initial has the wrong role")
        parsed = json.loads(self.transfer_capabilities_json)
        if not isinstance(parsed, dict):
            raise ValueError("transfer capabilities must encode a JSON object")

    @property
    def digest(self) -> str:
        """Stable planning-content identity, excluding timing/cache evidence."""

        return _digest(
            {
                "schema": _STEP_PROGRAM_SCHEMA,
                "identity": {
                    "signature_digests": list(self.signature_digests),
                    "recurrent_program_digest": self.recurrent.program.digest,
                    "initial_program_digest": (
                        None if self.initial is None else self.initial.program.digest
                    ),
                },
                "programs": {
                    "recurrent": self.recurrent.to_dict(),
                    "initial": (
                        None if self.initial is None else self.initial.to_dict()
                    ),
                },
                "profiling": {
                    "metadata": [item.as_dict() for item in self.profiling_metadata],
                    "unique_profile_count": self.unique_profile_count,
                    "captured_stage_count": self.captured_stage_count,
                },
                "planning": {"optimizer_ordering": self.optimizer_ordering},
                "transfer_capabilities": json.loads(self.transfer_capabilities_json),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _STEP_PROGRAM_SCHEMA,
            "identity": {
                "signature_digests": list(self.signature_digests),
                "recurrent_program_digest": self.recurrent.program.digest,
                "initial_program_digest": (
                    None if self.initial is None else self.initial.program.digest
                ),
            },
            "programs": {
                "recurrent": self.recurrent.to_dict(),
                "initial": None if self.initial is None else self.initial.to_dict(),
            },
            "profiling": {
                "metadata": [item.as_dict() for item in self.profiling_metadata],
                "unique_profile_count": self.unique_profile_count,
                "captured_stage_count": self.captured_stage_count,
            },
            "planning": {
                "optimizer_ordering": self.optimizer_ordering,
                "phase_timings_ns": [list(item) for item in self.phase_timings_ns],
            },
            "transfer_capabilities": json.loads(self.transfer_capabilities_json),
            "cache_lineage": {
                "directories": [list(item) for item in self.cache_directories],
                "artifacts": [item.as_dict() for item in self.cache_artifacts],
            },
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> StepProgram:
        data = _mapping(value, "step_program")
        if data.get("schema") != _STEP_PROGRAM_SCHEMA:
            raise ValueError("step_program.schema: unsupported schema")
        identity = _mapping(data.get("identity"), "step_program.identity")
        programs = _mapping(data.get("programs"), "step_program.programs")
        profiling = _mapping(data.get("profiling"), "step_program.profiling")
        planning = _mapping(data.get("planning"), "step_program.planning")
        lineage = _mapping(data.get("cache_lineage"), "step_program.cache_lineage")
        initial_value = programs.get("initial")
        metadata = _list(profiling.get("metadata"), "step_program.profiling.metadata")
        timings = _list(
            planning.get("phase_timings_ns"), "step_program.planning.phase_timings_ns"
        )
        directories = _list(
            lineage.get("directories"), "step_program.cache_lineage.directories"
        )
        artifacts = _list(
            lineage.get("artifacts"), "step_program.cache_lineage.artifacts"
        )
        signatures = _list(
            identity.get("signature_digests"), "step_program.identity.signature_digests"
        )
        transfer = _mapping(
            data.get("transfer_capabilities"), "step_program.transfer_capabilities"
        )
        recurrent = PressureFitProgram.from_value(
            programs.get("recurrent"), "step_program.programs.recurrent"
        )
        initial = (
            None
            if initial_value is None
            else PressureFitProgram.from_value(
                initial_value, "step_program.programs.initial"
            )
        )
        expected_recurrent = _string(
            identity.get("recurrent_program_digest"),
            "step_program.identity.recurrent_program_digest",
        )
        expected_initial = _optional_string(
            identity.get("initial_program_digest"),
            "step_program.identity.initial_program_digest",
        )
        if recurrent.program.digest != expected_recurrent:
            raise ValueError("StepProgram recurrent identity digest mismatch")
        if (None if initial is None else initial.program.digest) != expected_initial:
            raise ValueError("StepProgram initial identity digest mismatch")
        return cls(
            recurrent=recurrent,
            initial=initial,
            optimizer_ordering=_string(
                planning.get("optimizer_ordering"),
                "step_program.planning.optimizer_ordering",
            ),
            signature_digests=tuple(
                _string(item, f"step_program.signature_digests[{index}]")
                for index, item in enumerate(signatures)
            ),
            profiling_metadata=tuple(
                PlanProfilingMetadata(
                    _integer(
                        item.get("position"),
                        f"step_program.profiling_metadata[{index}].position",
                    ),
                    _string(
                        item.get("digest"),
                        f"step_program.profiling_metadata[{index}].digest",
                    ),
                    _string(
                        item.get("canonical_json"),
                        f"step_program.profiling_metadata[{index}].canonical_json",
                    ),
                )
                for index, raw in enumerate(metadata)
                for item in (
                    _mapping(raw, f"step_program.profiling_metadata[{index}]"),
                )
            ),
            phase_timings_ns=tuple(
                (
                    _string(pair[0], f"step_program.phase_timings_ns[{index}][0]"),
                    _integer(pair[1], f"step_program.phase_timings_ns[{index}][1]"),
                )
                for index, raw in enumerate(timings)
                for pair in (_pair(raw, f"step_program.phase_timings_ns[{index}]"),)
            ),
            cache_directories=tuple(
                (
                    _string(pair[0], f"step_program.cache_directories[{index}][0]"),
                    _string(pair[1], f"step_program.cache_directories[{index}][1]"),
                )
                for index, raw in enumerate(directories)
                for pair in (_pair(raw, f"step_program.cache_directories[{index}]"),)
            ),
            cache_artifacts=tuple(
                PlanCacheArtifact(
                    category=_string(
                        item.get("category"),
                        f"step_program.cache_artifacts[{index}].category",
                    ),
                    kind=_string(
                        item.get("kind"), f"step_program.cache_artifacts[{index}].kind"
                    ),
                    digest=_optional_string(
                        item.get("digest"),
                        f"step_program.cache_artifacts[{index}].digest",
                    ),
                    path=_string(
                        item.get("path"), f"step_program.cache_artifacts[{index}].path"
                    ),
                    access=_string(
                        item.get("access"),
                        f"step_program.cache_artifacts[{index}].access",
                    ),
                    schema=_optional_string(
                        item.get("schema"),
                        f"step_program.cache_artifacts[{index}].schema",
                    ),
                    dependencies=tuple(
                        _string(
                            dependency,
                            f"step_program.cache_artifacts[{index}].dependencies[{dependency_index}]",
                        )
                        for dependency_index, dependency in enumerate(
                            _list(
                                item.get("dependencies"),
                                f"step_program.cache_artifacts[{index}].dependencies",
                            )
                        )
                    ),
                )
                for index, raw in enumerate(artifacts)
                for item in (_mapping(raw, f"step_program.cache_artifacts[{index}]"),)
            ),
            transfer_capabilities_json=_canonical_json(transfer),
            unique_profile_count=_integer(
                profiling.get("unique_profile_count"),
                "step_program.profiling.unique_profile_count",
            ),
            captured_stage_count=_integer(
                profiling.get("captured_stage_count"),
                "step_program.profiling.captured_stage_count",
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> StepProgram:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("step Program JSON is invalid") from error
        return cls.from_dict(value)
