"""Known preparation failures prune dependent cases, never requested capacity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, ClassVar, Literal, cast

from shadowspill.artifacts.archive import Artifact
from shadowspill.errors import PlanInfeasibleError
from shadowspill_client._records import text
from shadowspill_client.inputs.coverage import certify_coverage
from shadowspill_client.inputs.coverage_bounds import CoverageError

from .family_identity import candidate_identity
from .family_records import PlanRejection
from .layouts import LayoutRejection
from .templates import TemplateInventory


@dataclass(frozen=True)
class PreparationFailure(Artifact):
    """A recoverable device-exhaustion result, tied to one exact raw capture."""

    artifact_kind: ClassVar[str] = "preparation_failure"
    metadata_limit_bytes: ClassVar[int | None] = 1 << 20
    payload_limit_bytes: ClassVar[int | None] = 0
    template: str
    phase: Literal["build", "profile"]
    reason: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.template, str)
            or len(self.template) != 64
            or any(c not in "0123456789abcdef" for c in self.template)
        ):
            raise ValueError("preparation failure requires an exact template identity")
        if self.phase not in ("build", "profile"):
            raise ValueError("recoverable preparation failure must name build/profile")
        text(self.reason, "preparation failure reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "template": self.template,
            "phase": self.phase,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> PreparationFailure:
        if (
            payloads
            or set(value) != {"version", "template", "phase", "reason"}
            or type(value["version"]) is not int
            or value["version"] != 1
        ):
            raise ValueError("unsupported preparation failure record")
        return cls(value["template"], value["phase"], value["reason"])


def prune_failed_templates(
    inventory: TemplateInventory, failures: Mapping[str, PreparationFailure]
) -> tuple[TemplateInventory, tuple[PlanRejection, ...]]:
    """Remove complete candidate dependencies, then re-prove every capability.

    Callers supply only recorded recoverable build/profile failures. Unexpected
    exceptions abort their phase and never become a candidate rejection here.
    """
    if failures.keys() - inventory.templates.keys() or any(
        key != failure.template for key, failure in failures.items()
    ):
        raise ValueError("preparation failures differ from the capture inventory")
    families, rejected, needed = {}, [], set()
    layout_rejected = list(inventory.layouts.rejected)
    for operation, rows in inventory.families.items():
        kept = []
        for row in rows:
            dependencies = {
                use.template
                for worker in row.workers.values()
                for use in (worker.main, worker.masked, worker.optimizer)
                if use is not None
            }
            missing = sorted(dependencies & failures.keys())
            if not missing:
                kept.append(row)
                needed.update(dependencies)
                continue
            shape = row.layout.candidate
            for key in missing:
                failure = failures[key]
                reason = f"template {key}: {failure.reason}"
                rejected.append(
                    PlanRejection(
                        candidate_identity(row, None),
                        operation,
                        shape.batch,
                        shape.case,
                        shape.count,
                        None,
                        failure.phase,
                        reason,
                    )
                )
                layout_rejected.append(
                    LayoutRejection(
                        operation,
                        shape.batch,
                        shape.case,
                        shape.count,
                        failure.phase,
                        reason,
                    )
                )
        families[operation] = tuple(kept)
    setup = inventory.layouts.setup
    placement = setup.context.parallelism.inputs
    assert placement is not None
    coverage = {}
    for operation, rows in families.items():
        for batch, config in enumerate(setup.config.batches):
            try:
                schedules = [
                    row.layout.candidate.schedule
                    for row in rows
                    if row.layout.candidate.batch == batch
                ]
                if not schedules:
                    raise CoverageError("no prepared candidate remains")
                coverage[operation, batch] = certify_coverage(
                    setup.context.schema,
                    config,
                    placement,
                    schedules,
                    grouping=cast(
                        Literal["independent", "grouped"],
                        setup.model.definition.batching,
                    ),
                )
            except CoverageError as error:
                raise PlanInfeasibleError(
                    f"preparation failures removed coverage for {operation.operation}/"
                    f"{operation.outputs}/{operation.mode}, batch {batch}: {error}",
                    kind="prepared_capture_coverage",
                    diagnostics=tuple(rejected),
                ) from error
    layouts = replace(
        inventory.layouts,
        families=MappingProxyType(
            {op: tuple(row.layout for row in rows) for op, rows in families.items()}
        ),
        coverage=MappingProxyType(coverage),
        rejected=tuple(layout_rejected),
    )
    return TemplateInventory(
        layouts,
        MappingProxyType(
            {
                key: template
                for key, template in inventory.templates.items()
                if key in needed
            }
        ),
        MappingProxyType(families),
    ), tuple(rejected)
