"""One recomputation selection: what it chose, and what it cost."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .candidates import CandidateDiagnostic
from .counters import (
    PressureFitRepairDiagnostics,
    PressureFitWorkDiagnostics,
)
from .json import (
    _list,
    _mapping,
    _optional_integer,
    _optional_string,
    _string,
)


@dataclass(frozen=True, slots=True)
class RecomputationChoiceDiagnostic:
    """One graph-pair choice in a recomputation selection."""

    group_id: str
    option_id: str

    def to_dict(self) -> dict[str, str]:
        return {"group_id": self.group_id, "option_id": self.option_id}

    @classmethod
    def from_value(cls, value: object, path: str) -> RecomputationChoiceDiagnostic:
        data = _mapping(value, path)
        return cls(
            group_id=_string(data.get("group_id"), f"{path}.group_id"),
            option_id=_string(data.get("option_id"), f"{path}.option_id"),
        )


@dataclass(frozen=True, slots=True)
class RecomputationProblemDiagnostics:
    """Shared problem work and policy evaluations for one concrete selection."""

    selection_id: str
    choices: tuple[RecomputationChoiceDiagnostic, ...]
    selected_candidate_id: str | None
    selected_makespan_ns: int | None
    candidate_evaluations: tuple[CandidateDiagnostic, ...]
    work: PressureFitWorkDiagnostics = field(default_factory=PressureFitWorkDiagnostics)

    def __post_init__(self) -> None:
        if any(
            candidate.selection_id != self.selection_id
            for candidate in self.candidate_evaluations
        ):
            raise ValueError(
                "candidate evaluation does not reference its recomputation problem"
            )
        candidate_ids = tuple(
            candidate.candidate_id for candidate in self.candidate_evaluations
        )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(
                "candidate policy appears more than once in a recomputation problem"
            )
        valid = tuple(
            candidate
            for candidate in self.candidate_evaluations
            if candidate.status == "valid"
        )
        if self.selected_candidate_id is None:
            if self.selected_makespan_ns is not None:
                raise ValueError(
                    "selected_makespan_ns requires a selected candidate policy"
                )
        else:
            selected = tuple(
                candidate
                for candidate in valid
                if candidate.candidate_id == self.selected_candidate_id
            )
            if len(selected) != 1:
                raise ValueError(
                    "selected candidate policy is not a unique valid evaluation"
                )
            if self.selected_makespan_ns != selected[0].makespan_ns:
                raise ValueError(
                    "problem selected makespan does not match candidate evaluation"
                )

    @property
    def candidate_policy_count(self) -> int:
        return len(self.candidate_evaluations)

    @property
    def valid_candidate_policy_count(self) -> int:
        return sum(item.status == "valid" for item in self.candidate_evaluations)

    @property
    def candidate_status_counts(self) -> dict[str, int]:
        return dict(
            sorted(Counter(item.status for item in self.candidate_evaluations).items())
        )

    @property
    def repairs(self) -> PressureFitRepairDiagnostics:
        result = PressureFitRepairDiagnostics()
        for candidate in self.candidate_evaluations:
            result += candidate.repairs
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "recomputation_selection": {
                "selection_id": self.selection_id,
                "graph_pair_choices": [item.to_dict() for item in self.choices],
            },
            "selected_candidate_policy": {
                "candidate_id": self.selected_candidate_id,
                "makespan_ns": self.selected_makespan_ns,
            },
            "summary": {
                "candidate_policy_count": self.candidate_policy_count,
                "valid_candidate_policy_count": (self.valid_candidate_policy_count),
                "candidate_status_counts": self.candidate_status_counts,
            },
            "work": self.work.to_dict(),
            "repairs": self.repairs.to_dict(),
            "candidate_policy_evaluations": [
                item.to_dict() for item in self.candidate_evaluations
            ],
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> RecomputationProblemDiagnostics:
        data = _mapping(value, path)
        selection = _mapping(
            data.get("recomputation_selection"),
            f"{path}.recomputation_selection",
        )
        selected = _mapping(
            data.get("selected_candidate_policy"),
            f"{path}.selected_candidate_policy",
        )
        summary = _mapping(data.get("summary"), f"{path}.summary")
        selection_id = _string(
            selection.get("selection_id"),
            f"{path}.recomputation_selection.selection_id",
        )
        raw_candidates = _list(
            data.get("candidate_policy_evaluations"),
            f"{path}.candidate_policy_evaluations",
        )
        candidates = tuple(
            CandidateDiagnostic.from_value(
                item,
                f"{path}.candidate_policy_evaluations[{index}]",
            ).with_selection_id(selection_id)
            for index, item in enumerate(raw_candidates)
        )
        result = cls(
            selection_id=selection_id,
            choices=tuple(
                RecomputationChoiceDiagnostic.from_value(
                    item,
                    f"{path}.recomputation_selection.graph_pair_choices[{index}]",
                )
                for index, item in enumerate(
                    _list(
                        selection.get("graph_pair_choices"),
                        f"{path}.recomputation_selection.graph_pair_choices",
                    )
                )
            ),
            selected_candidate_id=_optional_string(
                selected.get("candidate_id"),
                f"{path}.selected_candidate_policy.candidate_id",
            ),
            selected_makespan_ns=_optional_integer(
                selected.get("makespan_ns"),
                f"{path}.selected_candidate_policy.makespan_ns",
            ),
            candidate_evaluations=candidates,
            work=PressureFitWorkDiagnostics.from_value(
                data.get("work"), f"{path}.work"
            ),
        )
        expected_summary = {
            "candidate_policy_count": result.candidate_policy_count,
            "valid_candidate_policy_count": result.valid_candidate_policy_count,
            "candidate_status_counts": result.candidate_status_counts,
        }
        for name, expected in expected_summary.items():
            if summary.get(name) != expected:
                raise ValueError(f"{path}.summary.{name} does not reconcile")
        if (
            PressureFitRepairDiagnostics.from_value(
                data.get("repairs"), f"{path}.repairs"
            )
            != result.repairs
        ):
            raise ValueError(f"{path}.repairs does not reconcile")
        return result
