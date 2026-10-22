"""Persistent candidate computations and the plans selected for each budget."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.pytorch.prepared import PreparedStep
from shadowspill.schema import artifact_schema
from shadowspill.search.report import StepSearchReport
from shadowspill.store import atomic_json, atomic_text


@dataclass(frozen=True)
class PreparedTraining:
    """Reusable computation and search results, with no training state attached."""

    candidates: Mapping[str, PreparedStep]
    search: StepSearchReport

    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        candidates = {}
        for name, prepared in self.candidates.items():
            # Publish immutable candidate directories before the training
            # manifest, so an interrupted replacement leaves the old bundle valid.
            with tempfile.TemporaryDirectory(dir=root, prefix=".preparing-") as stage:
                prepared.save(stage)
                key = (
                    "candidate-"
                    + hashlib.sha256(
                        (Path(stage) / "manifest.json").read_bytes()
                    ).hexdigest()
                )
                if not (root / key).exists():
                    try:
                        os.rename(stage, root / key)
                    except OSError:
                        # A concurrent publication may report EEXIST or
                        # ENOTEMPTY. Accept only the identical complete artifact.
                        manifest = root / key / "manifest.json"
                        if (
                            not manifest.is_file()
                            or manifest.read_bytes()
                            != (Path(stage) / "manifest.json").read_bytes()
                        ):
                            raise
            candidates[name] = key
        plans = []
        for budget, plan in self.search.winner_plans.items():
            filename = f"plan-{plan.digest}.json"
            atomic_text(root / filename, plan.to_json())
            plans.append({"budget": list(budget), "file": filename})
        # The search is embedded so publication of this manifest is the commit
        # point. An interrupted build has no complete artifact to resume from.
        atomic_json(
            root / "training.json",
            {
                "schema": artifact_schema("prepared_training"),
                "candidates": candidates,
                "plans": plans,
                "search": self.search.to_dict(),
            },
        )

    @classmethod
    def load(
        cls, directory: str | Path, *, budget: tuple[int, int] | None = None
    ) -> PreparedTraining:
        """Load all computations, or only one budget's winner for admission.

        A budget selection retains that budget's search outcomes and summary,
        but reads only its winning candidate/ordering and selected plan. Load
        without a budget to extend search over all prepared computations.
        """
        root = Path(directory)
        value = json.loads((root / "training.json").read_text())
        if value.get("schema") != artifact_schema("prepared_training"):
            raise ValueError("unsupported prepared training schema")

        def child(name: str) -> Path:
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("invalid prepared training artifact path")
            return root / name

        search = StepSearchReport.from_dict(value["search"])
        winner = None
        if budget is not None:
            if budget not in search.budgets:
                raise ValueError("budget was not searched in the prepared artifacts")
            winner = search.winner(*budget)
            search = replace(
                search,
                budgets=(budget,),
                points=tuple(
                    p
                    for p in search.points
                    if (p.execution_budget_bytes, p.spill_budget_bytes) == budget
                ),
            )
        candidates = {}
        for name, key in value["candidates"].items():
            if budget is not None and (winner is None or name != winner.candidate):
                continue
            directory = child(key)
            digest = hashlib.sha256(
                (directory / "manifest.json").read_bytes()
            ).hexdigest()
            if key != f"candidate-{digest}":
                raise ValueError("prepared training candidate checksum differs")
            candidates[name] = PreparedStep.load(
                directory, ordering=None if winner is None else winner.ordering
            )
        plans = {}
        for item in value["plans"]:
            plan_budget = tuple(item["budget"])
            if budget is not None and plan_budget != budget:
                continue
            plan = AnnotatedProgramPlan.from_json(child(item["file"]).read_text())
            if item["file"] != f"plan-{plan.digest}.json":
                raise ValueError("prepared training plan checksum differs")
            if plan_budget != (
                plan.memory_budgets.execution_bytes,
                plan.memory_budgets.spill_bytes,
            ):
                raise ValueError("prepared training plan budget differs")
            if plan_budget in plans:
                raise ValueError("duplicate prepared training budget")
            plans[plan_budget] = plan
        search = replace(search, winner_plans=plans)
        candidate_programs = {
            name: {(p.problem.digest, p.data_ordering) for p in candidate.programs}
            for name, candidate in candidates.items()
        }
        for plan_budget, plan in plans.items():
            winner = search.winner(*plan_budget)
            if winner is None or winner.candidate not in candidates:
                raise ValueError("prepared training plan has no matching candidate")
            if (plan.program.digest, winner.ordering) not in candidate_programs[
                winner.candidate
            ]:
                raise ValueError("prepared training plan differs from its candidate")
        if set(plans) != {b for b in search.budgets if search.winner(*b) is not None}:
            raise ValueError("prepared training winners and plans differ")
        return cls(candidates, search)


__all__ = ["PreparedTraining"]
