"""What one qualification case asked for, and what names it.

Both arms are built from the same request: the compiled reference and the
planned run. The identity is a digest of everything that decides what they
compute, so a reference recorded for another request is refused rather than
compared against.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from shadowspill.memory import SpillPool
from shadowspill.planner import StepDataOrdering
from shadowspill.store import StoreMode
from workloads.numerical import ModelImplementation, NumericalCase, build_case

#: The reference arm: the same step, fully compiled, without ShadowSpill.
REFERENCE_EXECUTION = "torch.compile.inductor.fullgraph"


@dataclass(frozen=True, slots=True)
class CaseRequest:
    """The case both arms build: the model, its data, and how it steps."""

    family: str
    model_implementation: ModelImplementation
    seed: int
    model_config: dict[str, Any]
    data_geometry: list[dict[str, Any]] | None
    case_factory: str | None
    case_options: dict[str, Any]
    optimizer_ordering: Literal["stage_interleaved", "tail"] = "stage_interleaved"
    data_ordering: str | None = None
    steps: int = 5

    def identity(self) -> str:
        """Digest everything that decides what the two arms compute.

        The walk is deliberately not part of it: the reference is the same
        step fully torch-compiled without ShadowSpill, which every walk of
        the step must reproduce.
        """

        payload = {
            "reference_execution": REFERENCE_EXECUTION,
            "model_name": self.family,
            "model_implementation": self.model_implementation,
            "seed": self.seed,
            "model_config": self.model_config,
            "data_geometry": self.data_geometry,
            "case_factory": self.case_factory,
            "case_options": self.case_options,
            "optimizer_ordering": self.optimizer_ordering,
            "steps": self.steps,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def build(self) -> NumericalCase:
        """Build the case itself, as either arm runs it."""

        return build_case(
            self.family,
            model_implementation=self.model_implementation,
            seed=self.seed,
            model_config=self.model_config,
            data_geometry=self.data_geometry,
            case_factory=self.case_factory,
            case_options=self.case_options,
        )

    def data_ordering_arguments(self) -> dict[str, Any]:
        """The plan_step keyword arguments this request's walk asks for."""

        if self.data_ordering is None:
            return {}
        ordering = StepDataOrdering.from_label(self.data_ordering)
        return {
            "depth": ordering.depth,
            "breadth": ordering.breadth,
            "reverse_breadth": ordering.reverse_breadth,
            "pair_loss": ordering.pair_loss,
        }


@dataclass(frozen=True, slots=True)
class PlannedRequest:
    """One planned case: the request, where its evidence goes, what it may use."""

    case: CaseRequest
    reference_path: Path
    result_path: Path
    device_budget: int
    checkpoint_step: int
    require_pressure: bool
    artifact_store: Path | None = None
    build_store: Path | None = None
    plan_store: Path | None = None
    profiling_metadata: list[object] | None = None
    build_store_mode: StoreMode = "contribute"
    plan_store_mode: StoreMode = "contribute"
    export_bypass_key: str | None = None
    detailed_artifacts: bool = False
    #: Where this case spills to. ``None`` means the pinned-host pool every
    #: local case uses; the remote gate supplies a pool on another machine.
    #: It is carried here rather than read from the environment because it is
    #: a property of the case, and a case's evidence should say what it ran on.
    spill_pool: SpillPool | None = None


def workload_metadata_for(case: Any, supplied: list[object] | None) -> list[object]:
    """Return explicit value-sensitive workload classes for task profiling.

    The built-in qualification cases place packed sequence lengths in the third
    microbatch position.  Custom cases can provide an arbitrary JSON list with
    ``--profiling-metadata`` instead of relying on that convenience.
    """

    if supplied is not None:
        if len(supplied) != len(case.microbatches):
            raise ValueError("profiling metadata must have one entry per microbatch")
        return supplied
    result: list[object] = []
    for microbatch in case.microbatches:
        sequence_lengths = microbatch[2] if len(microbatch) > 2 else None
        if isinstance(sequence_lengths, (list, tuple)) and all(
            isinstance(value, int) for value in sequence_lengths
        ):
            result.append({"sequence_lengths": list(sequence_lengths)})
        else:
            result.append(None)
    return result
