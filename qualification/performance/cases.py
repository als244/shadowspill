"""Qualification budgets, geometries and measured performance authorities."""

from __future__ import annotations

from dataclasses import dataclass, fields

from workloads.full_model import FullModelSpec, throughput_spec
from workloads.providers import ModelImplementation

_GIB = 1 << 30

#: Throughput recorded on the RTX 5090 configuration below, in tokens/second.
#: The performance gate applies its 0.95 floor only when the hardware and
#: configuration match (see qualification.performance.baseline). Other
#: cells still report throughput and run their runtime/budget/simulator checks.
#:
#: Each is the median of three consecutive matrix runs on 2026-09-01 with
#: first-use initial ordering (5ae17b7), on an idle RTX 5090 under the
#: standard probe (no checkpoint, warm step, three groups of four steps):
#:
#:     mlops_llama3    3423.5   3441.4   3410.2
#:     mlops_qwen35    3026.6   3057.1   3015.8
#:     mlops_olmoe    13907.1  14085.9  13897.6
#:
#: Run-to-run spread is now up to 1.4%, dominated by which schedule the
#: planner draws rather than by measurement jitter, so the 5% margin is
#: roughly 3.5x the spread. Re-measure and update these deliberately when a
#: change is meant to move throughput.
#:
#: These replace entries measured on 2026-08-29 at a54da6c, which were
#: 3254.9, 2897.5 and 12807.8. The whole rise (+5.2%, +4.5%, +8.6%) is
#: first-use ordering of the initial placement batch: the opening restore
#: no longer strands a first-task input at the end of the FIFO fetch
#: queue. See docs/internal/investigations/step-prologue-and-terminal-tail.md.
_REGRESSION_TOKENS_PER_SECOND = {
    "mlops_llama3": 3_423.5,
    "mlops_qwen35": 3_026.6,
    "mlops_olmoe": 13_907.1,
}

#: The same floors for the same cells with the spill pool on a peer, in
#: tokens per second: what the `remote_perf` gate judges against, at the same
#: 0.95 margin. A peer's pool is reached over a 25 Gb/s link against about
#: 25 GB/s to pinned host memory, so these sit far below the local floors and
#: the local floors say nothing about a remote run.
#:
#: Each is the median of one matrix run on 2026-09-21 at af39235b with the
#: remote lane rewritten around the direct path, on an idle RTX 5090 with the
#: pool on tubingen (112 GiB) under the standard probe (no checkpoint, warm
#: step, three groups of four steps). The run three days earlier read within
#: 0.25 % of these in every cell, so the margin is twenty times the spread.
#: Re-measure and update these deliberately when a change is meant to move
#: remote throughput.
_REMOTE_REGRESSION_TOKENS_PER_SECOND = {
    "mlops_llama3": 584.4,
    "mlops_qwen35": 777.0,
    "mlops_olmoe": 1_934.2,
}

#: What the predecessor `dataflow` system measured on the same geometry, in
#: tokens per second. ShadowSpill replaces that system, so these are a parity
#: target rather than a regression floor: the harness reports the ratio and
#: never fails a cell on it.
#:
#: Source: `dataflow` at e04b1454, qualification runs of 2026-08-08 and
#: 2026-08-09, archived at combating_fragmentation/experiments/
#: E004-recompute-refinement/archive_INDEX.json. The geometry matches this
#: manifest exactly - sequence 1024, 65,536 tokens per step, 16 GiB execution
#: budget - and the transfer bandwidths agree within 3%, so the comparison is
#: like for like.
#:
#: ShadowSpill measures 88-90% of these as of 2026-08-23. That gap is the open
#: plan-quality item, and it is the reason these are kept: re-basing them onto
#: current numbers would erase the only standing measure of it.
_PREDECESSOR_TOKENS_PER_SECOND = {
    "mlops_llama3": 3_669.2969982952136,
    "mlops_qwen35": 3_316.344617868151,
    "mlops_olmoe": 15_654.904932252315,
}


@dataclass(frozen=True, slots=True)
class FullModelManifest(FullModelSpec):
    """A workload plus the qualification policy used to judge its execution."""

    device_physical_capacity_bytes: int = 16 << 30
    spill_budget_bytes: int = 112 << 30
    regression_tokens_per_second: float | None = None
    remote_regression_tokens_per_second: float | None = None
    predecessor_tokens_per_second: float | None = None
    external_headroom_bytes: int = 512 << 20
    reject_overbudget: bool = False


def _manifest(
    family: str,
    implementation: ModelImplementation,
) -> FullModelManifest:
    spec = throughput_spec(family, implementation)
    return FullModelManifest(
        **{field.name: getattr(spec, field.name) for field in fields(spec)},
        device_physical_capacity_bytes=16 * _GIB,
        spill_budget_bytes=112 * _GIB,
        regression_tokens_per_second=_REGRESSION_TOKENS_PER_SECOND.get(
            f"{implementation}_{family}"
        ),
        remote_regression_tokens_per_second=_REMOTE_REGRESSION_TOKENS_PER_SECOND.get(
            f"{implementation}_{family}"
        ),
        predecessor_tokens_per_second=_PREDECESSOR_TOKENS_PER_SECOND.get(
            f"{implementation}_{family}"
        ),
    )


def manifests() -> tuple[FullModelManifest, ...]:
    """Return the five accepted provider cells in stable execution order."""

    return (
        _manifest("llama3", "mlops"),
        _manifest("qwen35", "mlops"),
        _manifest("olmoe", "mlops"),
        _manifest("llama3", "pytorch"),
        _manifest("qwen35", "pytorch"),
    )


def manifest_for(family: str, implementation: ModelImplementation) -> FullModelManifest:
    for item in manifests():
        if item.family == family and item.implementation == implementation:
            return item
    raise ValueError(f"unknown full-model cell {(family, implementation)!r}")
