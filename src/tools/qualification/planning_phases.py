"""How one planning call's time divides, and how to say so in one line.

A planning call reports what each of its phases cost. A summary that names
a few of them and stops cannot be read against its own total: when work
moves into a phase nobody named, the named ones shrink and the line looks
like time went missing. So every phase here is either named or counted in
``other``, and the parts always sum to the whole.
"""

from __future__ import annotations

from collections.abc import Mapping

#: Each reported bucket and the phases it is made of, in the order planning
#: reaches them. Where a bucket lists two names they are alternatives, not
#: parts: the profiler replaces its own enclosing intervals with the work
#: classes it measured, so exactly one of each pair is ever reported. That
#: is what lets every bucket simply sum what it finds.
_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lowering_aot", ("capture_lowering",)),
    ("optimizer_capture", ("optimizer_capture",)),
    ("saved_value_resolution", ("saved_value_resolution",)),
    (
        "compiled_entrypoint_construction",
        ("compiled_entrypoint_construction", "compilation"),
    ),
    ("profiling", ("unique_stage_warmup_profiling", "structural_profiling")),
    ("cached_entrypoint_warmup", ("cached_entrypoint_warmup",)),
    (
        "profile_cache_and_entrypoint_orchestration",
        ("profile_cache_and_entrypoint_orchestration",),
    ),
    ("canonical_program_lowering", ("program_lowering",)),
    ("callable_construction", ("callable_construction",)),
    ("search", ("search",)),
    (
        "physical_admission",
        ("admission_facts", "spill_admission", "slab_admission"),
    ),
)

#: What a one-line summary says, and in what order. The rest of the
#: breakdown stays in the record, where a reader can go looking.
_ANNOUNCED = (
    "lowering_aot",
    "optimizer_capture",
    "saved_value_resolution",
    "compiled_entrypoint_construction",
    "profiling",
    "search",
)


def planning_breakdown(
    phase_seconds: Mapping[str, float], *, planning_seconds: float
) -> dict[str, float]:
    """Return non-overlapping planning buckets that sum to the whole call."""

    breakdown = {
        bucket: sum(phase_seconds.get(name, 0.0) for name in phases)
        for bucket, phases in _BUCKETS
    }
    breakdown["other"] = max(0.0, planning_seconds - sum(breakdown.values()))
    breakdown["total"] = planning_seconds
    return breakdown


def planning_summary(label: str, breakdown: Mapping[str, float]) -> str:
    """One line naming where a planning call's time went, remainder included.

    ``other`` here is everything this line did not name, not the breakdown's
    own remainder, so the line adds up to its own total whichever buckets it
    chooses to announce.
    """

    parts = " ".join(f"{name}={breakdown.get(name, 0.0):.3f}s" for name in _ANNOUNCED)
    total = breakdown["total"]
    named = sum(breakdown.get(name, 0.0) for name in _ANNOUNCED)
    other = max(0.0, total - named)
    return f"planned {label}: total={total:.3f}s {parts} other={other:.3f}s"


__all__ = ["planning_breakdown", "planning_summary"]
