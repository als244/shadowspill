"""Portable phase controls; the engine remains the only builder and planner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from .._records import Record, Value, integer, mapping, number, plain, record, text


@dataclass(frozen=True)
class Policy(Record):
    """An installed build policy and its settings.

    The service runs only installed policies; no callable is uploaded.

    Parameters
    ----------
    definition : str
        Exact versioned ID of an installed policy, such as
        ``"shadowspill/text_chunks@1"``. Its kind must match the build option
        it is given for.
    config : mapping, default {}
        Settings the policy accepts. Setup checks them and fills omitted ones
        with the policy's defaults.
    """

    definition: str
    config: Mapping[str, Value] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text(self.definition, "policy definition")
        object.__setattr__(self, "config", mapping(self.config, "policy config"))


class _Omitted:
    pass


_OMITTED = _Omitted()


@dataclass(frozen=True, init=False)
class BuildOptions(Record):
    """Fixed build policy choices.

    An option left out inherits the preparation preset, then the engine
    default; an explicit None is kept. Each value is a `Policy` or a mapping
    of its fields.

    Parameters
    ----------
    partition : Policy or mapping, optional
        Policy that places checkpoint-stage boundaries. The engine default is
        ``Policy("shadowspill/partition@1", {"mode": "auto"})``, which also
        accepts ``"whole"`` mode. None is refused during setup.
    chunking : Policy or mapping or None, optional
        Policy that splits execution regions into chunks. The engine default,
        None, leaves every computation unchunked.
    graph_pair_policy : Policy or mapping or None, optional
        Retention policy: what each stage's save and recompute variants keep
        for the backward pass. The engine default, None, uses cost-input
        retention.

    Attributes
    ----------
    values : mapping
        The options given, by name, each as a policy record or None.
    """

    values: Mapping[str, Value]

    def __init__(
        self,
        *,
        partition: Policy | Mapping[str, Any] | _Omitted | None = _OMITTED,
        chunking: Policy | Mapping[str, Any] | _Omitted | None = _OMITTED,
        graph_pair_policy: Policy | Mapping[str, Any] | _Omitted | None = _OMITTED,
    ) -> None:
        values = {}
        for name, value in (
            ("partition", partition),
            ("chunking", chunking),
            ("graph_pair_policy", graph_pair_policy),
        ):
            if value is not _OMITTED:
                values[name] = (
                    None if value is None else record(Policy, value, name).to_dict()
                )
        object.__setattr__(self, "values", mapping(values, "build options"))

    def to_dict(self) -> dict[str, Any]:
        """Return the given options as plain containers.

        Returns
        -------
        dict
            Each given option by name, as a policy record or None.
        """
        return dict(plain(self.values))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        """Rebuild options from a `to_dict` record.

        Parameters
        ----------
        value : mapping
            Options by name: any of ``partition``, ``chunking`` and
            ``graph_pair_policy``.

        Returns
        -------
        BuildOptions
            The options.

        Raises
        ------
        TypeError
            If ``value`` is not a mapping or names an unknown option.
        """
        if not isinstance(value, Mapping):
            raise TypeError("build options require an object")
        return cls(**value)


def positive_choices(value: Sequence[int] | None, name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{name} must be a nonempty sequence")
    choices = tuple(integer(v, name, minimum=1) for v in value)
    if len(set(choices)) != len(choices):
        raise ValueError(f"{name} must not contain duplicates")
    return choices


@dataclass(frozen=True)
class BuildCase(Record):
    """One complete build case: a microbatch size and its policies.

    Use cases when a policy depends on the microbatch size, such as a chunk
    policy sized to the buffer.

    Parameters
    ----------
    microbatch_size : int
        Physical microbatch size, at least one, in the preset's units: tokens
        for packed text, records for stacked inputs.
    build : BuildOptions, default BuildOptions()
        Policy choices for this case. Options left out inherit the fixed
        choices; none may repeat a choice `Preparation` ``build`` fixes.
    max_datums : int, optional
        Maximum number of records in one physical buffer, at least one. None
        uses the preset's choice; the stacked preset accepts only
        ``microbatch_size``.
    """

    microbatch_size: int
    build: BuildOptions = field(default_factory=BuildOptions)
    max_datums: int | None = None

    def __post_init__(self) -> None:
        integer(self.microbatch_size, "microbatch_size", minimum=1)
        if self.max_datums is not None:
            integer(self.max_datums, "max_datums", minimum=1)
        object.__setattr__(self, "build", record(BuildOptions, self.build, "build"))


@dataclass(frozen=True)
class BuildCandidates(Record):
    """Build cases for preparation to compare.

    Give candidate axes, whose cross product forms the cases, or complete
    `BuildCase` records, not both. An axis left out inherits the preparation
    preset, unless `Preparation` ``build`` fixes that choice. A preset that
    supplies complete cases is overridden only by ``cases``. Each sequence
    given is nonempty and without duplicates; at most 256 cases result.

    Parameters
    ----------
    microbatch_sizes : sequence of int, optional
        Candidate physical microbatch sizes, each at least one, in the
        preset's units: tokens for packed text, records for stacked inputs.
    chunking : sequence of Policy or None, optional
        Candidate chunking policies. None among them leaves every
        computation unchunked.
    graph_pair_policies : sequence of Policy or None, optional
        Candidate retention policies. None among them uses cost-input
        retention.
    cases : sequence of BuildCase, optional
        Complete build cases, in place of the axes.

    Raises
    ------
    ValueError
        If a sequence is empty or repeats a choice, or ``cases`` is given
        with an axis.
    """

    microbatch_sizes: Sequence[int] | None = None
    chunking: Sequence[Policy | None] | None = None
    graph_pair_policies: Sequence[Policy | None] | None = None
    cases: Sequence[BuildCase] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "microbatch_sizes",
            positive_choices(self.microbatch_sizes, "microbatch_sizes"),
        )
        for name in ("chunking", "graph_pair_policies", "cases"):
            values = getattr(self, name)
            if values is None:
                continue
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"{name} must be a nonempty sequence")
            kind = BuildCase if name == "cases" else Policy
            choices = tuple(
                None if value is None and name != "cases" else record(kind, value, name)
                for value in values
            )
            if any(v in choices[:i] for i, v in enumerate(choices)):
                raise ValueError(f"duplicate {name} choices")
            object.__setattr__(self, name, choices)
        if self.cases is not None and any(
            value is not None
            for value in (
                self.microbatch_sizes,
                self.chunking,
                self.graph_pair_policies,
            )
        ):
            raise ValueError("complete build cases cannot be mixed with candidate axes")


@dataclass(frozen=True)
class ProfilingOptions(Record):
    """Bounds on how each task is timed during preparation.

    Time floors count device time; wall-time limits bound further runs.
    Timing stops once the minimum count, the device-time floor and both
    stability checks are met, or at a wall-time or count limit. Counts are at
    least one; durations and thresholds are at least zero.

    Parameters
    ----------
    conditioning_seconds : float, default 0.25
        Device time, in seconds, to run a task before timing it. Zero
        disables this floor.
    conditioning_wall_seconds : float, default 3.0
        Wall-time limit, in seconds, on conditioning runs.
    conditioning_max_samples : int, default 32
        Maximum conditioning runs, even if the time floor is unmet.
    minimum_samples : int, default 3
        Timing samples always taken.
    maximum_samples : int, default 128
        Maximum timing samples, at least ``minimum_samples``.
    measurement_seconds : float, default 0.18
        Device time, in seconds, to accumulate across timing samples. Zero
        disables this floor.
    measurement_wall_seconds : float, default 2.0
        Wall-time limit, in seconds, on timing samples after the minimum
        count.
    relative_mad_threshold : float, default 0.03
        Largest median absolute deviation, as a fraction of the median, at
        which samples are stable.
    half_drift_threshold : float, default 0.03
        Largest difference between the medians of the first and second half
        of the samples, as a fraction of the overall median, at which samples
        are stable.
    """

    conditioning_seconds: float = 0.25
    conditioning_wall_seconds: float = 3.0
    conditioning_max_samples: int = 32
    minimum_samples: int = 3
    maximum_samples: int = 128
    measurement_seconds: float = 0.18
    measurement_wall_seconds: float = 2.0
    relative_mad_threshold: float = 0.03
    half_drift_threshold: float = 0.03

    def __post_init__(self) -> None:
        counts = {"conditioning_max_samples", "minimum_samples", "maximum_samples"}
        for name, value in self.to_dict().items():
            if name in counts:
                integer(value, name, minimum=1)
            else:
                number(value, name, minimum=0)
        if self.maximum_samples < self.minimum_samples:
            raise ValueError("maximum_samples must be >= minimum_samples")


@dataclass(frozen=True)
class PlanningOptions(Record):
    """How planning searches for each worker's Plan.

    A field left out inherits the preparation preset, then the engine
    default.

    Parameters
    ----------
    microbatch_counts : sequence of int, optional
        Candidate numbers of microbatches per update on each worker, each at
        least one and without duplicates. Schedules whose padded size exceeds
        the batch limits are rejected. The engine default is ``(1,)``.
    search : mapping, optional
        Complete search settings, with ``"generic"`` and ``"algorithm"``
        entries in the engine's ``SearchOptions`` record form; a partial or
        extended record is refused. The engine default is its standard
        search. A preview's ``config.preparation`` shows the resolved record.
    workers : int, optional
        Threads the search uses, at least zero. Zero uses every available
        logical CPU and one searches serially. It changes how long planning
        takes, not its result. The engine default is zero.
    """

    microbatch_counts: Sequence[int] | None = None
    search: Mapping[str, Value] | None = None
    workers: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "microbatch_counts",
            positive_choices(self.microbatch_counts, "microbatch_counts"),
        )
        if self.search is not None:
            # SearchOptions.to_dict() is the existing engine's portable question.
            # Its operational worker count is supplied separately here.
            object.__setattr__(self, "search", mapping(self.search, "search"))
        if self.workers is not None:
            integer(self.workers, "workers")


@dataclass(frozen=True)
class Preparation(Record):
    """How the service builds, measures and plans a run before training.

    A setting left out inherits the installed preparation preset, then the
    engine default. A preview's ``config.preparation`` reports every resolved
    choice and its origin.

    Parameters
    ----------
    build : BuildOptions, default BuildOptions()
        Policy choices fixed for every build case.
    candidates : BuildCandidates, default BuildCandidates()
        Build cases to prepare and compare.
    profiling : ProfilingOptions, optional
        Task timing bounds. None inherits the preset, then the
        `ProfilingOptions` defaults.
    planning : PlanningOptions, default PlanningOptions()
        Plan search settings.
    preset : str, optional
        Exact ID of the installed preparation preset, such as
        ``"shadowspill/stacked@1"``. None uses the preset the model
        definition names, or ``"shadowspill/stacked@1"`` if it names none.
    reuse : mapping, default {}
        Earlier measurements to reuse, as
        `PreparationHandle.reuse
        <shadowspill_client.preparation.PreparationHandle.reuse>` gives them
        for a completed preparation the caller owns. A case with a matching
        capture reuses its build, and its measurements when the profiling
        settings and headroom also match; other cases are prepared again.
    external_headroom_bytes : int, optional
        Device memory, in bytes, reserved on each worker outside its
        execution pool for allocations the Plan does not manage. At least
        zero; it counts toward the `Memory` device budget. None inherits the
        preset, then 512 MiB.

    Raises
    ------
    ValueError
        If a choice fixed in ``build`` is also a candidate axis or is
        repeated in a build case.
    """

    build: BuildOptions = field(default_factory=BuildOptions)
    candidates: BuildCandidates = field(default_factory=BuildCandidates)
    profiling: ProfilingOptions | None = None
    planning: PlanningOptions = field(default_factory=PlanningOptions)
    preset: str | None = None
    reuse: Mapping[str, Value] = field(default_factory=dict)
    external_headroom_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.external_headroom_bytes is not None:
            integer(self.external_headroom_bytes, "external_headroom_bytes")
        for name, kind in (
            ("build", BuildOptions),
            ("candidates", BuildCandidates),
            ("planning", PlanningOptions),
        ):
            object.__setattr__(self, name, record(kind, getattr(self, name), name))
        if self.profiling is not None:
            object.__setattr__(
                self, "profiling", record(ProfilingOptions, self.profiling, "profiling")
            )
        if self.preset is not None:
            text(self.preset, "preset")
        object.__setattr__(self, "reuse", mapping(self.reuse, "reuse"))
        fixed = self.build.values.keys()
        for axis, option in (
            ("chunking", "chunking"),
            ("graph_pair_policies", "graph_pair_policy"),
        ):
            if getattr(self.candidates, axis) is not None and option in fixed:
                raise ValueError(f"{option} is both a fixed value and a candidate axis")
        for case in self.candidates.cases or ():
            if overlap := fixed & case.build.values.keys():
                raise ValueError(f"build case repeats fixed choices: {sorted(overlap)}")
