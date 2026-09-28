"""The ShadowSpill backend: state in a pinned host pool, every step planned.

The first launch of a run chooses its plan: ``plan_step_search`` finds the
fastest geometry at the budget within the bounds the backend names (or
prices the one the trainer gave), under the recompute shares and the walks
it was told to plan, and the backend records the choice -- geometry,
ordering, the search options, and the transfer bandwidths it was planned
against -- in the run's ``planning.json``, with the search's whole table
beside it in ``search.json``. A later launch of the same run plans that
choice directly with ``plan_step``, which reuses the artifact store's
captures, compiled graphs and profiles instead of searching again.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.planner import (
    SearchOptions,
    StepDataOrdering,
    named_resolution_options,
)
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.planner.search.algorithms.pressurefit import PressureFit
from shadowspill.planner.search.algorithms.pressurefit.options import (
    PressureFitOptions,
)
from shadowspill.pytorch import (
    Runtime,
    import_model_state,
    plan_forward,
    plan_step,
    plan_step_search,
    release_model_state,
)
from training.backends import GIB, Microbatch, Setup
from training.objectives import planned_objective

PLANNING_RECORD = "planning.json"
SEARCH_REPORT = "search.json"
#: Which walks the search tries per geometry: every depth x breadth factor
#: pair, or the depth-first walk alone.
ORDERINGS = ("factors", "depth-first")


@dataclass(frozen=True)
class PlanningChoice:
    """The geometry and ordering a run's step is planned for, what it was
    priced against -- the transfer bandwidths and the two budgets -- and what
    the search was told, which the replan asks the store for again."""

    max_tokens_per_microbatch: int
    microbatches: int
    depth: int
    breadth: int
    reverse_breadth: bool
    pair_loss: bool
    transfer_bandwidths: dict[str, Any] | None
    execution_bytes: int
    spill_bytes: int
    #: The recompute shares and the walks the search planned; a record from
    #: before they were written was planned under the library's defaults.
    resolution_options: list[str] = field(
        default_factory=lambda: list(named_resolution_options("quarters"))
    )
    orderings: str = "factors"
    #: The bounds the geometry was searched within, tokens a microbatch; a
    #: pinned geometry is both bounds equal, and ``None`` is open.
    planning_min_tokens_per_microbatch: int | None = None
    planning_max_tokens_per_microbatch: int | None = None


class ShadowSpill:
    """Plans each training step to fit ``execution_gib`` of the device, with the
    state in ``spill_gib`` of pinned host memory; the device pool is the step's
    budget. Evaluation's forward pass is planned with the step, right after
    it, into the step's slab -- the two run in turn, so the pool holds the
    bytes once -- within the step's own budget, the whole slab, unless
    ``eval_execution_gib`` names less.

    ShadowSpill creates the optimizer's state in its pool before any step runs,
    each entry where the optimizer's own first step starts it -- moments at
    zero, say -- with the master copies a run's ``master_dtype`` asks for, and a
    resumed run's checkpoint replaces them.

    ``round_accumulation_once`` is ``plan_step``'s: a matrix multiply then adds
    its product into bf16 running gradients as it writes it, rounding the sum
    once where PyTorch rounds it twice -- more precise, and no longer the
    PyTorch backend's step bit for bit.

    The geometry is the search's to choose within
    ``planning_min_tokens_per_microbatch`` and
    ``planning_max_tokens_per_microbatch``, tokens a microbatch: every split
    of the step whose microbatch falls between them is planned and the
    fastest runs; equal bounds pin one, a missing bound is open. Without
    either the trainer's ``max_tokens_per_microbatch`` pins the geometry, and
    without that too every split is searched. ``resolution_options`` are the
    shares of the flexible graph-pair groups the search plans recomputing --
    ``quarters``, ``eighths``, ``halves``, or exact fractions -- and
    ``orderings`` the walks it tries per geometry, every depth x breadth
    factor pair or the depth-first walk alone; both are part of the plan's
    identity and are recorded with the choice."""

    checkpoint_device = "cpu"  # mapped, and copied from there into the pool

    def __init__(
        self,
        execution_gib: float,
        spill_gib: float,
        eval_execution_gib: float | None = None,
        round_accumulation_once: bool = False,
        planning_min_tokens_per_microbatch: int | None = None,
        planning_max_tokens_per_microbatch: int | None = None,
        resolution_options: str | Sequence[str] = "quarters",
        orderings: str = "factors",
    ) -> None:
        self.budget = (int(execution_gib * GIB), int(spill_gib * GIB))
        self.round_accumulation_once = round_accumulation_once
        self.planning_bounds = _planning_bounds(
            planning_min_tokens_per_microbatch, planning_max_tokens_per_microbatch
        )
        self.resolution_options = list(named_resolution_options(resolution_options))
        if orderings not in ORDERINGS:
            raise ValueError(
                f"orderings is one of {', '.join(ORDERINGS)}, not {orderings!r}"
            )
        self.orderings = orderings
        # One policy for both planning phases: the search ranks under it and
        # the replan asks the store for the plan it promised under the same.
        self.search_options = SearchOptions(
            algorithm=PressureFit(
                PressureFitOptions(
                    resolution_options=tuple(
                        Fraction(share) for share in self.resolution_options
                    )
                )
            )
        )
        self.eval_budget = (
            None if eval_execution_gib is None else int(eval_execution_gib * GIB)
        )
        self.forward = None
        self.plan = None
        self.planning: PlanningChoice | None = None

    def setup(self, setup: Setup) -> None:
        # The runtime comes first: before any device allocation or model state.
        self.runtime = _runtime(*self.budget)
        self.store = setup.artifact_store
        torch.manual_seed(setup.seed)
        # Initialized on the host, as the PyTorch backend does, then moved to the pool.
        self.module = import_model_state(
            setup.module, runtime=self.runtime, pool="spill"
        )
        self.common = {
            "objective": planned_objective,
            "optimizer": functools.partial(setup.optimizer, **setup.optimizer_args),
            "hyperparams": setup.hyperparams,
            "master_dtype": setup.master_dtype,
            "grad_dtype": setup.grad_dtype,
            "round_accumulation_once": self.round_accumulation_once,
            "runtime": self.runtime,
            "execution": "device",
            "spill": "spill",
            "artifact_store": self.store,
        }
        record = setup.run_dir / PLANNING_RECORD
        incumbent = None
        if record.exists():
            self.planning = _read_planning(
                record, self.budget, self.resolution_options, self.orderings
            )
        else:
            self.planning, incumbent = self._search(setup)
            record.write_text(json.dumps(asdict(self.planning), indent=2) + "\n")
        self.geometry = (
            self.planning.max_tokens_per_microbatch,
            self.planning.microbatches,
        )
        documents = self.geometry[0] // setup.max_seq_len
        self.train_step = self._plan_step(
            setup.data.examples(documents, self.geometry[1], setup.max_seq_len),
            incumbent,
        )
        self.plan = self.train_step.plan_report.summary
        # Evaluation's forward, planned now: right after the step, into its
        # slab and over the same weights, so a run that cannot evaluate stops
        # before it trains rather than at its first evaluation.
        self.forward = plan_forward(
            self.module,
            example_inputs=setup.data.examples(documents, 1, setup.max_seq_len)[0],
            runtime=self.runtime,
            execution="device",
            spill="spill",
            execution_budget=self.eval_budget,
            share_slab_with=self.train_step,
            artifact_store=self.store,
        )

    def _search(self, setup: Setup) -> tuple[PlanningChoice, Any]:
        """Find the fastest geometry at the budget, or price the one given, and
        return it with the plan the search chose. Planning assumes whole
        documents of ``max_seq_len``, and the search counts in them; any packing
        of the same tokens runs on the plan it makes."""

        length = setup.max_seq_len
        low, high = self.planning_bounds
        if low is None and high is None:
            # No bounds of the backend's own: the trainer's microbatch pins
            # the geometry, and without one either every split is searched.
            low = high = setup.max_tokens_per_microbatch
        search = plan_step_search(
            self.module,
            example_microbatches=functools.partial(
                setup.data.examples, max_seq_len=length
            ),
            total_sequences_per_step=setup.max_tokens_per_step // length,
            sequence_length=length,
            budgets=[self.budget],
            min_tokens_per_microbatch=low,
            max_tokens_per_microbatch=high,
            orderings=self._orderings(),
            search_options=self.search_options,
            progress=print,
            **self.common,
        )
        # The whole table, not only the winner: which splits planned, at what
        # simulated step, is the evidence behind the choice the record keeps.
        search.save(setup.run_dir / SEARCH_REPORT)
        winner = search.winner(*self.budget)
        if winner is None:
            raise RuntimeError(
                f"no geometry {_within(low, high)} plans within"
                f" {self.budget[0] / GIB:g} GiB of the device"
            )
        lanes = search.planned_lanes
        choice = PlanningChoice(
            max_tokens_per_microbatch=winner.sequences_per_microbatch * length,
            microbatches=winner.accumulation_count,
            depth=winner.ordering.depth,
            breadth=winner.ordering.breadth,
            reverse_breadth=winner.ordering.reverse_breadth,
            pair_loss=winner.ordering.pair_loss,
            transfer_bandwidths=None if lanes is None else lanes.to_dict(),
            execution_bytes=self.budget[0],
            spill_bytes=self.budget[1],
            resolution_options=list(self.resolution_options),
            orderings=self.orderings,
            planning_min_tokens_per_microbatch=low,
            planning_max_tokens_per_microbatch=high,
        )
        return choice, search.winner_plans[self.budget]

    def _orderings(
        self,
    ) -> Callable[[int], Sequence[StepDataOrdering]] | None:
        """What ``plan_step_search`` tries per geometry: ``None`` is its
        default, every factor pair."""

        if self.orderings == "factors":
            return None
        return lambda accumulation: (StepDataOrdering.depth_first(accumulation),)

    def _plan_step(self, examples: list[Microbatch], incumbent: Any) -> Any:
        """Plan the step the choice describes; with the search's plan as
        ``incumbent``, execute that one, priced against the same lanes."""

        choice = self.planning
        bandwidths = choice.transfer_bandwidths
        return plan_step(
            self.module,
            example_inputs=examples,
            execution_budget=self.budget[0],
            spill_budget=self.budget[1],
            depth=choice.depth,
            breadth=choice.breadth,
            reverse_breadth=choice.reverse_breadth,
            pair_loss=choice.pair_loss,
            incumbent=incumbent,
            transfer_bandwidths=(
                None
                if bandwidths is None
                else TransferBandwidths.from_value(bandwidths)
            ),
            # The search's own options, so the store answers with the plan
            # the search promised rather than missing and searching again.
            search_options=self.search_options,
            **self.common,
        )

    def step(self, microbatches: list[Microbatch], lr: float | None) -> list[float]:
        hyperparams = {} if lr is None else {"lr": lr}
        result = self.train_step(microbatches, hyperparams=hyperparams)
        return [loss.item() for loss in result.objectives]

    def synchronize(self) -> None:
        self.train_step.synchronize()  # its end-of-step writeback included

    def evaluate(self, microbatches: list[Microbatch]) -> list[float]:
        return [self.forward(microbatch).item() for microbatch in microbatches]

    def device_peak_gib(self) -> float:
        return self.runtime.pool_statistics("device").peak_allocated_bytes / GIB

    def save(self, path: Path) -> None:
        self.train_step.save(path)  # straight from the pool, each weight once

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.train_step.load_state_dict(state)

    def close(self) -> None:
        if self.forward is not None:
            self.forward.close()
        self.train_step.close()
        release_model_state(self.module, runtime=self.runtime)
        self.runtime.close()


def _runtime(device_bytes: int, spill_bytes: int) -> Runtime:
    return Runtime(
        pools={
            "device": device(physical_capacity=device_bytes),
            "spill": pinned_host(capacity=spill_bytes),
        },
        routes={
            "fetch": transfer_route(source="spill", destination="device"),
            "evict": transfer_route(source="device", destination="spill"),
        },
    )


def _read_planning(
    record: Path,
    budget: tuple[int, int],
    resolution_options: Sequence[str],
    orderings: str,
) -> PlanningChoice:
    choice = PlanningChoice(**json.loads(record.read_text()))
    if (choice.execution_bytes, choice.spill_bytes) != budget:
        raise ValueError(
            f"{record} was planned at other budgets; a run keeps the geometry it "
            "started with, so remove the file to plan the run from scratch"
        )
    if (list(choice.resolution_options), choice.orderings) != (
        list(resolution_options),
        orderings,
    ):
        raise ValueError(
            f"{record} was planned under other search options"
            f" ({', '.join(choice.resolution_options)}; {choice.orderings});"
            " a run keeps the plan it started with, so remove the file to plan"
            " the run from scratch"
        )
    return choice


def _planning_bounds(
    low: int | None, high: int | None
) -> tuple[int | None, int | None]:
    for name, value in (
        ("planning_min_tokens_per_microbatch", low),
        ("planning_max_tokens_per_microbatch", high),
    ):
        if value is not None and (
            isinstance(value, bool) or type(value) is not int or value < 1
        ):
            raise ValueError(
                f"{name} must be a positive number of tokens, not {value!r}"
            )
    if low is not None and high is not None and low > high:
        raise ValueError(
            f"planning_min_tokens_per_microbatch {low} exceeds"
            f" planning_max_tokens_per_microbatch {high}"
        )
    return low, high


def _within(low: int | None, high: int | None) -> str:
    if low is not None and low == high:
        return f"of {low} tokens a microbatch"
    if low is None and high is None:
        return "at all"
    if low is None:
        return f"of at most {high} tokens a microbatch"
    if high is None:
        return f"of at least {low} tokens a microbatch"
    return f"between {low} and {high} tokens a microbatch"
