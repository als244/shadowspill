"""The ShadowSpill backend: state in a pinned host pool, every step planned.

The first launch of a run chooses its plan: ``plan_step_search`` finds the
fastest geometry at the budget (or prices the one given), and the backend
records the choice -- geometry, ordering, and the transfer bandwidths it was
planned against -- in the run's ``planning.json``. A later launch of the same
run plans that choice directly with ``plan_step``, which reuses the artifact
store's captures, compiled graphs and profiles instead of searching again.
"""

from __future__ import annotations

import functools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.planner.program_inputs import TransferBandwidths
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


@dataclass(frozen=True)
class PlanningChoice:
    """The geometry and ordering a run's step is planned for, and what it was
    priced against: the transfer bandwidths and the two budgets."""

    max_tokens_per_microbatch: int
    microbatches: int
    depth: int
    breadth: int
    reverse_breadth: bool
    pair_loss: bool
    transfer_bandwidths: dict[str, Any] | None
    execution_bytes: int
    spill_bytes: int


class ShadowSpill:
    """Plans each training step to fit ``execution_gib`` of the device, with the
    state in ``spill_gib`` of pinned host memory; the device pool is the step's
    budget. Evaluation plans its forward pass into the step's slab -- the two
    run in turn, so the pool holds the bytes once -- within
    ``eval_execution_gib`` when given, and the whole slab when not.

    ShadowSpill creates the optimizer's state in its pool before any step runs,
    each entry where the optimizer's own first step starts it -- moments at
    zero, say -- with the master copies a run's ``master_dtype`` asks for, and a
    resumed run's checkpoint replaces them.

    ``round_accumulation_once`` is ``plan_step``'s: a matrix multiply then adds
    its product into bf16 running gradients as it writes it, rounding the sum
    once where PyTorch rounds it twice -- more precise, and no longer the
    PyTorch backend's step bit for bit."""

    checkpoint_device = "cpu"  # mapped, and copied from there into the pool

    def __init__(
        self,
        execution_gib: float,
        spill_gib: float,
        eval_execution_gib: float | None = None,
        round_accumulation_once: bool = False,
    ) -> None:
        self.budget = (int(execution_gib * GIB), int(spill_gib * GIB))
        self.round_accumulation_once = round_accumulation_once
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
            self.planning = _read_planning(record, self.budget)
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
        self.eval_example = setup.data.examples(documents, 1, setup.max_seq_len)[0]

    def _search(self, setup: Setup) -> tuple[PlanningChoice, Any]:
        """Find the fastest geometry at the budget, or price the one given, and
        return it with the plan the search chose. Planning assumes whole
        documents of ``max_seq_len``, and the search counts in them; any packing
        of the same tokens runs on the plan it makes."""

        length = setup.max_seq_len
        pinned = setup.max_tokens_per_microbatch
        search = plan_step_search(
            self.module,
            example_microbatches=functools.partial(
                setup.data.examples, max_seq_len=length
            ),
            total_sequences_per_step=setup.max_tokens_per_step // length,
            sequence_length=length,
            budgets=[self.budget],
            min_tokens_per_microbatch=pinned,
            max_tokens_per_microbatch=pinned,
            progress=print,
            **self.common,
        )
        winner = search.winner(*self.budget)
        if winner is None:
            raise RuntimeError(
                f"no geometry plans within {self.budget[0] / GIB:g} GiB of the device"
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
        )
        return choice, search.winner_plans[self.budget]

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
            **self.common,
        )

    def step(self, microbatches: list[Microbatch], lr: float | None) -> list[float]:
        hyperparams = {} if lr is None else {"lr": lr}
        result = self.train_step(microbatches, hyperparams=hyperparams)
        return [loss.item() for loss in result.objectives]

    def synchronize(self) -> None:
        self.train_step.synchronize()  # its end-of-step writeback included

    def evaluate(self, microbatches: list[Microbatch]) -> list[float]:
        if self.forward is None:  # a second plan over the same weights and slab
            self.forward = plan_forward(
                self.module,
                example_inputs=self.eval_example,
                runtime=self.runtime,
                execution="device",
                spill="spill",
                execution_budget=self.eval_budget,
                share_slab_with=self.train_step,
                artifact_store=self.store,
            )
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


def _read_planning(record: Path, budget: tuple[int, int]) -> PlanningChoice:
    choice = PlanningChoice(**json.loads(record.read_text()))
    if (choice.execution_bytes, choice.spill_bytes) != budget:
        raise ValueError(
            f"{record} was planned at other budgets; a run keeps the geometry it "
            "started with, so remove the file to plan the run from scratch"
        )
    return choice
