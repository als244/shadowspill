"""One process of one case: what every engine's trainer shares.

A trainer is its system's setup and its step; this module is the rest.
``main`` reads the trainer's command line, ``--output`` and ``--case``, the
case ``baselines.common.launch`` validated and resolved for it. It writes
``rank-NNNNN/request.json`` as the process starts, hands the engine's
``train`` the ``Case`` and the step ``Clock``, and writes ``result.json`` as
it ends, or ``failure.json`` with the stage it failed in and whether it ran
out of device memory; the first rank then writes ``results.json`` with every
rank's measurements and ``row.json``, the row every engine shares.

The clock is the measurement contract. A step starts once every rank's
device is idle and the ranks have met, and ends when this rank's device is
idle again; the first ``warmup_steps`` steps are not measured, and at the
boundary the device peaks are reset and an nsys capture begins. An engine
whose loop is its own calls ``begin`` and ``end`` around each step; one
whose system owns the loop calls them from the system's step hooks.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import os
import resource
import sys
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from baselines.common.data import Corpus, synthetic_tokens
from baselines.common.provenance import (
    environment_identity,
    hardware_identity,
    source_identity,
)
from baselines.common.results import memory_peaks, summarize_cycles, write_json
from baselines.common.results import row as result_row

#: How ``useful_flops_per_step`` is counted, named in ``results.json``.
FLOPS_CONVENTION = "baselines.common.models.useful_flops_per_token"


@dataclasses.dataclass
class Outcome:
    """What an engine's ``train`` returns once its last step is done.

    ``useful_flops_per_step`` replaces the count the launcher resolved, for
    an engine that counts from the model it built; ``performance`` is false
    when the engine's own case is no throughput result. ``result`` adds to
    this rank's ``result.json`` and ``provenance`` to the row's;
    ``peaks`` are this rank's device peaks where PyTorch's allocator does
    not hold the device, and ``memory`` the row's memory from every rank's
    result in place of ``memory_peaks``.
    """

    useful_flops_per_step: float | None = None
    evidence: Any = None
    reference_checks: list | None = None
    performance: bool = True
    result: dict[str, Any] = dataclasses.field(default_factory=dict)
    provenance: dict[str, Any] = dataclasses.field(default_factory=dict)
    peaks: Mapping[str, int] | None = None
    memory: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None
    #: The geometry the engine ran, for one that chose it, such as a
    #: microbatch size its own search picked; the case's otherwise.
    geometry: Mapping[str, Any] | None = None


class Tokens:
    """This rank's tokens for every step, on the host or ``device``: one
    synthetic draw per data-parallel replica, the same at every step, or
    the replica's corpus window of the step."""

    def __init__(self, settings, geometry, vocabulary, *, replica, replicas, device):
        self.count = geometry.tokens_per_rank_per_step
        self.replica, self.replicas, self.device = replica, replicas, device
        self.corpus = None
        self.wrapped = 0
        if settings.data is not None:
            self.corpus = Corpus(settings.data)
            self.corpus.check_vocabulary(vocabulary)
            self.meta = self.corpus.meta
        else:
            self.meta = None
            draw = synthetic_tokens(
                vocabulary, self.count, seed=settings.seed, rank=replica
            )
            self.fixed = self.place(draw)

    def place(self, pair):
        if self.device is None:
            return pair
        return tuple(value.to(self.device) for value in pair)

    def __call__(self, step):
        """``(tokens, targets)``, each ``(1, tokens_per_rank_per_step)``."""
        if self.corpus is None:
            return self.fixed
        tokens, targets, self.wrapped = self.corpus.window(
            rank=self.replica,
            world_size=self.replicas,
            step=step,
            tokens=self.count,
        )
        return self.place((tokens, targets))


@dataclasses.dataclass
class Case:
    """One process's view of its case: the settings, where it writes, the
    rank, what the row will describe, and the request record it keeps
    current as setup learns more."""

    engine: str
    settings: Any
    output: Path
    rank: int
    world: int
    local_rank: int
    geometry: Any
    precision: Any
    description: dict[str, Any]
    size: str
    resolved: dict[str, Any]
    record: dict[str, Any]
    standalone: bool
    started: float
    stage: str = "distributed_initialization"
    tokens: Tokens | None = None

    @property
    def directory(self) -> Path:
        return self.output / f"rank-{self.rank:05d}"

    @property
    def device(self):
        import torch

        return torch.device("cuda", self.local_rank)

    def save(self) -> None:
        write_json(self.directory / "request.json", self.record)

    def data(self, vocabulary, *, device=None, replica=None, replicas=None) -> Tokens:
        """The tokens this rank trains on, recorded in the request; a
        replica is this rank unless the system shards a replica over
        several ranks."""

        self.tokens = Tokens(
            self.settings,
            self.geometry,
            vocabulary,
            replica=self.rank if replica is None else replica,
            replicas=self.world if replicas is None else replicas,
            device=device,
        )
        if self.tokens.meta is not None:
            self.record["corpus"] = self.tokens.meta
        return self.tokens

    def cap_allocator(self) -> None:
        """Hold PyTorch's allocator to ``allocator_cap_gib``, when set."""

        import torch

        cap = self.settings.allocator_cap_gib
        if cap is None:
            return
        total = torch.cuda.get_device_properties(self.device).total_memory
        fraction = cap * 2**30 / total
        if not 0 < fraction <= 1:
            raise ValueError("allocator cap must fit the physical device")
        torch.cuda.set_per_process_memory_fraction(fraction, self.device)


class Clock:
    """The step clock every engine measures with."""

    def __init__(self, case: Case, *, reset_peaks: bool = True) -> None:
        self.case = case
        self.reset_peaks = reset_peaks
        self.cycles: list[float] = []
        self.losses: list[float] = []
        self.head_losses: list[float] = []
        self.setup_seconds: float | None = None
        self.tick: float | None = None
        self.capturing = False
        self.capture_group = None
        self.closed = False

    @property
    def steps(self) -> int:
        return self.case.settings.warmup_steps + self.case.settings.steps

    def measured(self, step: int) -> bool:
        return step >= self.case.settings.warmup_steps

    def barrier(self) -> None:
        import torch.distributed as dist

        if self.case.world > 1 and dist.is_initialized():
            dist.barrier()

    def ready(self, note: str = "") -> None:
        """Setup is over: its seconds are recorded and the trainer says so."""

        if self.setup_seconds is not None:
            return
        self.setup_seconds = time.perf_counter() - self.case.started
        self.case.stage = "training"
        print(
            f"rank={self.case.rank} initialized seconds={self.setup_seconds:.3f} "
            f"{note + ' ' if note else ''}"
            f"accumulation={self.case.geometry.accumulation}",
            flush=True,
        )

    def begin(self, step: int) -> None:
        import torch

        self.ready()
        if step == self.case.settings.warmup_steps:
            torch.cuda.synchronize()
            self.barrier()
            if self.reset_peaks:
                torch.cuda.reset_peak_memory_stats()
            if self.case.settings.nsys_capture:
                import torch.distributed as dist

                if self.case.world > 1 and dist.is_initialized():
                    self.capture_group = dist.new_group(backend="gloo")
                self._capture_boundary(torch.cuda.cudart().cudaProfilerStart)
                # Nsight enables sibling-process NVTX collection asynchronously.
                # Settle before any timed range or further CUDA collective.
                time.sleep(1.0)
                if self.capture_group is not None:
                    dist.barrier(group=self.capture_group)
                self.capturing = True
        torch.cuda.synchronize()
        self.barrier()
        self.tick = time.perf_counter()
        torch.cuda.nvtx.range_push(
            f"baseline.{self.case.engine}/rank-{self.case.rank}/step-{step + 1}"
        )

    def end(self, step: int, loss, head_loss=None) -> float:
        """Close the step: its seconds, and this rank's share of the update's
        loss (the head loss alone for a mixture), recorded once measured; a
        rank that holds no loss (a pipeline stage before the last) records
        its time and none."""

        import torch

        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.tick
        if loss is not None:
            loss = float(loss)
            if not math.isfinite(loss):
                raise ValueError("training loss is not finite")
        head = loss if head_loss is None else float(head_loss)
        measured = self.measured(step)
        if measured:
            self.cycles.append(elapsed)
            self.losses.append(loss)
            self.head_losses.append(head)
        shown = "none" if loss is None else f"{loss:.6f}"
        print(
            f"rank={self.case.rank} step={step + 1} measured={measured} "
            f"seconds={elapsed:.6f} loss={shown}",
            flush=True,
        )
        return elapsed

    def close(self) -> None:
        """After the last step: synchronize; Nsight finishes when workers exit."""

        import torch

        if self.closed:
            return
        self.closed = True
        if self.capturing:
            torch.cuda.synchronize()
            # cudaProfilerStop may block on multi-process report generation.
            # The benchmark ends immediately after measured updates; let nsys
            # finalize at normal process-tree exit and release the GPUs first.
            self.capturing = False
        torch.cuda.synchronize()

    def _capture_boundary(self, action) -> None:
        """Keep peers off CUDA while rank zero starts process-tree capture."""
        import torch.distributed as dist

        if self.capture_group is not None:
            dist.barrier(group=self.capture_group)
        if self.case.rank == 0:
            action()
        if self.capture_group is not None:
            dist.barrier(group=self.capture_group)


def out_of_memory(error: BaseException) -> dict[str, Any]:
    """A failure's status: out of device memory, the frontier's boundary,
    or failed."""

    import torch

    oom = isinstance(error, torch.OutOfMemoryError)
    return {"status": "cuda_out_of_memory" if oom else "failed"}


def parse(engine, argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"One process of a {engine} case; started by"
        " baselines.common.launch under torchrun."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case", type=Path, required=True, help="the launcher's case.json"
    )
    return parser.parse_args(argv)


def open_case(engine, module, arguments, *, standalone) -> Case:
    """The case this process runs, as the launcher validated and resolved
    it."""

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    value = json.loads(arguments.case.read_text())
    if value.get("engine") != engine:
        raise ValueError(f"{arguments.case} is not a {engine} case")
    settings = module.Settings.from_dict(value["settings"])
    geometry = settings.geometry(world)
    description = value["description"]
    if description["geometry"] != geometry.as_dict():
        raise ValueError(
            f"{arguments.case} was resolved for another world than {world}"
        )
    differs = {
        name: os.environ.get(name)
        for name, wanted in value["environment"].items()
        if os.environ.get(name) != wanted
    }
    if differs:
        raise ValueError(
            f"this process's environment is not its case's ({value['environment']}):"
            f" it has {differs}; a case starts through baselines.common.launch"
        )
    precision = settings.precision()
    record = {
        **description,
        "output": str(arguments.output),
        "rank": rank,
        "world_size": world,
        "hardware": None,
        "resolved_precision": precision.resolved(),
        "microbatches_per_step": geometry.accumulation,
        "shadowspill": not standalone,
        "summary": value.get("model_summary"),
        "environment": value["environment"],
    }
    return Case(
        engine=engine,
        settings=settings,
        output=arguments.output,
        rank=rank,
        world=world,
        local_rank=local_rank,
        geometry=geometry,
        precision=precision,
        description=description,
        size=value["size"],
        resolved=value,
        record=record,
        standalone=standalone,
        started=time.perf_counter(),
    )


def finish(case: Case, clock: Clock, outcome: Outcome, source_paths) -> None:
    """``result.json`` on every rank; ``results.json`` and the row on the
    first."""

    import torch
    import torch.distributed as dist

    clock.close()
    if case.standalone:
        forbidden = [
            name
            for name in sys.modules
            if name == "shadowspill" or name.startswith("shadowspill.")
        ]
        if forbidden:
            raise RuntimeError(
                f"standalone baseline imported ShadowSpill: {forbidden[:5]}"
            )
    settings = case.settings
    performance = (
        outcome.performance
        and not settings.verify_reference
        and not settings.nsys_capture
        and case.size not in ("tiny", "numerical")
    )
    peaks = outcome.peaks or {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    result = {
        "status": "completed",
        "rank": case.rank,
        "cycles": clock.cycles,
        "losses": clock.losses,
        "head_losses": clock.head_losses,
        "precision_evidence": outcome.evidence,
        "corpus_wrapped": 0 if case.tokens is None else case.tokens.wrapped,
        **peaks,
        "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        **outcome.result,
        "setup_seconds": clock.setup_seconds,
        "total_wall_seconds": time.perf_counter() - case.started,
        "finished_at": datetime.datetime.now().astimezone().isoformat(),
        "qualified_against_reference": settings.verify_reference,
        "reference_checks": outcome.reference_checks,
        "performance_result": performance,
    }
    write_json(case.directory / "result.json", result)
    all_ranks = [result]
    if case.world > 1:
        all_ranks = [None] * case.world
        dist.all_gather_object(all_ranks, result)
    if case.rank != 0:
        return
    useful = outcome.useful_flops_per_step
    if useful is None:
        useful = case.resolved.get("useful_flops_per_step")
    summary = summarize_cycles(
        [row["cycles"] for row in all_ranks],
        global_tokens=case.geometry.tokens_per_step,
        useful_flops_per_step=useful,
    )
    summary["performance_result"] = performance
    summary["losses"] = [row["losses"] for row in all_ranks]
    summary["head_losses"] = [row["head_losses"] for row in all_ranks]
    write_json(
        case.output / "results.json",
        {
            **summary,
            "status": "completed",
            "request": case.record,
            "ranks": all_ranks,
            "useful_flops_per_step": useful,
            "flops_convention": FLOPS_CONVENTION,
        },
    )
    memory = (outcome.memory or memory_peaks)(all_ranks)
    write_json(
        case.output / "row.json",
        result_row(
            engine=case.engine,
            status="completed",
            **{
                **case.description,
                "geometry": dict(outcome.geometry or case.description["geometry"]),
            },
            hardware=case.record["hardware"],
            provenance={
                "source": source_identity(source_paths),
                "environment": environment_identity(sys.executable),
                "model_summary": case.record.get("summary"),
                "corpus": case.record.get("corpus"),
                "checkpoint": case.record.get("checkpoint"),
                **outcome.provenance,
            },
            measurement=summary,
            memory=memory,
        ),
    )


def main(
    engine: str,
    module,
    train: Callable[[Case, Clock], Outcome],
    argv=None,
    *,
    standalone: bool = True,
    device_first: bool = True,
    failure: Callable[[BaseException], dict[str, Any]] = out_of_memory,
) -> None:
    """Run one process of a case of ``engine``, whose settings module is
    ``module`` and whose ``train(case, clock)`` sets the system up and runs
    every step under the clock.

    ``standalone`` refuses a run that imported ShadowSpill;
    ``device_first`` selects the device, probes the hardware and caps
    PyTorch's allocator before ``train`` begins, for a system that does not
    install its own allocator first, and resets PyTorch's peaks at the
    warmup boundary; ``failure`` classifies an exception for
    ``failure.json``.
    """

    import torch
    import torch.distributed as dist

    arguments = parse(engine, argv)
    case = open_case(engine, module, arguments, standalone=standalone)
    torch.set_num_threads(case.settings.cpu_threads)
    if device_first:
        torch.cuda.set_device(case.device)
        case.record["hardware"] = hardware_identity()
    case.directory.mkdir(parents=True, exist_ok=False)
    case.save()
    clock = Clock(case, reset_peaks=device_first)
    try:
        if device_first:
            case.cap_allocator()
        outcome = train(case, clock)
        finish(case, clock, outcome, module.SOURCE_PATHS)
    except Exception as error:
        # What the engine recorded before failing, the device it probed
        # once its allocator was in place among it, joins the request.
        case.save()
        write_json(
            case.directory / "failure.json",
            {
                "rank": case.rank,
                "stage": case.stage,
                **failure(error),
                "exception_type": type(error).__name__,
                "error": traceback.format_exc(),
            },
        )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


__all__ = [
    "FLOPS_CONVENTION",
    "Case",
    "Clock",
    "Outcome",
    "Tokens",
    "finish",
    "main",
    "open_case",
    "out_of_memory",
    "parse",
]
