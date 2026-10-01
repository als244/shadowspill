"""Optional logger helpers; the trainer accepts any callable(record)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from ._reporting import RecordExchange, Reducer, sum_rank_records
from .observations import MetricTable


class Wandb:
    """Log at the actual completed update count, retaining W&B system telemetry.

    Importing this module does not import or start W&B. Construct the logger
    explicitly and close it after the run. Additional init options, including
    project/entity, run identity and explicit resume behavior, go to W&B.
    """

    def __init__(self, *, project: str, run_dir: str | Path, **options: Any) -> None:
        import wandb

        root = Path(run_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(project=project, dir=str(root), **options)
        self.run.define_metric("step")
        self.run.define_metric("*", step_metric="step")

    def __call__(self, record: dict[str, Any]) -> None:
        self.run.log(dict(record), step=int(record["step"]), commit=False)

    def table(self, step: int, name: str, table: MetricTable) -> None:
        import wandb

        self.run.log(
            {
                "step": step,
                name: wandb.Table(
                    columns=list(table.columns),
                    data=list(table.rows),
                ),
            },
            step=step,
            commit=False,
        )

    def close(self, *, exit_code: int = 0) -> None:
        self.run.finish(exit_code=exit_code)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(exit_code=0 if exception_type is None else 1)


class DistributedLogger:
    """Keep per-rank W&B runs and asynchronously combine completed CPU records.

    Pass as Trainer.fit's logger. Trainer retains per-rank console/JSONL output;
    this helper adds aggregate console/JSONL output and optional grouped W&B runs.
    All ranks construct/close it in matching order using their CPU control group.
    """

    def __init__(
        self,
        control_group: Any,
        *,
        run_dir: str | Path,
        wandb: Mapping[str, Any] | None = None,
        reduce: Reducer = sum_rank_records,
        max_pending: int = 256,
        timeout: float = 120,
    ) -> None:
        import torch.distributed as dist

        self.rank = dist.get_rank()
        self.members = tuple(dist.get_process_group_ranks(control_group))
        self.local: Wandb | None = None
        self.aggregate: Wandb | None = None
        self.closed = False
        root = Path(run_dir)
        aggregate_root = root / "aggregate"
        self.output = None
        if self.rank == self.members[0]:
            aggregate_root.mkdir(parents=True, exist_ok=True)
            self.output = (aggregate_root / "metrics.jsonl").open("a", buffering=1)
        try:
            if wandb is not None:
                options = dict(wandb)
                project = options.pop("project")
                name = options.pop("name", root.name)
                options.setdefault("group", root.name)
                options["reinit"] = "create_new"
                # Explicit run IDs, when supplied for resume, remain distinct.
                identity = options.pop("id", None)
                local_options = dict(options)
                local_options["job_type"] = "rank"
                if identity is not None:
                    local_options["id"] = f"{identity}-rank-{self.rank:05d}"
                self.local = Wandb(
                    project=project,
                    run_dir=root / f"rank-{self.rank:05d}",
                    name=f"{name}/rank-{self.rank:05d}",
                    **local_options,
                )
                if self.rank == self.members[0]:
                    aggregate_options = dict(options)
                    aggregate_options["job_type"] = "aggregate"
                    if identity is not None:
                        aggregate_options["id"] = f"{identity}-aggregate"
                    self.aggregate = Wandb(
                        project=project,
                        run_dir=aggregate_root,
                        name=f"{name}/aggregate",
                        **aggregate_options,
                    )
            self.exchange = RecordExchange(
                control_group,
                self._emit,
                reduce,
                max_pending=max_pending,
                timeout=timeout,
            )
        except BaseException:
            self._finish_loggers(exit_code=1)
            raise

    def _emit(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True)
        assert self.output is not None
        self.output.write(line + "\n")
        print("[aggregate] " + line, flush=True)
        if self.aggregate is not None:
            self.aggregate(record)

    def __call__(self, record: dict[str, Any]) -> None:
        self.exchange.submit(record)
        if self.local is not None:
            self.local(record)

    def table(self, step: int, name: str, table: MetricTable) -> None:
        # Detailed tables stay rank-local. A scalar reducer never guesses how
        # rows from distinct model shards should be joined.
        if self.local is not None:
            self.local.table(step, name, table)

    def _finish_loggers(self, *, exit_code: int) -> None:
        try:
            if self.aggregate is not None:
                self.aggregate.close(exit_code=exit_code)
        finally:
            try:
                if self.local is not None:
                    self.local.close(exit_code=exit_code)
            finally:
                if self.output is not None:
                    self.output.close()

    def close(self, *, exit_code: int = 0) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.exchange.close(exit_code=exit_code)
        finally:
            self._finish_loggers(exit_code=exit_code)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close(exit_code=0 if exception is None else 1)
        except Exception as cleanup_error:
            if exception is None:
                raise
            exception.add_note(f"Distributed reporting cleanup failed: {cleanup_error}")
