"""Optional single-process logging for the independent training reference."""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from shadowspill.training.observations import MetricTable


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
        # The worker's output is its log; the record lets whoever launched it
        # print where the run went.
        (root / "wandb_run.json").write_text(
            json.dumps(
                {
                    "project": project,
                    "id": self.run.id,
                    "name": self.run.name,
                    "url": self.run.url,
                },
                indent=2,
            )
            + "\n"
        )

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
