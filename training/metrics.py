"""Where a run's metrics go: stdout, the run's ``metrics.jsonl``, and W&B.

Every metric goes to ``metrics.jsonl`` and, when a W&B project is given, to
W&B; those logged with ``echo=False`` stay off stdout, which keeps one line per
step there.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


class Logger:
    def __init__(
        self, run_dir: Path, record: dict[str, Any], project: str | None, mode: str
    ) -> None:
        self.metrics = open(run_dir / "metrics.jsonl", "a")  # noqa: SIM115
        self.wandb = None
        if project is not None:
            import wandb

            # One W&B run per run directory, resumed along with the run.
            id_file = run_dir / "wandb_id.txt"
            if not id_file.exists():
                id_file.write_text(uuid.uuid4().hex[:8])
            self.wandb = wandb.init(
                project=project,
                name=run_dir.name,
                id=id_file.read_text(),
                resume="allow",
                dir=run_dir,
                config=record,
                mode=mode,
            )

    def log(self, step: int, *, echo: bool = True, **metrics: float) -> None:
        self.metrics.write(
            json.dumps({"step": step, "time": time.time(), **metrics}) + "\n"
        )
        self.metrics.flush()
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)
        if echo:
            fields = " | ".join(
                f"{name} {_format(value)}" for name, value in metrics.items()
            )
            print(f"step {step:>6} | {fields}", flush=True)

    def close(self, exit_code: int = 0) -> None:
        """Close the metrics file and finish the W&B run, as failed when
        ``exit_code`` is not 0."""

        self.metrics.close()
        if self.wandb is not None:
            self.wandb.finish(exit_code=exit_code)


def per_trained_token(losses: list[float], tokens: int, trained: list[int]) -> float:
    """The mean loss over trained positions, from losses that each divide by a
    microbatch's ``tokens`` positions (see ``training.objectives``)."""

    return sum(loss * tokens for loss in losses) / sum(trained)


def host_rss_gib() -> float:
    """This process's resident host memory, a pinned pool included."""

    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / (1 << 20)
    return float("nan")


def _format(value: float) -> str:
    if isinstance(value, int) or abs(value) >= 1000:
        return f"{value:,.0f}"
    if 0 < abs(value) < 0.01:
        return f"{value:.2e}"
    return f"{value:.4f}"
