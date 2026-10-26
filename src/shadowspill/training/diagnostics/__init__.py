"""Uncounted, opt-in diagnostics over an already admitted training Plan."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shadowspill.execution.distributed.checks import coordinated

from . import report, state

if TYPE_CHECKING:
    from ..trainer import Trainer


def diagnose(
    trainer: Trainer, *, directory: str | Path, warmup: int = 1, **inputs: Any
) -> dict[str, Any]:
    trainer._require_open()
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    executor = trainer.executor
    executor.synchronize()
    path = executor._paths[0]
    values = dict(inputs.get("hyperparams") or {})

    def validate() -> tuple[tuple[str, ...], set[str]]:
        if not executor._initialized:
            raise RuntimeError("initialize persistent tensors before diagnostics")
        rates = state.learning_rates(path)
        return rates, state.buffer_aliases(path, values)

    rates, aliases = coordinated(executor._distributed, validate)
    root = Path(directory)
    if executor._distributed is not None:
        root /= f"rank_{executor.logical_rank:04d}"
    root.mkdir(parents=True, exist_ok=True)
    values.update(dict.fromkeys(rates, 0.0))
    inputs = {**inputs, "hyperparams": values, "collect_parameter_norms": False}
    started = time.perf_counter()
    try:
        with state.preserve(executor, path, aliases, directory=root) as buffer_bytes:
            print(
                f"startup diagnostics: {warmup} warmup call(s), one traced call; lr=0",
                flush=True,
            )
            for _ in range(warmup):
                executor.run(**inputs).result()
                executor.synchronize()
            traced = time.perf_counter()
            pending = executor.run(**inputs, runtime_trace=True)
            pending.result()
            diagnostics = pending.diagnostics
            assert diagnostics is not None
            seconds = time.perf_counter() - traced
            executor.synchronize()
            del pending
        summary = {
            "version": 1,
            "training_step": trainer.step_count,
            "warmup_steps": warmup,
            "learning_rate": 0.0,
            "buffer_snapshot_bytes": buffer_bytes,
            "traced_seconds": seconds,
        }
        report.write(path.plan, diagnostics, root, summary)
        print(f"startup diagnostics complete: {root / 'summary.json'}", flush=True)
        return summary
    finally:
        if trainer._started is not None:
            trainer._started += time.perf_counter() - started
