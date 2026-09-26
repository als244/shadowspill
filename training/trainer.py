"""The trainer: one run, from its first step or its last checkpoint to its end."""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from training import config
from training.backends import Backend, Setup
from training.data import PackedTokens
from training.metrics import Logger, host_rss_gib, per_trained_token
from training.objectives import Objective
from training.schedules import Constant, WarmupCosine

CHECKPOINT = "checkpoint.pt"


class Trainer:
    """Train a model on packed documents, on PyTorch or ShadowSpill.

    ``model`` is the model on ``meta``, structure only; ``objective(model,
    tokens, targets, seq_lens, **objective_args)`` is the loss a step
    differentiates. The optimizer is ``optimizer(parameters,
    **optimizer_args)``, and a ``schedule`` sets its learning rate every step;
    without one, its own rate stays. ``master_dtype`` gives every weight
    trained at another dtype a master copy at that one, which the optimizer
    steps in its place and each step writes the weight from; ``grad_dtype`` is
    the dtype gradients are summed at over a step's microbatches, the weights'
    own when not given. ``data`` supplies the microbatches, each at
    most ``max_tokens_per_microbatch`` tokens, ``max_tokens_per_step`` a step,
    in documents of at most ``max_seq_len`` tokens; leaving the microbatch size
    out lets ShadowSpill search for the fastest.

    A run lives in ``run_dir``: its config, metrics, and the documents each
    step trained on. Its checkpoint goes to ``checkpoint_dir`` (by default the
    run's directory) every ``checkpoint_every`` steps, and a later ``train()``
    resumes from it; it evaluates every ``eval_every`` steps, and a
    ``wandb_project`` sends the metrics to W&B too. What planning builds --
    captures, compiled graphs, profiles, plans -- is kept in
    ``artifact_store`` for later runs to reuse, by default inside the run's
    directory.

    ``setup()`` builds the backend -- on ShadowSpill, planning the step --
    without training: ``plan`` then says what the plan promises, ``planning``
    what it was made for. ``train()`` sets up when that has not been done, and
    ``close()`` releases everything; ``train()`` closes when it ends.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        model: nn.Module,
        objective: Callable[..., torch.Tensor],
        optimizer: type[torch.optim.Optimizer],
        data: PackedTokens,
        steps: int,
        max_seq_len: int,
        max_tokens_per_step: int,
        max_tokens_per_microbatch: int | None = None,
        objective_args: Mapping[str, Any] | None = None,
        optimizer_args: Mapping[str, Any] | None = None,
        schedule: Constant | WarmupCosine | None = None,
        backend: Backend | None = None,
        master_dtype: torch.dtype | None = None,
        grad_dtype: torch.dtype | None = None,
        seed: int = 0,
        eval_every: int = 0,
        eval_batches: int = 0,
        checkpoint_every: int = 0,
        checkpoint_dir: str | Path | None = None,
        artifact_store: str | Path | None = None,
        wandb_project: str | None = None,
        wandb_mode: str = "online",
        record: dict[str, Any] | None = None,
    ) -> None:
        _check_geometry(max_seq_len, max_tokens_per_step, max_tokens_per_microbatch)
        if backend is None:
            from training.backends.pytorch import PyTorch

            backend = PyTorch()
        self.run_dir = Path(run_dir)
        self.model = model
        self.objective = objective
        self.objective_args = dict(objective_args or {})
        self.optimizer = optimizer
        self.optimizer_args = dict(optimizer_args or {})
        self.data = data
        self.steps = steps
        self.max_seq_len = max_seq_len
        self.max_tokens_per_step = max_tokens_per_step
        self.max_tokens_per_microbatch = max_tokens_per_microbatch
        self.schedule = schedule
        self.backend = backend
        self.master_dtype = master_dtype
        self.grad_dtype = grad_dtype
        self.seed = seed
        self.eval_every = eval_every
        self.eval_batches = eval_batches
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.run_dir
        self.artifact_store = (
            Path(artifact_store) if artifact_store else self.run_dir / "artifact_store"
        )
        self.wandb_project = wandb_project
        self.wandb_mode = wandb_mode
        self.record = record if record is not None else self._describe()
        self.is_setup = False
        self.log: Logger | None = None
        self.last: dict[str, float] = {}
        self.open_step: _OpenStep | None = None

    @classmethod
    def from_config(cls, path: str | Path, overrides: Sequence[str] = ()) -> Trainer:
        """The trainer a JSON config describes (see ``training.config``), its
        ``run_dir`` included. Its ``settings`` -- calls that set process-wide
        state, such as which kernels a model's operations use -- are made first,
        in order."""

        raw = config.load(path, list(overrides))
        arguments = {key: value for key, value in raw.items() if key != "settings"}
        config.resolve(raw.get("settings", []))
        return cls(record=raw, **config.resolve(arguments))

    @property
    def plan(self) -> Any:
        """What the planned step promises -- ShadowSpill's ``PlanSummary``: the
        simulated step time and its parts (unconstrained compute, recomputation,
        idle, end-of-step writeback), transfer volumes, the spill pool's peak --
        or ``None`` when nothing is planned. Available after ``setup()``."""

        return self.backend.plan

    @property
    def planning(self) -> Any:
        """The geometry and ordering the plan was made for, or ``None``."""

        return self.backend.planning

    def setup(self) -> Any:
        """Build the backend at its geometry, planning the step on ShadowSpill,
        and return ``plan``."""

        if self.is_setup:
            return self.plan
        self.run_dir.mkdir(parents=True, exist_ok=True)
        began = time.perf_counter()
        self.backend.setup(
            Setup(
                module=Objective(self.model, self.objective, self.objective_args),
                optimizer=self.optimizer,
                optimizer_args=self.optimizer_args,
                data=self.data,
                max_seq_len=self.max_seq_len,
                max_tokens_per_step=self.max_tokens_per_step,
                max_tokens_per_microbatch=self.max_tokens_per_microbatch,
                hyperparams=("lr",) if self.schedule is not None else (),
                master_dtype=self.master_dtype,
                grad_dtype=self.grad_dtype,
                seed=self.seed,
                run_dir=self.run_dir,
                artifact_store=self.artifact_store,
            )
        )
        self.setup_seconds = time.perf_counter() - began
        self.tokens, self.microbatches = self.backend.geometry
        self.is_setup = True
        return self.plan

    def train(self) -> dict[str, float]:
        """Train to the last step from wherever the run stands, evaluating and
        checkpointing on the way; close the run and return the last metrics."""

        failed = True
        try:
            self._write_record()
            self.setup()
            self._start()
            for step in range(self.start, self.steps):
                self._train_step(step)
                done = step + 1
                evaluate = _due(self.eval_every, done, self.steps)
                checkpoint = _due(self.checkpoint_every, done, self.steps)
                if evaluate or checkpoint or done == self.steps:
                    # Neither is charged to the step nor overlaps it.
                    self._finish_step()
                if evaluate:
                    self._evaluate(step)
                if checkpoint:
                    self._checkpoint()
            failed = False
        finally:
            self.close(failed=failed)
        return self.last

    def close(self, failed: bool = False) -> None:
        """Release the backend and the run's logs."""

        if self.is_setup:
            self.backend.close()
            self.is_setup = False
        if self.log is not None:
            # A step the failure came after has its losses but no end.
            self._close_step(None)
            self.packing.close()
            self.log.close(exit_code=1 if failed else 0)
            self.log = None

    def _write_record(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.record, indent=2, default=repr)
        (self.run_dir / "config.json").write_text(text + "\n")
        print(json.dumps(self.record, default=repr), flush=True)

    def _start(self) -> None:
        """The data at the backend's geometry, the resumed state, and the logs."""

        self.train_data = self.data.train_packer(self.tokens, self.max_seq_len)
        validation = self.data.validation_packer(self.tokens, self.max_seq_len)
        indices = range(self.eval_batches)
        self.val_batches = [validation.microbatch(index) for index in indices]
        self.val_trained = [validation.trained_tokens([index]) for index in indices]
        self.start = self._resume()
        self.log = Logger(
            self.run_dir, self.record, self.wandb_project, self.wandb_mode
        )
        self.packing = open(self.run_dir / "packing.jsonl", "a")  # noqa: SIM115
        setup = {
            "setup_seconds": self.setup_seconds,
            "max_tokens_per_microbatch": self.tokens,
            "microbatches": self.microbatches,
        }
        if self.plan is not None:
            text = json.dumps(self.plan.as_dict(), indent=2, default=str)
            (self.run_dir / "plan.json").write_text(text + "\n")
            setup["planned_step_seconds"] = self.plan.simulated_step_seconds
        self.log.log(self.start, **setup)
        if self.plan is not None:
            summary = dataclasses.asdict(self.plan)
            plan = {f"plan/{k}": v for k, v in summary.items() if _is_number(v)}
            self.log.log(self.start, echo=False, **plan)
        # What the steps before this one trained on, so a resumed run's total
        # continues where it stopped.
        self.trained_total = self.train_data.trained_tokens(
            range(self.start * self.microbatches)
        )

    def _resume(self) -> int:
        """The step to start from: the checkpoint's, or 0."""

        path = self.checkpoint_dir / CHECKPOINT
        if not path.exists():
            return 0
        state = torch.load(path, map_location=self.backend.checkpoint_device, mmap=True)
        self.backend.load_state_dict(state)
        print(f"resumed from {path} at step {state['step']}", flush=True)
        return int(state["step"])

    def _train_step(self, step: int) -> None:
        indices = range(step * self.microbatches, (step + 1) * self.microbatches)
        documents = [self.train_data.documents(index) for index in indices]
        self.packing.write(json.dumps({"step": step, "microbatches": documents}) + "\n")
        self.packing.flush()
        microbatches = [self.train_data.microbatch(index) for index in indices]
        lr = None if self.schedule is None else self.schedule.rate(step, self.steps)
        began = time.perf_counter()
        # A backend returns from a step before the device has finished it, so
        # a step's time runs to where the next one begins.
        self._close_step(began)
        losses = self.backend.step(microbatches, lr)
        trained = [self.train_data.trained_tokens([index]) for index in indices]
        metrics = {"loss": per_trained_token(losses, self.tokens, trained)}
        if lr is not None:
            metrics["lr"] = lr
        self.open_step = _OpenStep(step, began, metrics, self.train_data.stats(indices))

    def _finish_step(self) -> None:
        """Wait for the device to finish the open step, and record it."""

        self.backend.synchronize()
        self._close_step(time.perf_counter())

    def _close_step(self, end: float | None) -> None:
        """Record the open step, its time running to ``end``; without one, its
        losses alone."""

        closing, self.open_step = self.open_step, None
        if closing is None:
            return
        metrics, packed = closing.metrics, closing.packed
        if end is not None:
            seconds = end - closing.began
            metrics["step_seconds"] = seconds
            # the targets the loss trains on: no padding, no document's last token
            metrics["tokens_per_second"] = packed["trained_tokens"] / seconds
        self.log.log(closing.step, **metrics)
        # What the step packed, for W&B and metrics.jsonl rather than stdout.
        self.trained_total += packed["trained_tokens"]
        packed["trained_tokens_total"] = self.trained_total
        self.log.log(
            closing.step, echo=False, **{f"packing/{k}": v for k, v in packed.items()}
        )
        self.last = {**self.last, **metrics}

    def _evaluate(self, step: int) -> None:
        began = time.perf_counter()
        losses = self.backend.evaluate(self.val_batches)
        metrics = {
            "val_loss": per_trained_token(losses, self.tokens, self.val_trained),
            "eval_seconds": time.perf_counter() - began,
            "device_peak_gib": self.backend.device_peak_gib(),
            "host_rss_gib": host_rss_gib(),
        }
        self.log.log(step, **metrics)
        self.last = {**self.last, **metrics}

    def _checkpoint(self) -> None:
        """Write through a temporary file, so the checkpoint is always whole."""

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / CHECKPOINT
        partial = path.with_suffix(".partial")
        self.backend.save(partial)
        partial.replace(path)

    def _describe(self) -> dict[str, Any]:
        """What the run's config.json and W&B record when no config was given."""

        return {
            name: _described(value)
            for name, value in vars(self).items()
            if name != "record"
        }


@dataclasses.dataclass
class _OpenStep:
    """A step whose call has returned and whose time has not ended yet."""

    step: int
    began: float
    metrics: dict[str, float]
    packed: dict[str, Any]


def _check_geometry(
    max_seq_len: int, per_step: int, per_microbatch: int | None
) -> None:
    for name, tokens in (
        ("max_tokens_per_step", per_step),
        ("max_tokens_per_microbatch", per_microbatch or 0),
    ):
        if tokens % max_seq_len:
            raise ValueError(f"{name} must be a multiple of max_seq_len")
    if per_microbatch and per_step % per_microbatch:
        raise ValueError("max_tokens_per_microbatch must divide max_tokens_per_step")


def _is_number(value: Any) -> bool:
    return isinstance(value, bool | int | float)


def _described(value: Any) -> Any:
    """A value as a record can hold it: plain values as they are, a module by its
    type, a class or function by its name, anything else by its repr."""

    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, nn.Module):
        return type(value).__qualname__
    if isinstance(value, dict):
        return {key: _described(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_described(item) for item in value]
    name = getattr(value, "__qualname__", None)
    if name is not None:
        return f"{value.__module__}:{name}"
    return repr(value)


def _due(every: int, done: int, steps: int) -> bool:
    """Whether something done every ``every`` steps (0: never) falls after step
    ``done``; the last step always counts."""

    return bool(every) and (done % every == 0 or done == steps)
