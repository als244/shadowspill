"""One ordinary PyTorch update, with optional masters and accumulated gradients."""

import functools
import time

import torch
from torch.utils._pytree import tree_map

from reference.training.hyperparameters import set_hyperparams
from reference.training.inputs import on_device
from reference.training.weights import Weights
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.initialization import initialize_model, model_mode
from shadowspill.pytorch.objectives import ScaledObjective
from shadowspill.training import config
from shadowspill.training.observations import StepObservations, parameter_norms
from shadowspill.training.results import StepResult
from shadowspill_workloads.recipes.text.experiment.hyperparameters import (
    scheduled_arguments,
)
from shadowspill_workloads.recipes.text.optimizer import build_optimizer


class EagerStep:
    def __init__(self, model, options, initialize):
        self.device = resolve_device(options.get("device", "auto"), allow_cpu=True)
        self.model = initialize_model(model, initialize=initialize).to(self.device)
        self.model.zero_grad(set_to_none=True)
        self.weights = Weights(
            self.model,
            config.resolve(options.get("master_dtype")),
            config.resolve(options.get("grad_dtype")),
        )
        self.optimizer = build_optimizer(
            config.resolve(options["optimizer"]),
            self.weights.parameters(),
            scheduled_arguments(
                config.resolve(options.get("optimizer_args", {})),
                options.get("schedules", {}),
            ),
        )
        objective = ScaledObjective(
            functools.partial(
                config.resolve(options["objective"]),
                **config.resolve(options.get("objective_args", {})),
            )
        )
        self.objective = (
            torch.compile(objective, fullgraph=True, dynamic=False)
            if options.get("compile", True)
            else objective
        )
        self.schedules = config.resolve(options.get("schedules", {}))
        self.observer = config.resolve(options.get("parameter_metrics"))
        self.norms_every = options.get(
            "parameter_norms_every", 1 if self.observer is not None else 0
        )
        if type(self.norms_every) is not int or self.norms_every < 0:
            raise ValueError("parameter_norms_every must be a nonnegative integer")
        if self.norms_every and self.observer is None:
            self.observer = parameter_norms

    def synchronize(self):
        if self.device.type != "cpu":
            torch.accelerator.synchronize(self.device)

    def step(self, args, step, reduce_metrics):
        values = {name: schedule(step - 1) for name, schedule in self.schedules.items()}
        started = time.perf_counter()
        set_hyperparams(self.model, self.optimizer, values)
        losses, metrics = [], []
        with model_mode(self.model, True):
            for call in args:
                loss, observations = self.objective(
                    self.model, *on_device(call, self.device)
                )
                loss.backward()
                self.weights.accumulate()
                losses.append(loss.detach())
                metrics.append(tree_map(_detach, observations))
            observer = (
                self.observer
                if self.norms_every and step % self.norms_every == 0
                else None
            )
            parameters = self.weights.observe(observer)
            self.weights.step(self.optimizer)
        self.synchronize()
        host = StepObservations.collect(losses, metrics, parameters)
        seconds = time.perf_counter() - started
        return StepResult(
            host.losses,
            host.metrics,
            host.parameter_metrics,
            step,
            seconds,
            values,
            reduce_metrics(host),
        )

    def evaluate(self, args):
        losses, metrics = [], []
        with model_mode(self.model, False), torch.no_grad():
            for call in args:
                loss, observations = self.objective(
                    self.model, *on_device(call, self.device)
                )
                losses.append(loss)
                metrics.append(observations)
        return StepObservations.collect(losses, metrics)


def _detach(value):
    return value.detach() if isinstance(value, torch.Tensor) else value
