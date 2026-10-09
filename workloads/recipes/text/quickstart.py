"""Optional language-model quickstart presets; generic runners know no text shapes."""

from __future__ import annotations

import argparse
import functools
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, cast

import torch

from workloads.common.training import LEARNING_RATE
from workloads.full_model import build_model, full_model_objective, throughput_spec

from .geometry import search_geometries


@dataclass(frozen=True)
class Precision:
    """The dtypes and roundings a step trains under, named as the training
    harness names them.

    ``model_dtype`` controls model construction. ``master_dtype``,
    ``grad_dtype`` and ``round_accumulation_once`` are
    ``plan_step``'s. The rest are the optimizer's own, given to it when it is
    built: the dtype it keeps its state at and how it rounds what it stores
    and the weights it steps. A gradient dtype reaches two more places on its
    own, because a step that names one and sets neither silently rounds: the
    mlops kernels are asked for weight gradients at it, and the optimizer
    reads gradients at it. ``None`` everywhere is the library's and the
    optimizer's defaults, which is what a request that names nothing gets.
    """

    model_dtype: str = "bfloat16"
    master_dtype: str | None = None
    grad_dtype: str | None = None
    opt_state_dtype: str | None = None
    parameter_rounding: str | None = None
    opt_state_rounding: str | None = None
    round_accumulation_once: bool = False

    @classmethod
    def from_arguments(cls, arguments: argparse.Namespace) -> Precision:
        return cls(
            model_dtype=arguments.model_dtype,
            master_dtype=arguments.master_dtype,
            grad_dtype=arguments.grad_dtype,
            opt_state_dtype=arguments.opt_state_dtype,
            parameter_rounding=arguments.parameter_rounding,
            opt_state_rounding=arguments.opt_state_rounding,
            round_accumulation_once=bool(arguments.round_accumulation_once),
        )

    @staticmethod
    def _dtype(name: str | None) -> torch.dtype | None:
        return None if name is None else cast(torch.dtype, getattr(torch, name))

    @property
    def master(self) -> torch.dtype | None:
        """``plan_step``'s ``master_dtype``."""

        return self._dtype(self.master_dtype)

    @property
    def gradients(self) -> torch.dtype | None:
        """``plan_step``'s ``grad_dtype``."""

        return self._dtype(self.grad_dtype)

    def plan_arguments(self) -> dict[str, object]:
        """What planning is told: the same three keywords the harness passes."""

        return {
            "master_dtype": self.master,
            "grad_dtype": self.gradients,
            "round_accumulation_once": self.round_accumulation_once,
        }

    def optimizer_arguments(self) -> dict[str, object]:
        """What the optimizer is built with, beyond its defaults."""

        given: dict[str, object] = {}
        if self.grad_dtype is not None:
            given["gradient_dtype"] = self._dtype(self.grad_dtype)
        if self.opt_state_dtype is not None:
            given["opt_state_dtype"] = (
                "parameter"
                if self.opt_state_dtype == "parameter"
                else self._dtype(self.opt_state_dtype)
            )
        if self.parameter_rounding is not None:
            given["parameter_rounding"] = self.parameter_rounding
        if self.opt_state_rounding is not None:
            given["opt_state_rounding"] = self.opt_state_rounding
        return given

    def optimizer(self, base: Any) -> Any:
        """``base`` built with these settings, or ``base`` itself when it
        names none, so a request that names nothing plans exactly as before."""

        given = self.optimizer_arguments()
        return functools.partial(base, **given) if given else base

    def apply(self) -> None:
        """Ask the mlops kernels for weight gradients at the gradient dtype,
        so what the step sums comes back unrounded."""

        if self.grad_dtype is not None:
            from mlops.dispatch import set_weight_gradient_dtype

            set_weight_gradient_dtype(self._dtype(self.grad_dtype))

    def lines(self) -> tuple[tuple[str, str, str], ...]:
        """The banner's rows: a label, the value, and what it means."""

        gradients = self.grad_dtype or "the weights'"
        return (
            ("model dtype", self.model_dtype, "the model weights and activations"),
            (
                "master dtype",
                self.master_dtype or "none",
                "the optimizer steps the weights themselves"
                if self.master_dtype is None
                else "a master copy of every weight trained at another dtype",
            ),
            (
                "grad dtype",
                self.grad_dtype or "weights'",
                f"gradients summed over the microbatches at {gradients} dtype;"
                + (
                    " a multiply adds its product in as it writes, rounded once"
                    if self.round_accumulation_once
                    else " a product rounded to a narrower gradient is added after"
                ),
            ),
            (
                "opt state dtype",
                self.opt_state_dtype or "default",
                "the moments at the optimizer's default dtype"
                if self.opt_state_dtype is None
                else "the moments at that dtype ('parameter': what it steps)",
            ),
            (
                "parameter rounding",
                self.parameter_rounding or "default",
                "how the optimizer rounds the weights it steps; its default"
                " is to nearest",
            ),
            (
                "opt state rounding",
                self.opt_state_rounding or "default",
                "ShadowSpill defaults to stochastic for BF16 AdamW moments;"
                " nearest otherwise",
            ),
        )


def recipe(parser, arguments):
    implementation, family = arguments.model.split("_", 1)
    manifest = throughput_spec(family, implementation)
    precision = Precision.from_arguments(arguments)
    length = arguments.sequence_length or manifest.sequence_length
    sequences = (
        arguments.sequences_per_step
        or manifest.sequences_per_microbatch * manifest.accumulation_count
    )
    manual = arguments.sequences_per_microbatch
    if manual is not None and (manual <= 0 or sequences % manual):
        parser.error("--sequences-per-microbatch must divide --sequences-per-step")
    geometries, skipped = search_geometries(
        sequences,
        sequence_length=length,
        min_tokens_per_microbatch=arguments.min_tokens_per_microbatch,
        max_tokens_per_microbatch=arguments.max_tokens_per_microbatch,
    )
    if manual is not None:
        geometries = ((manual, sequences // manual),)
    if not geometries:
        parser.error("no text geometry satisfies the token bounds")
    manifest = replace(
        manifest,
        sequence_length=length,
        sequences_per_microbatch=sequences,
        accumulation_count=1,
        model_dtype=arguments.model_dtype,
        master_dtype=arguments.master_dtype or "none",
        grad_dtype=arguments.grad_dtype or arguments.model_dtype,
        opt_state_dtype=(
            (arguments.master_dtype or arguments.model_dtype)
            if arguments.opt_state_dtype == "parameter"
            else arguments.opt_state_dtype or manifest.opt_state_dtype
        ),
    )
    world_size, rank = 1, 0
    if arguments.distributed:
        import torch.distributed as dist

        world_size, rank = dist.get_world_size(), dist.get_rank()
    # Every rank has independent text data, reused across candidates/budgets.
    generator = torch.Generator().manual_seed(arguments.seed * 1_000_003 + rank)
    vocabulary = int(manifest.model_config.vocab_size)
    whole = (1, length * sequences)
    tokens = torch.randint(vocabulary, whole, generator=generator)
    targets = torch.randint(vocabulary, whole, generator=generator)
    candidates = {}
    for per_batch, _accumulation in geometries:
        capacity = per_batch * length
        candidates[str(per_batch)] = tuple(
            (
                tokens[:, i : i + capacity].clone(),
                targets[:, i : i + capacity].clone(),
                (length,) * per_batch,
            )
            for i in range(0, whole[1], capacity)
        )

    def setup(*, device):
        precision.apply()

        def initialize(model):
            from shadowspill.training import reset_parameters

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(arguments.seed)
                reset_parameters(model)

        def model_factory():
            model = build_model(manifest)
            if arguments.lora:
                from workloads.lora import configure_lora

                model = configure_lora(
                    model,
                    rank=arguments.lora_rank,
                    alpha=arguments.lora_alpha,
                    factor_dtype=arguments.lora_dtype,
                    targets=arguments.lora_target,
                    head=arguments.lora_head,
                    shared_experts=arguments.lora_shared_experts,
                    trainable_base=arguments.trainable_base,
                )
            return model

        experiment = {
            "model_factory": model_factory,
            "initialize": initialize,
            "objective": (
                functools.partial(_distributed_objective, manifest, world_size)
                if arguments.distributed
                else functools.partial(full_model_objective, manifest)
            ),
            "optimizer": precision.optimizer(manifest.dtypes.optimizer()),
            "hyperparams": {"lr": LEARNING_RATE},
            "candidates": candidates,
            "metric_reducer": text_metrics,
            "plan_options": precision.plan_arguments(),
            "units_per_step": length * sequences,
            "unit_label": "tokens",
            "metadata": {
                "trainable": {
                    "mode": "lora" if arguments.lora else "full",
                    **(
                        {
                            "rank": arguments.lora_rank,
                            "alpha": arguments.lora_alpha,
                            "factor_dtype": arguments.lora_dtype,
                            "targets": arguments.lora_target,
                            "head": arguments.lora_head,
                            "shared_experts": arguments.lora_shared_experts,
                            "trainable_base": arguments.trainable_base,
                        }
                        if arguments.lora
                        else {}
                    ),
                },
                "world_size": world_size,
                "rank": rank,
                "units_scope": "per_rank",
                "sequence_length": length,
                "sequences_per_step": sequences,
                "skipped": skipped,
                "dtypes": {name: value for name, value, _ in precision.lines()},
            },
        }
        if arguments.distributed:
            import torch.distributed as dist

            from shadowspill.pytorch import Distributed

            group = dist.new_group(backend="nccl", device_id=device)

            @contextmanager
            def resources():
                try:
                    yield
                finally:
                    dist.destroy_process_group(group)

            experiment["distributed"] = Distributed(
                group, timeout=arguments.preparation_timeout
            )
            experiment["context"] = resources
        return experiment

    defaults = {
        "execution_gib": 16,
        "spill_gib": 112,
    }
    return setup, defaults, None if manual is None else str(manual)


def text_metrics(observed):
    """Preserve the text preset's head-loss display; also retain total objective."""
    return {
        "loss": sum(
            value.get("head_loss", loss) if isinstance(value, dict) else loss
            for loss, value in zip(observed.losses, observed.metrics, strict=True)
        )
    }


def _distributed_objective(manifest, world_size, model, *values):
    """DP recipe: each rank contributes its token-loss sum / global tokens."""
    result = full_model_objective(manifest, model, *values)
    if isinstance(result, tuple):
        loss, metrics = result
        return loss / world_size, metrics
    return result / world_size, {"head_loss": result.detach()}
