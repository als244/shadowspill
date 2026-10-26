"""Optional language-model quickstart presets; generic runners know no text shapes."""

from __future__ import annotations

import argparse
import functools
import inspect
from dataclasses import replace

import torch

from workloads.full_model import build_model, full_model_objective, spec
from workloads.models.shared.training import LEARNING_RATE

from .geometry import search_geometries
from .precision import Precision


def _lora_settings(arguments: argparse.Namespace) -> dict[str, object]:
    """Rank, alpha and factor dtype as the preset builds them: the flags
    given, else configure_lora's own defaults, so the metadata records what
    was built rather than what was typed."""

    from workloads.lora import configure_lora

    defaults = inspect.signature(configure_lora).parameters
    return {
        name: defaults[name].default if value is None else value
        for name, value in (
            ("rank", arguments.lora_rank),
            ("alpha", arguments.lora_alpha),
            ("factor_dtype", arguments.lora_dtype),
        )
    }


def recipe(parser, arguments):
    implementation = arguments.implementation or "mlops"
    lora = _lora_settings(arguments) if arguments.lora else None
    try:
        manifest = spec(arguments.model, implementation, **arguments.model_config)
    except ValueError as error:
        parser.error(str(error))
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
                    **lora,
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
                            "rank": lora["rank"],
                            "alpha": lora["alpha"],
                            "factor_dtype": lora["factor_dtype"],
                            "targets": arguments.lora_target,
                            "head": arguments.lora_head,
                            "shared_experts": arguments.lora_shared_experts,
                            "trainable_base": arguments.trainable_base,
                        }
                        if arguments.lora
                        else {}
                    ),
                },
                "model": arguments.model,
                "implementation": implementation,
                "model_config": dict(arguments.model_config),
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
            from .distributed import data_parallel_groups

            specification, resources = data_parallel_groups(
                device=device,
                shard_size=arguments.transfer_shard_size,
                shard_gradients=arguments.shard_gradients,
                timeout=arguments.preparation_timeout,
            )
            experiment["distributed"] = specification
            experiment["context"] = resources
            experiment["metadata"]["transfer_shard_size"] = (
                arguments.transfer_shard_size
            )
            experiment["metadata"]["shard_gradients"] = arguments.shard_gradients
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
