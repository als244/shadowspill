"""Baseline model, objective and input candidates for the shared phase workers."""

import torch

from baselines.common.runner import Tokens
from baselines.common.training import TrainingModel
from benchmarking.quickstart.phases.common import read
from shadowspill import Parallelism

from .microbatches import split_microbatches
from .precision import build_arguments, optimizer_arguments
from .settings import Settings


def experiment(*, case_file, device):
    import mlops
    from mlops.dispatch import set_weight_gradient_dtype

    case = read(case_file)
    settings = Settings.from_dict(case["settings"])
    world = case["description"]["geometry"]["world_size"]
    geometry, precision = settings.geometry(world), settings.precision()
    request = settings.request()
    torch.set_num_threads(settings.cpu_threads)
    torch.manual_seed(settings.seed)
    set_weight_gradient_dtype(
        getattr(torch, precision.grad_dtype or precision.model_dtype)
    )

    def model_factory():
        return TrainingModel(request, geometry, dtype=precision.model_dtype)

    def objective(model, *args):
        loss, metrics = model(*args)
        # The shared objective normalizes by rank-local tokens; DP sums gradients.
        return loss / world, metrics

    def optimizer(parameters):
        return mlops.optim.AdamW(
            parameters,
            lr=torch.tensor(settings.learning_rate),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=settings.weight_decay,
            **optimizer_arguments(precision),
        )

    config = request.entry.config(**request.config)
    data = Tokens(
        settings, geometry, config.vocab_size, replica=0, replicas=world, device=None
    )
    pair = data(0)
    candidates = {
        str(size): {
            "args": [
                args
                for args, _ in split_microbatches(
                    pair, size=size, sequence_length=geometry.sequence_length
                )
            ]
        }
        for size in settings.microbatch_candidates(world)
    }
    return {
        "model_factory": model_factory,
        "optimizer": optimizer,
        "objective": objective,
        "candidates": candidates,
        "build_options": {
            **build_arguments(precision),
            "parallelism": Parallelism.dp(ranks_per_replica=world),
        },
        "units_per_step": geometry.tokens_per_step,
        "unit_label": "tokens",
        "metadata": {"model": request.as_dict(), "precision": precision.resolved()},
    }
