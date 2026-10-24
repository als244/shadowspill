"""Model construction and final task implementations; no profiling or pools."""

import sys

import torch
from mlops.dispatch import set_weight_gradient_dtype
from mlops.optim import AdamW

from shadowspill import build_forward, build_step
from workloads.models.llama3.implementations.mlops import Model

from .common import loss, model_config, settings
from .data import Inputs


def run(directory):
    root, options = settings(directory)
    config = model_config(options)
    source = Inputs(options)
    if source.data.vocab_size > config.vocab_size:
        raise ValueError("corpus vocabulary exceeds model vocabulary")
    examples = source.example()
    args, kwargs = examples["args"][0], examples["kwargs"][0]
    with torch.device("meta"):
        model = Model(config).to(torch.bfloat16)
    set_weight_gradient_dtype(torch.float32)
    optimizer = AdamW(
        model.parameters(),
        lr=torch.tensor(options["learning_rate"]),
        opt_state_dtype=torch.float32,
        gradient_dtype=torch.float32,
    )
    print("building training", flush=True)
    training = build_step(
        model,
        optimizer=optimizer,
        objective=loss,
        example_args=args,
        example_kwargs=kwargs,
        num_examples=len(examples["args"]),
        master_dtype=torch.float32,
        grad_dtype=torch.float32,
    )
    training.save(root / "programs/training.ssp")
    print("building evaluation", flush=True)
    evaluation = build_forward(
        model.eval(),
        forward_fn=loss,
        example_args=args,
        example_kwargs=kwargs,
    )
    evaluation.save(root / "programs/evaluation.ssp")
    print(f"built {len(training.tasks)} training tasks", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
