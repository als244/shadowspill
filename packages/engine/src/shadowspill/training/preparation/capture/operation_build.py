"""Component-aware calls into the one public build engine; no execution loop."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from shadowspill.artifacts.program import Program
from shadowspill.training.components.registry import ResolvedOptimizer
from shadowspill.training.inputs.adapters import AdaptedInputs
from shadowspill.training.preparation.capture.binding import (
    OperationBinding,
    control_tensors,
)
from shadowspill.training.preparation.capture.validation import (
    validate_operation_program,
    validate_optimizer_program,
)


def build_operation(
    model: Any,
    binding: OperationBinding,
    example: AdaptedInputs,
    *,
    optimizer: ResolvedOptimizer | None = None,
    options: Mapping[str, Any] | None = None,
) -> Program:
    """Build one saved shape template; CPU instantiation supplies later counts.

    Capture uses replicas and the original module modes are always restored.
    All optimizer operations opt into the existing final-scale consumer binding.
    """
    from shadowspill import build_forward, build_forward_backward, build_step

    settings = dict(options or {})
    reserved = {
        "objective",
        "forward_fn",
        "optimizer",
        "model",
        "example_args",
        "example_kwargs",
        "example_calls",
        "num_examples",
        "runtime_gradient_scale",
        "preserve_buffers",
    }
    if settings.keys() & reserved:
        raise ValueError("operation build options override a prepared binding")
    kind = binding.contract.key.operation
    if kind == "step" and optimizer is None:
        raise ValueError("step requires a resolved optimizer")
    if (
        kind != "forward"
        and optimizer is not None
        and optimizer.gradient_dtype is not None
    ):
        import torch

        gradient_dtype = (
            None
            if optimizer.gradient_dtype == "parameter"
            else getattr(torch, optimizer.gradient_dtype)
        )
        if "grad_dtype" in settings and settings["grad_dtype"] != gradient_dtype:
            raise ValueError(
                "build gradient dtype differs from the optimizer component"
            )
        settings["grad_dtype"] = gradient_dtype
    arguments = binding.arguments(example)
    program: Program
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.train(binding.contract.key.mode == "training")
        if kind == "forward":
            for name in (
                "graph_pair_policy",
                "grad_dtype",
                "master_dtype",
                "gradient_destination_rules",
                "fuse_narrow_gradient_accumulation",
            ):
                settings.pop(name, None)
            program = build_forward(
                model,
                forward_fn=binding.forward,
                example_args=arguments,
                **settings,
            )
        elif kind == "step":
            assert optimizer is not None
            if optimizer.master_dtype is not None:
                import torch

                master_dtype = getattr(torch, optimizer.master_dtype)
                if (
                    "master_dtype" in settings
                    and settings["master_dtype"] != master_dtype
                ):
                    raise ValueError(
                        "build master dtype differs from the optimizer component"
                    )
                settings["master_dtype"] = master_dtype
            program = build_step(
                model,
                objective=binding.objective,
                example_args=arguments,
                optimizer=optimizer.create(
                    model,
                    control_tensors(optimizer.component.controls, optimizer.defaults),
                ),
                runtime_gradient_scale=True,
                **settings,
            )
            validate_optimizer_program(program, optimizer)
        else:
            if "master_dtype" in settings:
                raise ValueError("backward-only build has no optimizer master dtype")
            program = build_forward_backward(
                model, objective=binding.objective, example_args=arguments, **settings
            )
        validate_operation_program(program, binding)
    finally:
        for module, mode in modes:
            module.training = mode
    optimizer_identity = None
    if kind == "step":
        assert optimizer is not None
        optimizer_identity = {
            "id": optimizer.component.id,
            "source": optimizer.source.to_dict(),
            "config": dict(optimizer.config),
            "controls": optimizer.component.controls.to_dict(),
            "defaults": dict(optimizer.defaults),
            "bindings": dict(optimizer.component.control_bindings),
            "master_dtype": optimizer.master_dtype,
            "gradient_dtype": optimizer.gradient_dtype,
        }
    provenance = {
        **binding.signature(),
        "optimizer": optimizer_identity,
    }
    return replace(
        program,
        bindings={
            **program.bindings,
            "operation_mode": binding.contract.key.mode,
            "training_operation": provenance,
        },
    )
