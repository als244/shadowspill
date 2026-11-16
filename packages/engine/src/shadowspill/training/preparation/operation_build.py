"""Definition-aware calls into the one public build engine; no execution loop."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from shadowspill.artifacts.program import Program
from shadowspill.training.definitions.registry import ResolvedOptimizer
from shadowspill.training.inputs.adapters import AdaptedInputs

from .binding import OperationBinding, control_tensors
from .capture_checks import validate_operation_program, validate_optimizer_program


def build_operation(
    model: Any,
    binding: OperationBinding,
    example: AdaptedInputs,
    *,
    optimizer: ResolvedOptimizer | None = None,
    output_only: bool = False,
    options: Mapping[str, Any] | None = None,
) -> Program:
    """Build one saved shape template; CPU instantiation supplies later counts.

    A training binding also builds an output-only path for all-masked calls.
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
    if type(output_only) is not bool:
        raise TypeError("output_only must be boolean")
    kind = binding.contract.key.operation
    if kind == "step" and optimizer is None and not output_only:
        raise ValueError("step requires a resolved optimizer")
    if (
        kind != "forward"
        and not output_only
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
                "build gradient dtype differs from the optimizer definition"
            )
        settings["grad_dtype"] = gradient_dtype
    arguments = binding.arguments(example)
    program: Program
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.train(binding.contract.key.mode == "training")
        if kind == "forward" or output_only:
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
                preserve_buffers=binding.model.definition.mutable_buffers
                if output_only
                else (),
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
                        "build master dtype differs from the optimizer definition"
                    )
                settings["master_dtype"] = master_dtype
            program = build_step(
                model,
                objective=binding.objective,
                example_args=arguments,
                optimizer=optimizer.create(
                    model,
                    control_tensors(optimizer.definition.controls, optimizer.defaults),
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
        validate_operation_program(program, binding, output_only=output_only)
    finally:
        for module, mode in modes:
            module.training = mode
    optimizer_identity = None
    if kind == "step" and not output_only:
        assert optimizer is not None
        optimizer_identity = {
            "id": optimizer.definition.id,
            "source": optimizer.source.to_dict(),
            "config": dict(optimizer.config),
            "controls": optimizer.definition.controls.to_dict(),
            "defaults": dict(optimizer.defaults),
            "bindings": dict(optimizer.definition.control_bindings),
            "master_dtype": optimizer.master_dtype,
            "gradient_dtype": optimizer.gradient_dtype,
        }
    provenance = {
        **binding.signature(),
        "output_only": output_only,
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
