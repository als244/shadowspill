"""Build one resolved capture through installed factories and the existing engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from shadowspill.artifacts.component_layouts import attach_component_layouts
from shadowspill.artifacts.program import Program
from shadowspill.training.definitions.registry import Registry
from shadowspill.training.preparation.capture.allocation_coverage import (
    allocation_coverage,
)
from shadowspill.training.preparation.capture.operation_build import (
    build_operation,
)
from shadowspill.training.preparation.capture.representative import (
    representative_inputs,
)
from shadowspill.training.preparation.capture.templates import TemplateInventory
from shadowspill.training.preparation.configuration.policies import (
    materialize_build_options,
)


def build_template(
    registry: Registry,
    inventory: TemplateInventory,
    template_id: str,
    *,
    target: str = "cuda",
    rank_map: Mapping[int, int] | None = None,
) -> Program:
    """Capture once per immutable template, with no live training initialization.

    The installed preset supplies a bounded representative pattern for this form.
    Saved provenance describes timing only, not maximum-memory admission.
    Factories run here on meta; inspection and count instantiation stay CPU-only.
    """
    import torch

    setup = inventory.layouts.setup
    template = inventory.templates[template_id]
    options = materialize_build_options(
        registry, template.build, setup.config.policy_sources
    )
    if template.model_build.grad_dtype is not None:
        options["grad_dtype"] = getattr(torch, template.model_build.grad_dtype)
    with (
        setup.model.capture_context(template.model_build),
        representative_inputs(
            template.binding.inputs,
            template.representative,
            max_bytes=setup.config.execution.max_staging_bytes,
        ) as example,
    ):
        if template.kind != "forward":
            options["gradient_destination_rules"] = (
                setup.model.gradient_destination_rules(template.model_build)
            )
        with torch.device("meta"):
            model = setup.model.create(build=template.model_build)
        program = build_operation(
            model,
            template.binding,
            example,
            optimizer=setup.optimizer,
            options={
                **options,
                "resources": template.model_build.resources,
                "operator_preparation": template.model_build.operator_preparation,
                "parallelism": setup.context.parallelism,
                "rank_map": rank_map,
                "target": target,
            },
        )
    program = attach_component_layouts(program, template.model_build.component_layouts)
    program = replace(
        program,
        bindings={
            **program.bindings,
            "preparation_template": template.signature(),
            "profiling_scenario": template.signature()["scenario"],
        },
    )
    return replace(
        program,
        bindings={
            **program.bindings,
            "allocation_coverage": allocation_coverage(template, program),
        },
    )
