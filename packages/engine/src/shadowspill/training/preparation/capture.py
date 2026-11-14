"""Build one resolved capture through installed factories and the existing engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from shadowspill.artifacts.program import Program
from shadowspill.training.definitions.registry import Registry

from .allocation_coverage import allocation_coverage
from .operation_build import build_operation
from .policies import materialize_build_options
from .representative import representative_inputs
from .templates import TemplateInventory


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
    with representative_inputs(
        template.binding.inputs,
        template.representative,
        max_bytes=setup.config.execution.max_staging_bytes,
    ) as example:
        with torch.device("meta"):
            model = setup.model.create(build=template.model_build)
        program = build_operation(
            model,
            template.binding,
            example,
            optimizer=setup.optimizer,
            output_only=template.output_only,
            options={
                **options,
                "resources": template.model_build.resources,
                "operator_preparation": template.model_build.operator_preparation,
                "parallelism": setup.context.parallelism,
                "rank_map": rank_map,
                "target": target,
            },
        )
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
