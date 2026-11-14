"""Build one resolved capture through installed factories and the existing engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from shadowspill.artifacts.program import Program
from shadowspill.training.definitions.registry import Registry
from shadowspill.training.inputs.adapters import AdaptedInputs

from .operation_build import build_operation
from .policies import materialize_build_options
from .templates import TemplateInventory


def build_template(
    registry: Registry,
    inventory: TemplateInventory,
    template_id: str,
    example: AdaptedInputs,
    *,
    target: str = "cuda",
    rank_map: Mapping[int, int] | None = None,
) -> Program:
    """Capture once per immutable template, with no live training initialization.

    The phase worker supplies an owned representative example for this physical
    form. Saved provenance describes timing only, not maximum-memory admission.
    Factories run here on meta; inspection and count instantiation stay CPU-only.
    """
    import torch

    setup = inventory.layouts.setup
    template = inventory.templates[template_id]
    options = materialize_build_options(
        registry, template.build, setup.config.policy_sources
    )
    with torch.device("meta"):
        model = setup.model.create()
    program = build_operation(
        model,
        template.binding,
        example,
        optimizer=setup.optimizer,
        output_only=template.output_only,
        options={
            **options,
            "parallelism": setup.context.parallelism,
            "rank_map": rank_map,
            "target": target,
        },
    )
    return replace(
        program,
        bindings={
            **program.bindings,
            "preparation_template": template.signature(),
        },
    )
