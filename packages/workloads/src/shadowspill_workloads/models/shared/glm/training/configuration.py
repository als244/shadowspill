"""GLM geometry and mixed frozen storage; no worker framework at description."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import ClassVar

from shadowspill.training.components import ModelBuild
from shadowspill_workloads.components import training_options
from shadowspill_workloads.components.decoder.configuration import (
    Settings as DecoderSettings,
)
from shadowspill_workloads.models.shared.glm.initialization.metadata import DTYPES


@dataclass(frozen=True)
class Settings(DecoderSettings):
    expert_lora: ClassVar[bool] = True
    storage: dict | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.compute_dtype != "bfloat16":
            raise ValueError("GLM sparse attention requires BF16 compute")
        if self.architecture["lora_rank"]:
            raise ValueError(
                "configure adapters with Model.lora, not architecture.lora_rank"
            )
        if self.implementation == "mlops_ep":
            from .expert_parallel import validate

            validate(self)
        if self.storage is not None:
            if set(self.storage) != {"tensor_dtypes"} or not isinstance(
                self.storage["tensor_dtypes"], dict
            ):
                raise ValueError(
                    "GLM storage requires checkpoint tensor dtype declarations"
                )
            if any(
                not isinstance(name, str) or dtype not in DTYPES.values()
                for name, dtype in self.storage["tensor_dtypes"].items()
            ):
                raise ValueError("unsupported GLM checkpoint tensor dtype")


def create(config):
    from shadowspill_workloads.models.shared.glm.initialization.storage import apply

    family = f"shadowspill_workloads.models.{config.family}"
    module = import_module(family + ".implementations.mlops")
    names = import_module(family + ".weights")
    factory = None
    if config.implementation == "mlops_ep":
        from shadowspill_workloads.models.shared.expert_parallel.resources import (
            lora_factory,
        )

        if config.expert_options is None:
            raise ValueError("EP construction requires a described ModelBuild")
        factory = lora_factory(config.expert_options)
    model = module.Model(
        config.catalog.Config(**config.architecture), expert_factory=factory
    )
    model = training_options.adapt(model, config)
    if factory is not None and any(
        getattr(layer.mlp, "expert_factory", None) is not None
        and layer.mlp.experts is None
        for layer in model.layers
    ):
        raise ValueError("EP LoRA targets must include every routed expert module")
    return apply(model, config.storage, names.source_name)


def build(config, context):
    config = training_options.for_build(config, context)
    from .allocations import contracts, preparations
    from .components import layouts

    if config.implementation == "mlops_ep":
        from .expert_parallel import build as build_experts

        return build_experts(config, context)
    return ModelBuild(
        asdict(config),
        grad_dtype=config.grad_dtype,
        operator_preparation=preparations(config),
        allocation_invariants=contracts(config),
        component_layouts=layouts(config),
    )


@contextmanager
def capture_context(config):
    import torch._inductor.config as inductor

    # Preserve the original GLM-5.3 checkpoint-precision recurrence. Flash's
    # mHC/KDA paths already expose their casts through the provider boundaries.
    with (
        training_options.capture_context(config),
        inductor.patch(emulate_precision_casts=config.family == "glm53"),
    ):
        yield
