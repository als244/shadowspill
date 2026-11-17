"""CPU-only declarations shared by the built-in family registrations."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from shadowspill_workloads.providers import ModelImplementation


@dataclass(frozen=True)
class ParameterInitialization:
    """Parameter-distribution choices for the shared bounded decoder initializer."""

    log_uniform_decay: bool = False
    time_bias: int = 0
    normal_leaves: tuple[str, ...] = ("w13_experts", "w2_experts")


@dataclass(frozen=True)
class Family:
    """One family's catalog, checkpoint and public training integration.

    ``presets`` contains (size, microbatch tokens, description) rows. Shapes
    themselves live in the family's config module. ``training`` names an
    installed registration hook, imported only when loading definitions.
    """

    name: str
    presets: tuple[tuple[str, int, str], ...] = ()
    implementations: tuple[ModelImplementation, ...] = ("pytorch", "mlops")
    mixture: bool = False
    expert_parallel: bool = False
    factory: str | None = None
    checkpoint_reader: str | None = None
    checkpoint_types: tuple[str, ...] = ()
    initialization: ParameterInitialization | None = None
    training: str | None = None
    validate_pretrained: Callable[[dict, Any], None] | None = None
    pretrained_config: Callable[[dict, dict, dict], dict] | None = None
    parameter_dtypes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        from shadowspill_client.config.precision import parameter_dtypes

        object.__setattr__(
            self, "parameter_dtypes", parameter_dtypes(self.parameter_dtypes)
        )
