"""Installed descriptions for the existing BF16 text execution policies."""

from dataclasses import asdict, dataclass

from shadowspill.training.definitions import PolicyDefinition

from .source import definition_source


@dataclass(frozen=True)
class Chunks:
    token_capacity: int
    chunk_size: int
    expert_chunk_size: int | None = None

    def __post_init__(self):
        if self.expert_chunk_size is None:
            object.__setattr__(self, "expert_chunk_size", self.chunk_size)
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("text chunk geometry must contain positive integers")
        if self.token_capacity >= 2**31:
            raise ValueError("text chunk geometry exceeds int32 token offsets")


@dataclass(frozen=True)
class Retention:
    name: str = "cost-input"
    chunked: bool = False

    def __post_init__(self):
        if self.name not in ("cost-input", "cost-attention", "attention-input"):
            raise ValueError("unknown text retention policy")
        if type(self.chunked) is not bool:
            raise ValueError("text retention chunked must be boolean")


def chunks(config):
    from shadowspill_workloads.models.shared.execution_policies import chunks

    return chunks(
        sequence_length=config.token_capacity,
        batch_size=1,
        chunk_size=config.chunk_size,
        expert_chunk_size=config.expert_chunk_size,
    )


def retention(config):
    from shadowspill_workloads.models.shared.execution_policies import retention

    return retention(**asdict(config))


def register(registry):
    source = definition_source(
        (
            "definitions/text_policies.py",
            "models/shared/execution_policies.py",
            "models/shared/execution.py",
            "models/shared/chunking.py",
            "models/shared/retention.py",
        ),
        providers=("mlops",),
    )
    registry.add_policy(
        PolicyDefinition("shadowspill/text_chunks@1", "chunking", Chunks, chunks),
        source=source,
    )
    registry.add_policy(
        PolicyDefinition(
            "shadowspill/text_retention@1", "graph_pair_policy", Retention, retention
        ),
        source=source,
    )
