"""Deterministic expansion of collection configuration into Program requests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from benchmarking.data_geometry import DataGeometry

from .config import CollectionConfig, ModelSpec, PlanningSpec, RuntimeSpec


@dataclass(frozen=True, slots=True)
class ProgramRequest:
    """One fully resolved model/geometry Program collection request."""

    config_digest: str
    collection_name: str
    model: ModelSpec
    tokens_per_microbatch: int
    sequence_length: int
    accumulation_rounds: int
    runtime: RuntimeSpec
    planning: PlanningSpec
    seed: int

    @property
    def sequences_per_microbatch(self) -> int:
        return self.tokens_per_microbatch // self.sequence_length

    @property
    def data_geometry(self) -> DataGeometry:
        return DataGeometry(
            sequence_length=self.sequence_length,
            tokens_per_microbatch=self.tokens_per_microbatch,
            accumulation_rounds=self.accumulation_rounds,
        )

    @property
    def tokens_per_step(self) -> int:
        return self.tokens_per_microbatch * self.accumulation_rounds

    @property
    def case_id(self) -> str:
        return f"{self.model.name}__{self.data_geometry.identity_fragment}"

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "config_digest": self.config_digest,
            "collection_name": self.collection_name,
            "model": self.model.to_dict(),
            "geometry": {
                "tokens_per_microbatch": self.tokens_per_microbatch,
                "sequence_length": self.sequence_length,
                "sequences_per_microbatch": self.sequences_per_microbatch,
                # Preserve the schema-v1 Program request digest.
                "accumulation_steps": self.accumulation_rounds,
                "tokens_per_step": self.tokens_per_step,
            },
            "runtime": self.runtime.to_dict(),
            "planning": self.planning.to_dict(),
            "seed": self.seed,
        }


def expand_program_requests(config: CollectionConfig) -> tuple[ProgramRequest, ...]:
    """Expand valid geometries and round-robin providers for early coverage."""

    per_model = tuple(_model_requests(config, model) for model in config.models)
    requests: list[ProgramRequest] = []
    for index in range(max(map(len, per_model))):
        requests.extend(group[index] for group in per_model if index < len(group))
    case_ids = tuple(request.case_id for request in requests)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("expanded Program case identities are not unique")
    if (
        config.expected_programs is not None
        and len(requests) != config.expected_programs
    ):
        raise ValueError(
            "expanded Program count does not match expected_programs: "
            f"expected {config.expected_programs}, observed {len(requests)}"
        )
    if not requests:
        raise ValueError("collection configuration expands to no valid Programs")
    return tuple(requests)


def _model_requests(
    config: CollectionConfig,
    model: ModelSpec,
) -> tuple[ProgramRequest, ...]:
    axes = model.geometry or config.geometry
    model_seed = _model_seed(config.seed, model.name)
    return tuple(
        ProgramRequest(
            config_digest=config.digest,
            collection_name=config.name,
            model=model,
            tokens_per_microbatch=tokens,
            sequence_length=sequence_length,
            accumulation_rounds=accumulation_rounds,
            runtime=config.runtime,
            planning=config.planning,
            seed=model_seed,
        )
        for tokens in axes.tokens_per_microbatch
        for sequence_length in axes.sequence_lengths
        if tokens % sequence_length == 0
        for accumulation_rounds in axes.accumulation_rounds
    )


def select_program_request(
    config: CollectionConfig,
    case_id: str,
) -> ProgramRequest:
    """Resolve exactly one case identity from a validated snapshot."""

    matches = tuple(
        request
        for request in expand_program_requests(config)
        if request.case_id == case_id
    )
    if len(matches) != 1:
        raise ValueError(f"configuration does not contain case {case_id!r}")
    return matches[0]


def _model_seed(seed: int, model_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{model_name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


__all__ = [
    "ProgramRequest",
    "expand_program_requests",
    "select_program_request",
]
