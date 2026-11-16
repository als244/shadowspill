"""Resolve training settings and acquire a ready client before entering the loop."""

import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from shadowspill_client import (
    BatchConfig,
    BuildCandidates,
    Memory,
    Model,
    Objective,
    Optimizer,
    PlanningOptions,
    Preparation,
    ProfilingOptions,
    Weights,
)
from shadowspill_client.protocol.requests import ArtifactRef

APPLICATION = "shadowspill.example.llama3@1"


@dataclass(frozen=True)
class Settings:
    data: str
    model: str = "llama3-8b"
    sequence_length: int = 4096
    sequences_per_step: int = 16
    sequences_per_example: int = 4
    steps: int = 5
    checkpoint_every: int = 5
    learning_rate: float = 3e-4
    seed: int = 0
    execution_gib: float = 28
    spill_gib: float = 144
    warmup_seconds: float = ProfilingOptions().conditioning_seconds
    measure_seconds: float = ProfilingOptions().measurement_seconds
    backward_pieces: int = 1

    def __post_init__(self):
        if self.model not in {"llama3-tiny", "llama3-1b", "llama3-8b"}:
            raise ValueError("unknown Llama preset")
        for name in (
            "sequence_length",
            "sequences_per_step",
            "sequences_per_example",
            "steps",
            "backward_pieces",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.sequences_per_step % self.sequences_per_example:
            raise ValueError("sequences_per_example must divide sequences_per_step")
        if self.backward_pieces > self.sequences_per_step:
            raise ValueError("backward_pieces cannot exceed sequences_per_step")
        if type(self.checkpoint_every) is not int or self.checkpoint_every < 0:
            raise ValueError("checkpoint cadence must be a nonnegative integer")
        if not isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("learning rate must be finite and nonnegative")
        object.__setattr__(self, "data", str(Path(self.data).resolve()))
        self.request()  # Validate portable precision, memory and profile settings.

    def request(self):
        capacity = self.sequence_length * self.sequences_per_step
        size = self.sequence_length * self.sequences_per_example
        return dict(
            model=Model(
                "shadowspill/llama3@1",
                config={
                    "preset": self.model.removeprefix("llama3-"),
                    "architecture": {"max_seq_len": self.sequence_length},
                },
                precision={"compute_dtype": "bfloat16", "grad_dtype": "float32"},
                initialize=Weights.random(seed=self.seed),
            ),
            objective=Objective("shadowspill/cross_entropy@1"),
            optimizer=Optimizer(
                "mlops/adamw@1",
                defaults={"lr": self.learning_rate},
                precision={
                    "gradient_dtype": "float32",
                    "opt_state_dtype": "float32",
                    "master_dtype": "float32",
                },
            ),
            batch=BatchConfig(
                limits={
                    "token_capacity": capacity,
                    "max_sequence_length": self.sequence_length,
                    "max_sequences": self.sequences_per_step,
                }
            ),
            memory=Memory(self.execution_gib, self.spill_gib),
            operations={
                "step": ["metrics"],
                "forward_backward": ["metrics"],
                "forward": ["metrics"],
            },
            prepare=Preparation(
                candidates=BuildCandidates(microbatch_sizes=sorted({size, capacity})),
                planning=PlanningOptions(
                    microbatch_counts=sorted({1, capacity // size})
                ),
                profiling=ProfilingOptions(
                    conditioning_seconds=self.warmup_seconds,
                    measurement_seconds=self.measure_seconds,
                ),
            ),
        )


def read_resume(path):
    with Path(path).open("rb") as stream:
        raw = stream.read((64 << 10) + 1)
    if len(raw) > 64 << 10:
        raise ValueError("resume record exceeds 64 KiB")
    value = json.loads(raw)
    if value["application"] != APPLICATION:
        raise ValueError("resume record belongs to another application")
    return value, Settings(**value["settings"])


def start(service, settings, source, directory, *, resume=None):
    cursor = 0
    if resume is None:
        preview = service.describe_training(**settings.request()).result()
        if (
            source.vocab_size
            > preview.config.model.config["architecture"]["vocab_size"]
        ):
            raise ValueError("corpus vocabulary exceeds model vocabulary")
        print(preview.summary(), flush=True)
        creation = service.create_training_client(config=preview.config)
    else:
        checkpoint = service.artifact(ArtifactRef.from_dict(resume["checkpoint"]))
        info = checkpoint.manifest().result().info
        metadata = info.metadata
        if (
            metadata.get("application") != APPLICATION
            or dict(metadata["settings"]) != asdict(settings)
            or dict(metadata["data"]) != source.identity
            or metadata["next_update"] != info.completed_updates
            or info.plans != ArtifactRef.from_dict(resume["plans"])
        ):
            raise ValueError("checkpoint settings, corpus or progress differ")
        cursor = info.completed_updates
        if not 0 <= cursor <= settings.steps:
            raise ValueError("checkpoint progress lies outside the requested training")
        creation = service.create_training_client(
            plans=info.plans, checkpoint=checkpoint
        )
    # Retain reconnect identifiers before waiting for preparation/initialization.
    (directory / "run.json").write_text(
        json.dumps(
            {
                "application": APPLICATION,
                "settings": asdict(settings),
                "session_id": creation.session_id,
                "run_id": creation.run_id,
            },
            indent=2,
        )
        + "\n"
    )
    return creation.result(), cursor
