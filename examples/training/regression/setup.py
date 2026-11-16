"""Describe and create a ready client separately from submitting training calls."""

import json
from dataclasses import asdict, dataclass
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

APPLICATION = "shadowspill.example.regression@1"


def parallelism(ranks):
    """The existing generic Parallelism.dp record, without importing the engine."""
    return {
        "ranks_per_replica": ranks,
        "num_replicas": 1,
        "groups": {"all_ranks": [list(range(ranks))]},
        "default_parameter_placement": {"kind": "Replicated", "group": "all_ranks"},
        "parameter_placements": {},
        "default_parameter_storage": {
            "group": "all_ranks",
            "weights": "sharded",
            "gradients": "sharded",
            "optimizer": "sharded",
            "weight_group": None,
            "gradient_group": None,
            "optimizer_group": None,
            "min_shard_bytes": 0,
        },
        "parameter_storage": {},
        "inputs": {
            "version": 1,
            "partitions": {
                "name": "inputs",
                "groups": [[rank] for rank in range(ranks)],
            },
            "fields": [
                {"path": path, "mode": "replicate", "axis": None}
                for path in (
                    ["inputs", "value"],
                    ["loss_inputs", "targets"],
                    ["loss_inputs", "valid"],
                    ["loss_inputs", "weights"],
                )
            ],
            "loss": "leader",
            "outputs": {"prediction": {"mode": "leader", "axis": None}},
        },
    }


@dataclass(frozen=True)
class Settings:
    ranks: int = 2
    steps: int = 10
    samples_per_rank: int = 8
    backward_pieces: int = 1
    checkpoint_every: int = 10
    seed: int = 711
    execution_gib: float = 4
    spill_gib: float = 1
    warmup_seconds: float = ProfilingOptions().conditioning_seconds
    measure_seconds: float = ProfilingOptions().measurement_seconds

    def __post_init__(self):
        for name in ("ranks", "steps", "samples_per_rank", "backward_pieces"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.backward_pieces > self.ranks * self.samples_per_rank:
            raise ValueError("backward_pieces cannot exceed the global sample count")
        if type(self.checkpoint_every) is not int or self.checkpoint_every < 0:
            raise ValueError("checkpoint cadence must be a nonnegative integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.request()

    def request(self):
        return dict(
            model=Model("shadowspill/regression@1", initialize=Weights.random(seed=0)),
            objective=Objective("shadowspill/weighted_squared_error@1"),
            optimizer=Optimizer(
                "mlops/adamw@1",
                defaults={"lr": 0.001},
                precision={"gradient_dtype": "float32", "opt_state_dtype": "float32"},
            ),
            parallelism=parallelism(self.ranks),
            batch=BatchConfig(
                limits={"max_examples": self.ranks * self.samples_per_rank}
            ),
            memory=Memory(self.execution_gib, self.spill_gib),
            operations={
                "step": ["metrics"],
                "forward_backward": ["metrics"],
                "forward": ["predictions"],
            },
            prepare=Preparation(
                candidates=BuildCandidates(microbatch_sizes=[self.samples_per_rank]),
                planning=PlanningOptions(microbatch_counts=[1]),
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
            or not 0 <= info.completed_updates <= settings.steps
        ):
            raise ValueError("checkpoint settings, input generator or progress differ")
        cursor = info.completed_updates
        creation = service.create_training_client(
            plans=info.plans, checkpoint=checkpoint
        )
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
