"""Describe and create a ready client separately from submitting training calls."""

from dataclasses import dataclass

from examples.training.application.parallelism import data_parallel
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

APPLICATION = "shadowspill.example.regression@1"


def parallelism(ranks):
    return data_parallel(
        ranks,
        fields=(
            ("inputs", "value"),
            ("loss_inputs", "targets"),
            ("loss_inputs", "valid"),
            ("loss_inputs", "weights"),
        ),
        outputs={"prediction": {"mode": "leader", "axis": None}},
    )


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

    @property
    def application(self):
        return APPLICATION

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
