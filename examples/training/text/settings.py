"""Text batch, training and preparation choices independent of model selection."""

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from pathlib import Path

from shadowspill_client import (
    BatchConfig,
    BuildCandidates,
    BuildCase,
    BuildOptions,
    Execution,
    Memory,
    Optimizer,
    PlanningOptions,
    Policy,
    Preparation,
    ProfilingOptions,
)


@dataclass(frozen=True, kw_only=True)
class TextSettings:
    data: str
    sequence_length: int = 32768
    sequences_per_step: int = 8
    microbatch_sizes: tuple[int, ...] | None = None
    chunk_sizes: tuple[int, ...] = (0,)
    moe_chunk_sizes: tuple[int, ...] = (0,)
    graph_pair_policies: tuple[str, ...] = ("cost-input",)
    resolution_options: tuple[str, ...] | None = None
    tail_save_counts: tuple[int, ...] | None = None
    steps: int = 5
    schedule_steps: int = 10
    ranks: int = 1
    backward_pieces: int = 1
    learning_rate: float = 2e-4
    seed: int = 0
    eval_every: int = 2
    eval_updates: int = 2
    checkpoint_every: int = 2
    execution_gib: float = 28
    spill_gib: float = 144
    search_workers: int = 8
    warmup_seconds: float = ProfilingOptions().conditioning_seconds
    measure_seconds: float = ProfilingOptions().measurement_seconds
    external_headroom_mib: int | None = None
    reject_overbudget: bool = False

    def __post_init__(self):
        for name in (
            "sequence_length",
            "sequences_per_step",
            "steps",
            "schedule_steps",
            "ranks",
            "backward_pieces",
            "eval_updates",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "eval_every", "checkpoint_every", "search_workers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.schedule_steps < max(2, self.steps):
            raise ValueError("schedule_steps must exceed one and cover steps")
        if self.sequences_per_step % self.ranks:
            raise ValueError("global sequences_per_step must divide evenly over ranks")
        if self.backward_pieces > self.sequences_per_step:
            raise ValueError("backward_pieces exceeds the global sequence count")
        value = self.learning_rate
        if type(value) not in (int, float) or not isfinite(value) or value <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.external_headroom_mib is not None and (
            type(self.external_headroom_mib) is not int
            or self.external_headroom_mib < 0
        ):
            raise ValueError("external_headroom_mib must be nonnegative or None")
        object.__setattr__(self, "data", str(Path(self.data).resolve()))
        per_worker = self.sequences_per_step // self.ranks
        sizes = self.microbatch_sizes
        if sizes is None:
            sizes = tuple(
                n for n in self.default_microbatch_sizes if per_worker % n == 0
            )
        for name, values in (
            ("microbatch_sizes", sizes),
            ("chunk_sizes", self.chunk_sizes),
            ("moe_chunk_sizes", self.moe_chunk_sizes),
            ("tail_save_counts", self.tail_save_counts),
        ):
            if values is None:
                continue
            minimum = 1 if name == "microbatch_sizes" else 0
            if (
                not isinstance(values, (tuple, list))
                or not values
                or any(type(n) is not int or n < minimum for n in values)
            ):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, tuple(dict.fromkeys(values)))
        if any(per_worker % n for n in self.microbatch_sizes):
            raise ValueError(
                "microbatch_sizes must divide each worker's sequence count"
            )
        choices = tuple(dict.fromkeys(self.graph_pair_policies))
        if not choices or set(choices) - {
            "cost-input",
            "cost-attention",
            "attention-input",
        }:
            raise ValueError("invalid graph_pair_policies")
        object.__setattr__(self, "graph_pair_policies", choices)
        if self.resolution_options is not None:
            choices = tuple(str(Fraction(n)) for n in self.resolution_options)
            if not choices or any(not 0 <= Fraction(n) <= 1 for n in choices):
                raise ValueError("resolution_options must lie in [0, 1]")
            object.__setattr__(
                self, "resolution_options", tuple(dict.fromkeys(choices))
            )

    @property
    def default_microbatch_sizes(self):
        return (1, 2, 4, 8)

    def request(self, *, model, objective, parallelism):
        cases = []
        for sequences in self.microbatch_sizes:
            capacity = sequences * self.sequence_length
            for chunk in self.chunk_sizes:
                for expert in self.moe_chunk_sizes if chunk else (0,):
                    for retention in self.graph_pair_policies:
                        case = BuildCase(
                            capacity,
                            BuildOptions(
                                chunking=Policy(
                                    "shadowspill/text_chunks@1",
                                    {
                                        "token_capacity": capacity,
                                        "chunk_size": chunk,
                                        "expert_chunk_size": expert or chunk,
                                    },
                                )
                                if chunk
                                else None,
                                graph_pair_policy=Policy(
                                    "shadowspill/text_retention@1",
                                    {
                                        "name": retention,
                                        "chunked": bool(chunk),
                                    },
                                ),
                            ),
                        )
                        if case not in cases:
                            cases.append(case)
        return dict(
            model=model,
            objective=objective,
            optimizer=Optimizer(
                "mlops/adamw@1",
                defaults={"lr": self.learning_rate},
                precision={
                    "gradient_dtype": "float32",
                    "master_dtype": "float32",
                    "opt_state_dtype": "float32",
                },
            ),
            batch=BatchConfig(
                limits={
                    "token_capacity": self.sequence_length * self.sequences_per_step,
                    "max_sequence_length": self.sequence_length,
                    "max_sequences": self.sequences_per_step,
                }
            ),
            parallelism=parallelism,
            memory=Memory(self.execution_gib, self.spill_gib),
            execution=Execution(
                seed=self.seed, reject_overbudget=self.reject_overbudget
            ),
            operations={
                "step": ["metrics"],
                "forward_backward": ["metrics"],
                "forward": ["metrics"],
            },
            prepare=Preparation(
                external_headroom_bytes=None
                if self.external_headroom_mib is None
                else self.external_headroom_mib << 20,
                candidates=BuildCandidates(cases=cases),
                planning=PlanningOptions(
                    microbatch_counts=sorted(
                        {
                            self.sequences_per_step // (self.ranks * n)
                            for n in self.microbatch_sizes
                        }
                    ),
                    workers=self.search_workers,
                ),
                profiling=ProfilingOptions(
                    conditioning_seconds=self.warmup_seconds,
                    measurement_seconds=self.measure_seconds,
                ),
            ),
        )
