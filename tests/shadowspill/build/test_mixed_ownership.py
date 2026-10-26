"""Shared dense parameters and local model shards need different reductions."""

import pytest
import torch
from torch import nn

from shadowspill import (
    MemoryBudgets,
    Parallelism,
    ParameterStorage,
    Sharded,
    apply_profiles,
    build_forward,
    build_forward_backward,
    build_step,
    measure_program,
    plan_program,
)
from shadowspill.artifacts.tensors import shared_model_aliases
from tests.shadowspill.offline._communication import estimated_communication


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Parameter(torch.empty(7, 5, device="meta"))
        self.local = nn.Parameter(torch.empty(5, 3, device="meta"))

    def forward(self, x):
        return (x @ self.shared).sin() @ self.local


def parallelism(mode):
    replicas = mode == "replicated"
    storage = ParameterStorage(
        group="all_ranks",
        weights="replicated",
        gradients="replicated" if replicas else "sharded",
        optimizer="replicated" if replicas else "sharded",
    )
    return Parallelism(
        ranks_per_replica=2,
        groups={"all_ranks": ((0, 1),), "local": ((0,), (1,))},
        default_parameter_storage=storage,
        parameter_placements={} if replicas else {"local": Sharded("all_ranks", 0)},
        parameter_storage={}
        if replicas
        else {"local": ParameterStorage(group="local")},
    )


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["replicated", "local_experts"])
def test_mixed_ownership_builds_and_profiles_without_groups(
    tmp_path, monkeypatch, mode
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    def forbidden(*args, **kwargs):
        raise AssertionError("local preparation attempted a live collective")

    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)
    monkeypatch.setattr(torch.distributed, "new_group", forbidden)
    model = Network()
    program = build_step(
        model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False),
        objective=lambda model, x: model(x).square().mean(),
        example_args=(torch.empty(2, 7),),
        num_examples=2,
        partition="whole",
        parallelism=parallelism(mode),
    )
    completions = {row["name"]: row for row in program.bindings["gradient_templates"]}
    if mode == "replicated":
        assert not program.bindings["gradient_shards"]
        assert not program.bindings["owned_parameter_views"]
        for row in completions.values():
            assert row["source"] != row["stored"] == row["padding"]["output"]
            assert [phase["operation"] for phase in row["phases"]] == ["all_reduce"]
    else:
        local = completions["local"]
        assert local["source"] == local["stored"] and not local["phases"]
        assert local["padding"] is None
        assert len(program.bindings["gradient_shards"]) == 1
        assert [phase["operation"] for phase in completions["shared"]["phases"]] == [
            "reduce_scatter"
        ]
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    profiled = apply_profiles(program, profiles)
    info = estimated_communication(program, topology={"node0": (0, 1)})
    for ordering in ("tail", "stage_interleaved"):
        plan = plan_program(
            profiled,
            memory_budgets=MemoryBudgets(
                profiled.fixed_device_bytes + (8 << 20), 8 << 20
            ),
            communication_info=info,
            optimizer_ordering=ordering,
        )
        assert plan.simulation_only_reasons
        plan.save(tmp_path / f"{ordering}.ssp")
    program.save(tmp_path / "program.ssp")
    profiled.save(tmp_path / "profiled.ssp")

    # Sharded optimizer publication pads the 35-element dense weight. All
    # builders must derive the same storage, even when no update is captured.
    builders = (
        (build_forward, build_forward_backward)
        if mode == "local_experts"
        else (build_forward,)
    )
    for builder in builders:
        options = (
            {"objective": lambda model, x: model(x).square().mean()}
            if builder is build_forward_backward
            else {}
        )
        other = builder(
            model,
            example_args=(torch.empty(2, 7),),
            partition="whole",
            parallelism=parallelism(mode),
            **options,
        )
        mapping = shared_model_aliases(program, other)
        assert len(mapping) == 2
        aliases = {a.alias_group_id: a for a in program.alias_groups}
        assert sorted(aliases[a].size_bytes for a in mapping.values()) == [
            60,
            140 if mode == "replicated" else 144,
        ]
