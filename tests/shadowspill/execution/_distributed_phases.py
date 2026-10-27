"""Independent build, measurement, communication and CPU planning processes."""

import sys
from pathlib import Path


def build(path, kind, ownership="sharded"):
    import torch

    from shadowspill import (
        Parallelism,
        build_forward,
        build_forward_backward,
        build_step,
    )
    from tests.shadowspill.execution._distributed_fixture import (
        Network,
        objective,
        oracle,
    )

    dtype = torch.bfloat16 if kind == "step" else torch.float32
    fixture = oracle(kind, dtype) if ownership == "sharded" else None
    template = Network(dtype=dtype, device="meta")
    parallelism = Parallelism.dp(ranks_per_replica=2)
    if ownership in {"fp8", "fp8_cutoff"}:
        from tests.shadowspill.execution import _fp8_fixture as quantized

        cutoff = 1024 if ownership == "fp8_cutoff" else 0
        fixture = quantized.oracle(min_shard_bytes=cutoff)
        template = quantized.Network(device="meta")
        parallelism = Parallelism.dp(ranks_per_replica=2, min_shard_bytes=cutoff)
        objective = quantized.objective
    elif ownership != "sharded":
        from tests.shadowspill.execution import _ownership_fixture as case

        assert kind == "step"
        fixture, template = case.oracle(ownership), case.Network(device="meta")
        objective, parallelism = case.objective, case.parallelism(ownership)
    torch.save(fixture, path / "oracle.pt")
    args, kwargs = fixture["examples"][0][0]
    common = dict(
        example_args=args,
        parallelism=parallelism,
        partition="auto",
    )
    if kind == "forward":
        program = build_forward(template, **common)
    else:
        common.update(
            example_kwargs=kwargs,
            objective=objective,
            num_examples=2,
            grad_dtype=torch.float32,
        )
        if kind == "step":
            optimizer = torch.optim.AdamW(
                template.parameters(), lr=0.001, foreach=False
            )
            if ownership == "diagnostics":
                from mlops.optim import AdamW

                optimizer = AdamW(template.parameters(), lr=torch.tensor(0.001))
            program = build_step(
                template,
                optimizer=optimizer,
                master_dtype=torch.float32 if ownership == "sharded" else None,
                **common,
            )
        else:
            program = build_forward_backward(template, **common)
    program.save(path / "program.ssp")
    if kind == "step":
        build_forward(
            template,
            example_args=args,
            forward_fn=(
                None if ownership == "sharded" else lambda model, x: model.predict(x)
            ),
            parallelism=common["parallelism"],
            partition="auto",
        ).save(path / "evaluation-program.ssp")
    assert not torch.distributed.is_initialized()


def profile(path):
    from shadowspill import Program, apply_profiles, measure_program

    for prefix in ("", "evaluation-"):
        archive = path / f"{prefix}program.ssp"
        if archive.exists():
            program = Program.load(archive)
            profiles = measure_program(
                program, warmup_seconds=0.001, measure_seconds=0.002
            )
            apply_profiles(program, profiles).save(path / f"{prefix}profiled.ssp")


def communicate(path):
    import os
    from datetime import timedelta

    import torch.distributed as dist

    from shadowspill import Program, measure_communication

    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        info = measure_communication(
            program=Program.load(path / "program.ssp"),
            rank_map={0: 1, 1: 0},
            device=f"cuda:{os.environ['LOCAL_RANK']}",
            sizes_bytes=(1024, 1 << 20),
            warmup_seconds=0.003,
            measure_seconds=0.006,
        )
        if dist.get_rank() == 0:
            info.save(path / "communication.ssp")
            print(
                "Measured communication:",
                info.fixed_device_bytes,
                "fixed bytes",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def select_variants(profiled, variant):
    from dataclasses import replace

    bindings = dict(profiled.program.bindings)
    inventories = bindings["training_templates"]
    assert all(len(inventory) > 1 for inventory in inventories)
    # The terminal forward must save its outputs. Earlier stages exercise the
    # requested variant, including real recomputation in the recompute cases.
    bindings["training_templates"] = [
        [
            [
                form
                for form in stage
                if form["variant"]
                == ("save" if index == len(inventory) - 1 else variant)
            ]
            for index, stage in enumerate(inventory)
        ]
        for inventory in inventories
    ]
    return replace(profiled, program=replace(profiled.program, bindings=bindings))


def plan(path, variant="auto"):
    from shadowspill import (
        CommunicationInfo,
        MemoryBudgets,
        ProfiledProgram,
        plan_program,
    )

    info = CommunicationInfo.load(path / "communication.ssp")
    for prefix in ("", "evaluation-"):
        archive = path / f"{prefix}profiled.ssp"
        if archive.exists():
            profiled = ProfiledProgram.load(archive)
            if prefix == "" and variant != "auto":
                profiled = select_variants(profiled, variant)
            result = plan_program(
                profiled,
                memory_budgets=MemoryBudgets(4 << 30, 64 << 20),
                communication_info=info,
                optimizer_ordering="tail"
                if variant == "recompute"
                else "stage_interleaved",
            )
            assert not result.simulation_only_reasons, result.simulation_only_reasons
            if prefix == "" and variant == "recompute":
                assert "recompute" in {choice.option_id for choice in result.selections}
            result.save(path / f"{prefix}plan.ssp")
    assert "torch" not in sys.modules, "planning imported PyTorch"


if __name__ == "__main__":
    phase, directory, *extra = sys.argv[1:]
    globals()[phase](Path(directory), *extra)
