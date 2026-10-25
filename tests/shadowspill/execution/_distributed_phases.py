"""Independent build, measurement, communication and CPU planning processes."""

import sys
from pathlib import Path


def build(path, kind):
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
    fixture = oracle(kind, dtype)
    torch.save(fixture, path / "oracle.pt")
    template = Network(dtype=dtype, device="meta")
    args, kwargs = fixture["examples"][0][0]
    common = dict(
        example_args=args,
        parallelism=Parallelism.dp(ranks_per_replica=2),
        partition="whole",
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
            program = build_step(
                template,
                optimizer=torch.optim.AdamW(
                    template.parameters(), lr=0.001, foreach=False
                ),
                master_dtype=torch.float32,
                **common,
            )
        else:
            program = build_forward_backward(template, **common)
    program.save(path / "program.ssp")
    if kind == "step":
        build_forward(
            template,
            example_args=args,
            parallelism=common["parallelism"],
            partition="whole",
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


def plan(path):
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
            result = plan_program(
                ProfiledProgram.load(archive),
                memory_budgets=MemoryBudgets(4 << 30, 64 << 20),
                communication_info=info,
            )
            assert not result.simulation_only_reasons, result.simulation_only_reasons
            result.save(path / f"{prefix}plan.ssp")
    assert "torch" not in sys.modules, "planning imported PyTorch"


if __name__ == "__main__":
    phase, directory, *extra = sys.argv[1:]
    globals()[phase](Path(directory), *extra)
