"""One persisted Program/profile pair trains in fresh workers at two budgets."""

import copy
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from shadowspill import apply_profiles, build_step, measure_program
from tests.shadowspill.training._replanned_model import fixture


def child(arguments, **options):
    result = subprocess.run(
        [sys.executable, "-S", *arguments],
        capture_output=True,
        text=True,
        timeout=240,
        **options,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["linear", "embedding", "llama"])
def test_saved_profiles_replan_offline_and_train_without_model(tmp_path, kind):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    model, objective, oracle = fixture(kind)
    torch.save(oracle, tmp_path / "oracle.pt")
    template = copy.deepcopy(model).to("meta")
    args, kwargs = oracle["calls"][0]
    program = build_step(
        template,
        optimizer=torch.optim.SGD(template.parameters(), lr=0.01),
        objective=objective,
        example_args=args,
        example_kwargs=kwargs,
        num_examples=len(oracle["calls"]),
    )
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    apply_profiles(program, profiles).save(tmp_path / "profiled.ssp")
    child(
        [
            "-c",
            textwrap.dedent("""
            import importlib.abc
            import sys
            from pathlib import Path

            class Block(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "torch" or any(
                        fullname == key or fullname.startswith(key + ".") for key in (
                            "shadowspill.build", "shadowspill.profiling.measure",
                            "shadowspill.execution", "shadowspill.runtime",
                        )
                    ):
                        raise AssertionError("offline planning imported " + fullname)

            sys.meta_path.insert(0, Block())
            from shadowspill import (
                CommunicationInfo, MemoryBudgets, ProfiledProgram, plan_program,
            )
            directory = Path(sys.argv[1])
            profiled = ProfiledProgram.load(directory / "profiled.ssp")
            original = (directory / "profiled.ssp").read_bytes()
            for gib in (2, 3):
                plan = plan_program(
                    profiled, memory_budgets=MemoryBudgets(gib << 30, 64 << 20),
                    communication_info=CommunicationInfo(1e10, 1e10),
                )
                assert not plan.simulation_only_reasons
                assert plan.memory_layout is not None
                # Planning materializes traversal, changing the Program digest;
                # selected executable code and build identity must be unchanged.
                assert plan.program.executables == profiled.program.executables
                assert plan.program.setup == profiled.program.setup
                assert plan.fixed_device_bytes > 0
                required = plan.memory_layout.required_bytes + plan.fixed_device_bytes
                assert required <= gib << 30
                plan.save(directory / f"plan-{gib}.ssp")
            assert (directory / "profiled.ssp").read_bytes() == original
        """),
            str(tmp_path),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    for gib in (2, 3):
        child(
            [
                str(Path(__file__).with_name("_replanned_execute.py")),
                str(tmp_path),
                str(gib),
                kind,
            ],
            env={
                **os.environ,
                "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / f"cold-inductor-{gib}"),
                "TRITON_CACHE_DIR": str(tmp_path / f"cold-triton-{gib}"),
            },
        )
