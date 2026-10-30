"""Saved-phase EP qualification; launch directly on a two-H100 allocation.

Every execution worker starts with only a saved Plan and keeps the state it
holds after each step. The eager reference, which shares MLOps' expert kernels
but uses ordinary PyTorch accumulation and FP32 AdamW, then takes each step
from the state ShadowSpill held before it and compares the step it took.
"""

import argparse
import os
import subprocess
from fractions import Fraction
from pathlib import Path

from benchmarking.processes import worker_command
from benchmarking.quickstart import main as resume
from benchmarking.quickstart import run
from shadowspill import CommunicationInfo, MemoryBudgets, ProfiledProgram, plan_program
from shadowspill.planner import SearchOptions
from shadowspill.planner.search.algorithms.pressurefit import PressureFit
from shadowspill.planner.search.algorithms.pressurefit.options import PressureFitOptions
from shadowspill.step import StepDataOrdering


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, choices=(0, 1), required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("launch once; the runner creates the distributed phase workers")
    run(
        "workloads.recipes.text.expert_parallel:experiment",
        output_dir=root,
        phase="build",
        rank_map={0: 1, 1: 0},
        steps=args.steps,
        external_headroom_gib=2,
        warmup_seconds=0.005,
        measure_seconds=0.01,
        plots=False,
        factory_args={
            # The reference arm sums each microbatch's gradient in the BF16
            # parameter's .grad; the Program under test accumulates at the same
            # dtype, so the two arms differ only in kernels, not in contract.
            "grad_dtype": "bfloat16",
            "ranks": 2,
            "sequence_length": 64,
            "sequences_per_step": 16,
            "sequences_per_example": 4,
            "config_overrides": {
                "n_layers": 2,
                "d_model": 256,
                "n_heads": 4,
                "n_kv_heads": 2,
                "head_dim": 64,
                "n_experts": 16,
                "top_k": 2,
                "d_ff_expert": 256,
                "vocab_size": 512,
            },
        },
    )
    for phase in ("profile", "communicate"):
        resume(["--phase", phase, "--output-dir", str(root)])
    profiled = ProfiledProgram.load(root / "profiled/candidate-0000.ssp")
    plan = plan_program(
        profiled,
        memory_budgets=MemoryBudgets(8 << 30, 2 << 30),
        communication_info=CommunicationInfo.load(root / "communication.ssp"),
        ordering=StepDataOrdering(depth=2, breadth=2),
        search_options=SearchOptions(
            workers=1,
            algorithm=PressureFit(
                PressureFitOptions(
                    resolution_options=(Fraction(args.resolution),),
                )
            ),
        ),
    )
    assert not plan.simulation_only_reasons
    plan.save(root / "plan.ssp")
    for phase in ("execute", "reference"):
        command = worker_command(
            "tests.shadowspill.execution.expert_parallel." + phase,
            [root],
            ranks=2,
        )
        with (root / "logs" / (phase + ".log")).open("w") as log:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
                env={**os.environ, "OMP_NUM_THREADS": "1"},
            )
        if result.returncode:
            raise RuntimeError((root / "logs" / (phase + ".log")).read_text())
        print(f"EP {phase} passed: {log.name}", flush=True)


if __name__ == "__main__":
    main()
