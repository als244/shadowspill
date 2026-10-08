"""Print the reviewed commit groups; write commits only with --execute.

Run from either existing checkout. No push, branch creation, or worktree changes.
Large evidence, logs, generated code, checkpoints, and source backups stay out.
"""

import argparse
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
PLAN = "docs/internal/plans/qwen_moe_ep8_1007"
GROUPS = [
    (
        "planner: restore complete retained resolutions on cache hits",
        [
            "src/shadowspill/planner/plan_store.py",
            "src/shadowspill/planner/result.py",
            "tests/shadowspill/planner/test_plan_store.py",
            "docs/python/artifact-store.md",
        ],
    ),
    (
        "distributed: share CPU planning after verifying symmetric requirements",
        [
            "README.md",
            "benchmarking/quickstart.md",
            "benchmarking/quickstart/__init__.py",
            "benchmarking/quickstart/cli.py",
            "benchmarking/quickstart/options.py",
            "benchmarking/quickstart/runner.py",
            "docs/python/api/distributed.md",
            "src/shadowspill/pytorch/distributed/__init__.py",
            "src/shadowspill/pytorch/distributed/_preparation.py",
            "src/shadowspill/pytorch/distributed/_search.py",
            "src/shadowspill/pytorch/distributed/_selection.py",
            "src/shadowspill/pytorch/distributed/_symmetry.py",
            "src/shadowspill/pytorch/distributed/_plan_exchange.py",
            "src/shadowspill/pytorch/distributed/_sweep.py",
            "src/shadowspill/pytorch/planning/forward/plan.py",
            "src/shadowspill/pytorch/planning/training/plan.py",
            "src/shadowspill/pytorch/step_search/__init__.py",
            "src/shadowspill/pytorch/step_search/sweep.py",
            "src/shadowspill/search/planner.py",
            "tests/benchmarking/test_quickstart_distributed.py",
            "tests/shadowspill/pytorch/api/test_07_distributed_training.py",
            "tests/shadowspill/pytorch/distributed/_training_case.py",
            "tests/shadowspill/pytorch/distributed/test_symmetric_planning.py",
        ],
    ),
    (
        "workloads: add Qwen3 30B and Qwen3.5 35B MoE text models",
        [
            "workloads/README.md",
            "workloads/full_model.py",
            "workloads/mlops/__init__.py",
            "workloads/mlops/qwen_moe/__init__.py",
            "workloads/mlops/qwen_moe/attention.py",
            "workloads/mlops/qwen_moe/config.py",
            "workloads/mlops/qwen_moe/experts.py",
            "workloads/mlops/qwen_moe/initialization.py",
            "workloads/mlops/qwen_moe/model.py",
            "tests/workloads/test_qwen_moe.py",
        ],
    ),
    (
        "docs: record Qwen EP8 experiment and symmetric DP validation",
        [
            f"{PLAN}/{name}"
            for name in (
                "README.md",
                "PROGRESS.md",
                "SYMMETRIC_PLANNING.md",
                "REAL_MODEL_DP4.md",
                "COMMIT_PLAN.md",
                "evidence/modularity.json",
                "evidence/modularity_validation.json",
                "evidence/fatnode_source_manifest.json",
                "evidence/fatnode/dp4/llama_comparison.json",
                "scripts/commit_changes.py",
                "scripts/experiment.py",
                "scripts/launch.sh",
                "scripts/prepare_data.py",
                "scripts/sweep.py",
                "scripts/train.py",
                "scripts/watch.py",
                "scripts/fatnode/compare_llama.py",
                "scripts/fatnode/container.py",
                "scripts/fatnode/health.py",
                "scripts/fatnode/llama.py",
                "scripts/fatnode/run.sh",
                "scripts/fatnode/run_dp4.sh",
                "scripts/fatnode/search.py",
            )
        ],
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--groups", type=int, nargs="+", choices=range(1, 5))
    args = parser.parse_args()
    selected = args.groups or range(1, 5)
    if args.execute:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip()
        if branch != "master":
            parser.error(f"expected master, found {branch!r}")
        subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=True)
        subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    for number, (message, files) in enumerate(GROUPS, 1):
        if number not in selected:
            continue
        missing = [path for path in files if not (ROOT / path).is_file()]
        if missing:
            parser.error(f"missing files in group {number}: {missing}")
        print(f"\nGroup {number}: {message} ({len(files)} files)", flush=True)
        commands = [
            ["git", "add", "-f", "--", *files],
            ["git", "commit", "-m", message],
        ]
        for command in commands:
            print(shlex.join(command), flush=True)
            if args.execute:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
