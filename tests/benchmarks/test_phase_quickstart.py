"""Independent phase entry points, not the legacy runtime-first orchestrator."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from benchmarks.quickstart.phases.common import write
from engine_tests.offline._example import communication, profiled


@pytest.mark.parametrize("pool_memory", [False, True])
def test_plan_worker_consumes_saved_profiles_without_torch_or_factory(
    tmp_path, pool_memory
):
    write(
        tmp_path / "profiled/index.json",
        {
            "candidates": [
                {"name": "sample", "file": "candidate-0000.ssp"},
                {
                    "name": "too_large",
                    "file": "absent.ssp",
                    "num_examples": 2,
                    "failure": {
                        "type": "OutOfMemoryError",
                        "phase": "profile",
                        "message": "profile: candidate exhausted the device",
                    },
                },
            ],
        },
    )
    profiled().save(tmp_path / "profiled/candidate-0000.ssp")
    (tmp_path / "profiled/absent.ssp").write_bytes(b"stale profile must not be loaded")
    communication().save(tmp_path / "communication.ssp")
    if pool_memory:
        from engine_tests.offline.test_pool_memory import evidence

        evidence().save(tmp_path / "pool.ssp")
    request = {
        "output_dir": str(tmp_path),
        "factory": "must_not_import:factory",
        "search_budgets": [99, 100, 200],
        "spill_bytes": 64,
        "search_workers": 1,
        "communication_info": None,
        "pool_memory": str(tmp_path / "pool.ssp") if pool_memory else None,
        "plots": False,
        "spill": {
            "type": "shadowspill.network:RemotePool",
            "arguments": {"capacity": 64, "host": "unavailable", "port": 19426},
        },
    }
    path = tmp_path / "request.json"
    write(path, request)
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'torch', 'must_not_import'} or any(
                    fullname == key or fullname.startswith(key + '.') for key in (
                        'shadowspill.build', 'shadowspill.profiling.measure',
                        'shadowspill.execution', 'shadowspill.runtime',
                        'shadowspill.store',
                        'shadowspill.network', 'shadowspill.ssd',
                    )
                ):
                    raise AssertionError('plan worker imported ' + fullname)
        sys.meta_path.insert(0, Block())
        from benchmarks.quickstart.phases.worker import main
        sys.argv = ['worker', 'plan', sys.argv[1]]
        main()
    """)
    child = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            script,
            str(path),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    winners = json.loads((tmp_path / "plans/index.json").read_text())
    # The default planner keeps these tiny objects resident between uses; its
    # reserve is larger than the minimal-eviction fixture's 100-byte case.
    assert [row["budget"] for row in winners] == [200]
    assert all(row["plan_digest"] for row in winners)
    assert (tmp_path / "report/index.html").is_file()
    from shadowspill.artifacts.sweep import SweepResult

    result = SweepResult.load(tmp_path / "planning/sweep.ssp")
    if pool_memory:
        assert all(
            case.plan.pool_memory is not None for case in result.cases if case.plan
        )
    failures = [
        case for case in result.cases if case.candidate.startswith("too_large:")
    ]
    assert len(failures) == 6  # Two orderings at each of three budgets.
    assert all(
        case.plan is None and case.failure["phase"] == "profile" for case in failures
    )
    assert (
        "candidate exhausted the device" in (tmp_path / "report/index.html").read_text()
    )
    assert not list((tmp_path / "report").rglob("*.svg"))
    assert not (tmp_path / "execution").exists()


def test_package_import_and_cli_help_do_not_import_torch():
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise AssertionError('quickstart orchestration imported torch')
sys.meta_path.insert(0, Block())
from benchmarks.quickstart import main
main(['--help'])
"""
    child = subprocess.run(
        [sys.executable, *(["-S"] if sys.flags.no_site else []), "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "--phase" in child.stdout


def test_initial_build_does_not_require_planning_budgets(tmp_path, monkeypatch):
    from benchmarks.quickstart import main, orchestrate

    calls = []
    monkeypatch.setattr(
        orchestrate, "_launch", lambda *args, **kwargs: calls.append(args[0])
    )
    assert (
        main(
            [
                "--phase",
                "build",
                "--factory",
                "unused:factory",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert calls == ["build"]
    request = json.loads((tmp_path / "request.json").read_text())
    assert request["search_budgets"] == []
    with pytest.raises(ValueError, match="budgets"):
        main(["--phase", "plan", "--output-dir", str(tmp_path)])


def test_public_tail_controls_reach_planning_and_require_replanning(
    tmp_path, monkeypatch
):
    from benchmarks.quickstart import main, orchestrate, run
    from benchmarks.quickstart.phases.search import search_options

    phases = []
    monkeypatch.setattr(orchestrate, "_launch", lambda phase, *a: phases.append(phase))
    run(
        factory="unused:factory",
        output_dir=tmp_path,
        phase="build",
        resolution_options=["1", "1/2"],
        tail_save_counts=[4, 0, 1, 4],
    )
    request = json.loads((tmp_path / "request.json").read_text())
    assert request["tail_save_counts"] == [0, 1, 4]
    assert request["resolution_options"] == ["1/2", "1"]
    assert search_options(request).resolved_algorithm.options.tail_save_counts == (
        0,
        1,
        4,
    )
    main(
        [
            "--phase",
            "plan",
            "--output-dir",
            str(tmp_path),
            "--search-budget-gib",
            "2",
            "--spill-gib",
            "2",
            "--tail-save-counts",
            "0",
            "2",
            "8",
            "--resolution-options",
            "quarters",
        ]
    )
    request = json.loads((tmp_path / "request.json").read_text())
    assert request["tail_save_counts"] == [0, 2, 8]
    assert (
        len(search_options(request).resolved_algorithm.options.resolution_options) == 5
    )
    assert phases == ["build", "plan"]
    with pytest.raises(SystemExit):
        main(
            [
                "--phase",
                "execute",
                "--output-dir",
                str(tmp_path),
                "--tail-save-counts",
                "0",
                "4",
            ]
        )
    assert json.loads((tmp_path / "request.json").read_text()) == request


def test_joint_factory_offers_each_policy_at_each_chunk_and_input_geometry():
    from benchmarks.quickstart.phases.candidates import build_options, example_inputs
    from tests.benchmarks._generic_quickstart import chunked_experiment

    setup = chunked_experiment(device="cpu")
    controls = set()
    for candidate in setup["candidates"].values():
        inputs = example_inputs(candidate)
        options = build_options(setup, candidate)
        policy = options["graph_pair_policy"]
        controls.add(
            (
                len(inputs["args"]),
                options["chunking"].chunk_size,
                policy.save.default,
                bool(policy.save.rules),
                bool(policy.recompute.rules),
            )
        )
        assert policy.recompute.default == "recompute"
    assert controls == {
        (count, size, save_default, save_rules, recompute_rules)
        for count in (2, 4)
        for size in (1, 3)
        for save_default, save_rules, recompute_rules in (
            ("cost", False, False),
            ("cost", False, True),
            ("recompute", True, False),
        )
    }


@pytest.mark.parametrize("form", ["both", "args", "kwargs"])
def test_build_keeps_every_examples_arguments(tmp_path, monkeypatch, form):
    from types import SimpleNamespace

    from benchmarks.quickstart.phases import build

    inputs = {}
    if form != "kwargs":
        inputs["args"] = [("first",), ("second",)]
    if form != "args":
        inputs["kwargs"] = [{"length": 3}, {"length": 5}]
    model = SimpleNamespace(parameters=lambda: ())
    setup = {
        "model_factory": lambda: model,
        "optimizer": lambda parameters: None,
        "objective": None,
        "candidates": {"different_inputs": inputs},
    }
    calls = []

    def capture(model, **options):
        calls.append(options)
        return SimpleNamespace(save=lambda path: None, parallelism=None)

    monkeypatch.setattr(build, "experiment", lambda request: setup)
    monkeypatch.setattr(build, "build_step", capture)
    build.run({"output_dir": str(tmp_path), "device": "cuda:0"})
    assert calls[0]["example_calls"] == list(
        zip(inputs.get("args", [(), ()]), inputs.get("kwargs", [{}, {}]), strict=True)
    )
    assert "num_examples" not in calls[0]


@pytest.mark.cuda
@pytest.mark.parametrize(
    "factory",
    [
        "shadowspill_workloads.recipes.regression.benchmark:experiment",
        "tests.benchmarks._phase_quickstart:mixed_examples",
    ],
)
@pytest.mark.slow
def test_quickstart_process_pipeline_and_offline_replanning(tmp_path, factory):
    import torch

    from benchmarks.quickstart import main, run
    from engine_tests.spill_pool import spill_pool

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    # This caller already uses PyTorch's native allocator. Execution must happen
    # in separate workers, not try switching this process's allocator.
    before = torch.ones(1, device="cuda")
    root = tmp_path / "quickstart"
    assert (
        run(
            factory,
            factory_args={"rows": 8, "width": 8, "outputs": 8},
            search_budget_gib=[2, 2.25],
            spill_gib=0.125,
            spill=spill_pool(128 << 20),
            steps=3,
            output_dir=root,
            warmup_seconds=0.001,
            measure_seconds=0.002,
            plots=False,
        )
        == 0
    )
    assert before.item() == 1 and torch.cuda.get_allocator_backend() == "native"
    runs = [
        json.loads(path.read_text())
        for path in sorted((root / "execution").glob("*/result.json"))
    ]
    assert len(runs) == 2
    for row in runs:
        assert len(row["steps"]) == 3
        assert row["steps"][-1]["loss"] < row["steps"][0]["loss"]
        assert row["units_per_second"] > 0
    assert [step["loss"] for step in runs[0]["steps"]] == pytest.approx(
        [step["loss"] for step in runs[1]["steps"]]
    )
    artifacts = {
        path: path.read_bytes()
        for directory in ("programs", "profiles", "profiled")
        for path in (root / directory).glob("*.ssp")
    }
    assert len(artifacts) == 6
    assert (
        main(
            [
                "--phase",
                "plan",
                "--output-dir",
                str(root),
                "--search-budget-gib",
                "2.5",
                "--no-plots",
            ]
        )
        == 0
    )
    assert all(path.read_bytes() == value for path, value in artifacts.items())
    assert [
        row["budget"] for row in json.loads((root / "plans/index.json").read_text())
    ] == [int(2.5 * (1 << 30))]
    assert (root / "report/index.html").is_file()
    assert not list(root.rglob("*.cubin"))
