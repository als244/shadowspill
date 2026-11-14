"""Fast runner contracts; real saved-Plan execution has an opt-in smoke test."""

import json
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from benchmarks.plan_runner import NsysCapture, run
from benchmarks.plan_runner.api import _traced_inputs, summarize, validate
from benchmarks.plan_runner.report import report
from shadowspill.orchestration.monitoring.lifecycle import Lifecycle


def test_runner_import_and_cli_help_need_no_torch_or_workload():
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {'torch', 'shadowspill_workloads', 'mlops'}
            or fullname.startswith(('shadowspill.runtime', 'shadowspill.pytorch',
                                    'shadowspill.execution'))):
            raise AssertionError('CPU launcher imported ' + fullname)
sys.meta_path.insert(0, Block())
from benchmarks.plan_runner import Context, NsysCapture, run, benchmark
assert NsysCapture().steps == 2
from benchmarks.plan_runner.__main__ import main
sys.argv = ['plan_runner', '--help']
main()
"""
    result = subprocess.run(
        [sys.executable, *(["-S"] if sys.flags.no_site else []), "-c", script],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_dir" in result.stdout


def test_aggregation_uses_slowest_rank_per_update_and_preserves_losses():
    per_rank = [
        dict(
            rank=rank,
            steps=[
                dict(step=i + 1, seconds=value, loss=rank + 0.5)
                for i, value in enumerate(values)
            ],
        )
        for rank, values in enumerate(((99, 8, 2, 8), (100, 2, 10, 2)))
    ]
    result = summarize(
        per_rank, steps=4, warmup_steps=1, units_per_step=80, unit_label="examples"
    )
    assert [r["seconds"] for r in result["steps"]] == [100, 8, 10, 8]
    assert result["median_seconds"] == 8
    assert result["units_per_second"] == 10
    assert result["per_rank"][1]["steps"][2]["loss"] == 1.5
    per_rank[1]["steps"].pop()
    with pytest.raises(ValueError, match="requested updates"):
        summarize(
            per_rank, steps=4, warmup_steps=1, units_per_step=80, unit_label="examples"
        )


def test_final_trace_is_excluded_from_steady_times_and_inputs_are_not_mutated():
    source = [{"args": [(i,)]} for i in range(4)]
    wrapped = list(_traced_inputs(source, 4, True))
    assert [row["runtime_trace"] for row in wrapped] == [False, False, False, True]
    assert all("runtime_trace" not in row for row in source)
    ranks = [
        dict(
            rank=rank, steps=[dict(step=i + 1, seconds=t) for i, t in enumerate(times)]
        )
        for rank, times in enumerate(((100, 2, 4, 999), (200, 6, 2, 1000)))
    ]
    result = summarize(
        ranks,
        steps=4,
        warmup_steps=1,
        units_per_step=50,
        unit_label="items",
        trace_final_step=True,
    )
    assert result["median_seconds"] == 5
    assert result["steady_steps"] == [2, 3] and result["trace_step"] == 4
    assert result["steps"][-1]["seconds"] == 1000
    assert result["units_per_second"] == 10
    with pytest.raises(ValueError, match="untraced measured update"):
        validate(2, 1, None, True)
    validate(2, 1, None, False)
    with pytest.raises(ValueError, match="owns runtime_trace"):
        list(_traced_inputs([{"runtime_trace": True}], 1, False))


def test_nsight_and_runtime_trace_exclusions_can_overlap():
    ranks = [dict(rank=0, steps=[dict(step=i, seconds=i) for i in range(1, 8)])]
    for capture, steady in (
        (NsysCapture(start_step=5), [2, 3, 4]),
        (NsysCapture(start_step=6), [2, 3, 4, 5]),
    ):
        result = summarize(
            ranks,
            steps=7,
            warmup_steps=1,
            units_per_step=None,
            unit_label="items",
            trace_final_step=True,
            nsys=capture,
        )
        assert result["steady_steps"] == steady
        assert result["nsys_steps"] == list(capture.step_numbers)
    with pytest.raises(ValueError, match="within the requested"):
        validate(3, 1, None, True, NsysCapture(start_step=3))
    with pytest.raises(ValueError, match="untraced measured"):
        validate(4, 1, None, True, NsysCapture())


@pytest.mark.parametrize("report_exists", [False, True])
@pytest.mark.parametrize("external", [False, True])
def test_recipe_launcher_requires_nsight_report_and_records_capture(
    tmp_path, monkeypatch, report_exists, external
):
    from benchmarks.plan_runner import recipe

    source, output = tmp_path / "source", tmp_path / "output"
    capture_dir = tmp_path / "shared" if external else output / "profiling"
    (source / "plans").mkdir(parents=True)
    (source / "plans/training.ssp").touch()
    for name, value in (("config", {"ranks": 2}), ("model", {}), ("selected", {})):
        (source / f"{name}.json").write_text(json.dumps(value))

    def launch(command, **options):
        assert command.index("torch.distributed.run") < command.index(
            "shadowspill.orchestration.capture.nsys_worker"
        )
        assert command[command.index("--output-dir") + 1] == str(capture_dir)
        request = json.loads((output / "request.json").read_text())
        assert request["nsys"]["start_step"] == 3
        (output / "result.json").write_text('{"status":"PASS"}')
        if report_exists:
            for rank in range(2):
                folder = capture_dir / f"rank-{rank:05d}"
                folder.mkdir(parents=True)
                (folder / "trace.nsys-rep").write_bytes(b"test-report")
                (folder / "capture.json").write_text(
                    json.dumps(
                        dict(
                            rank=rank,
                            world_size=2,
                            status="complete",
                            report_bytes=len(b"test-report"),
                        )
                    )
                )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(recipe.subprocess, "run", launch)
    options = dict(
        output_dir=output,
        steps=5,
        monitor=False,
        nsys=NsysCapture(
            start_step=3,
            executable="/bin/true",
            output_dir=capture_dir if external else None,
        ),
    )
    if not report_exists:
        with pytest.raises(RuntimeError, match="missing Nsight rank"):
            recipe.benchmark(source, **options)
    else:
        result = recipe.benchmark(source, **options)
        assert len(result["nsys"]["reports"]) == 2
        for record in result["nsys"]["reports"]:
            assert record["report_bytes"] == len(b"test-report")
            assert (output / record["report_path"]).is_file()
        assert result["nsys"]["content_verified"] is False


@pytest.mark.parametrize("fail", [False, True])
def test_runner_initializes_after_allocator_and_publishes_after_cleanup(
    tmp_path,
    monkeypatch,
    fail,
):
    import shadowspill

    actions = []
    plan = SimpleNamespace(
        program=SimpleNamespace(
            bindings={"kind": "step"}, parallelism=None, digest="program"
        ),
        schedule=SimpleNamespace(digest="schedule"),
        timeline=SimpleNamespace(makespan_ns=1_000_000_000),
        memory_budgets=SimpleNamespace(execution_bytes=4096, spill_bytes=2048),
    )
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(shadowspill.Plan, "load", lambda path: plan)

    class Trainer:
        def __init__(self, loaded, **options):
            assert loaded is plan
            actions.append("allocator")
            self.executor = SimpleNamespace(spill_capacity_bytes=123)

        def __enter__(self):
            return self

        def __exit__(self, kind, error, traceback):
            assert not (tmp_path / "result.json").exists()
            actions.append(("closed", kind))

        def initialize(self, fill):
            assert fill == "source"
            actions.append("initialized")

        def fit(self, source, *, callbacks, **options):
            assert list(source) == [{"args": [(1,)], "runtime_trace": False}]
            assert options["checkpoint_every"] == options["tables_every"] == 0
            assert options["startup_diagnostics"] is False
            if fail:
                raise RuntimeError("step failed")
            for step in range(1, options["steps"] + 1):
                callbacks[0](
                    self,
                    SimpleNamespace(
                        step=step,
                        seconds=1.0,
                        loss=2.0,
                        summary=SimpleNamespace(scalars={}),
                    ),
                )

    @contextmanager
    def initializer(context):
        assert actions == ["allocator"]
        assert context.rank == context.worker_rank == 0
        assert context.world_size == 1 and context.plan is plan
        yield "source"
        actions.append("source_closed")

    def inputs(context):
        assert actions[-1] == "source_closed"
        return iter([{"args": [(1,)]}])

    kwargs = dict(
        initializer=initializer,
        inputs=inputs,
        output_dir=tmp_path,
        trainer_type=Trainer,
        steps=2,
        monitor=False,
        trace_final_step=False,
    )
    if fail:
        with pytest.raises(RuntimeError, match="step failed"):
            run("unused.ssp", **kwargs)
        assert not (tmp_path / "result.json").exists()
    else:
        result = run("unused.ssp", **kwargs)
        assert result["median_seconds"] == result["predicted_seconds"] == 1.0
        assert result["per_rank"][0]["spill_capacity_bytes"] == 123
    assert actions[-1] == ("closed", RuntimeError if fail else None)
    events = [
        json.loads(line)
        for line in (tmp_path / "rank-00000/phases.jsonl").read_text().splitlines()
    ]
    assert events[-1]["name"] == "execution" and events[-1]["boundary"] == "end"
    assert events[-1].get("error") == ("RuntimeError" if fail else None)


def test_monitor_child_is_stopped_on_exception(tmp_path, monkeypatch):
    from benchmarks.plan_runner import monitor

    actions = []

    class Child:
        returncode = -15

        def __init__(self, command, **options):
            assert command[-2:] == ["--id", "GPU-test"]
            assert options["env"]["TZ"] == "UTC"

        def poll(self):
            return None

        def terminate(self):
            actions.append("terminate")

        def wait(self, **options):
            actions.append("wait")

    monkeypatch.setattr(monitor.subprocess, "Popen", Child)
    with (
        pytest.raises(RuntimeError, match="training failed"),
        monitor.GpuMonitor(tmp_path, devices="GPU-test"),
    ):
        raise RuntimeError("training failed")
    assert actions == ["terminate", "wait"]
    assert json.loads((tmp_path / "monitor.json").read_text())["status"] == "stopped"


def test_offline_report_preserves_incomplete_spans_and_missing_samples(tmp_path):
    with Lifecycle(tmp_path / "rank-00000/phases.jsonl", rank=0) as timing:
        with timing.phase("imports"):
            pass
        timing.event("training.fit", boundary="begin")
    result = report(tmp_path)
    assert result["result"] is None
    assert result["phases"][0]["device"]["sampled_gpu_util_percent_mean"] is None
    assert result["unfinished"][0]["name"] == "training.fit"


def test_recipe_bridge_reuses_hooks_without_changing_saved_inputs(
    tmp_path, monkeypatch
):
    from benchmarks.plan_runner import recipe
    from shadowspill.training.preparation import configuration

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    for name, value in (
        ("config", dict(recipe="sample", steps=10, resume="prior")),
        ("model", {"identity": "saved"}),
        ("selected", {"sequences_per_example": 4}),
    ):
        (source / f"{name}.json").write_text(json.dumps(value))
    before = {path: path.read_bytes() for path in source.iterdir()}
    (output / "request.json").write_text(json.dumps(dict(run_dir=str(source), steps=7)))
    calls = []
    monkeypatch.setattr(
        configuration,
        "adapter",
        lambda name: SimpleNamespace(
            initializer=lambda options, metadata, rank: calls.append(
                (options, metadata, rank)
            ),
            inputs=lambda options, rank: (options, rank),
            trainer_options=lambda options, metadata: {"frozen_state_id": "saved"},
        ),
    )

    def capture(path, **kwargs):
        assert path == source / "plans/training.ssp"
        context = SimpleNamespace(rank=3)
        kwargs["initializer"](context)
        options, rank = kwargs["inputs"](context)
        assert options["steps"] == 7 and options["sequences_per_example"] == 4
        assert options["resume"] is None
        assert options["eval_every"] == options["checkpoint_every"] == 0
        assert rank == 3 and calls[0][2] == 3
        assert kwargs["trainer_options"](context) == {"frozen_state_id": "saved"}

    monkeypatch.setattr(recipe, "run", capture)
    recipe.worker(output)
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.slow
def test_real_two_rank_gloo_coordination_and_logical_rank_permutation(tmp_path):
    from shadowspill.orchestration.launch.processes import worker_command

    result = subprocess.run(
        worker_command("tests.benchmarks._plan_runner", [tmp_path], ranks=2),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "result.json").read_text())["median_seconds"] == 2
    for rank in range(2):
        events = [
            json.loads(line)
            for line in (tmp_path / f"rank-{rank:05d}/phases.jsonl")
            .read_text()
            .splitlines()
        ]
        assert any(e["name"] == "control_group.close" for e in events)


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize("nsys", [None, NsysCapture(start_step=3)])
def test_real_saved_plan_benchmark_matches_eager_without_preparation(tmp_path, nsys):
    import copy
    import shutil
    import sqlite3

    import torch

    from benchmarks.plan_runner import benchmark
    from engine_tests.build.test_training import Network, objective
    from shadowspill import (
        CommunicationInfo,
        MemoryBudgets,
        apply_profiles,
        build_step,
        measure_program,
        plan_program,
    )

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if nsys is not None and shutil.which(nsys.executable) is None:
        pytest.skip("requires Nsight Systems and device-metric permission")
    steps = 5 if nsys is not None else 3
    # Two tiny updates otherwise finish in ~5 ms, before the hardware metrics
    # sampler produces a batch. Use more real microbatches, without padding the
    # production capture interval with sleeps or extra optimizer updates.
    microbatches = 128 if nsys is not None else 2
    torch.manual_seed(91)
    model = Network(tied=True)
    initial = copy.deepcopy(model.state_dict())
    examples = [
        [
            ((torch.randn(4, 8),), {"labels": torch.randn(4, 8)})
            for _ in range(microbatches)
        ]
        for _ in range(steps)
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=torch.tensor(0.001), foreach=False
    )
    losses = []
    for calls in examples:
        optimizer.zero_grad(set_to_none=True)
        values = [objective(model, *args, **kwargs)[0] for args, kwargs in calls]
        losses.append(sum(float(value.detach()) for value in values))
        for value in values:
            value.backward()
        optimizer.step()
    source = tmp_path / "source"
    (source / "plans").mkdir(parents=True)
    torch.save(dict(initial=initial, examples=examples), source / "oracle.pt")
    options = dict(recipe="tests.benchmarks._plan_runner", ranks=1, data=str(source))
    for name, value in (("config", options), ("model", {}), ("selected", {})):
        (source / f"{name}.json").write_text(json.dumps(value))
    with torch.device("meta"):
        template = Network(tied=True)
    program = build_step(
        template,
        optimizer=torch.optim.AdamW(
            template.parameters(), lr=torch.tensor(0.001), foreach=False
        ),
        objective=objective,
        example_args=examples[0][0][0],
        example_kwargs=examples[0][0][1],
        num_examples=microbatches,
    )
    profiled = apply_profiles(
        program,
        measure_program(
            program,
            warmup_seconds=0.001,
            measure_seconds=0.002,
        ),
    )
    plan = plan_program(
        profiled,
        memory_budgets=MemoryBudgets(2 << 30, 64 << 20),
        communication_info=CommunicationInfo(
            fetch_bytes_per_second=1e10, evict_bytes_per_second=1e10
        ),
    )
    plan.save(source / "plans/training.ssp")
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    output = tmp_path / "benchmark"
    result = benchmark(
        source,
        output_dir=output,
        steps=steps,
        units_per_step=4 * microbatches,
        nsys=nsys,
    )
    torch.testing.assert_close(
        torch.tensor([row["loss"] for row in result["per_rank"][0]["steps"]]),
        torch.tensor(losses),
    )
    assert result["median_seconds"] > 0 and result["units_per_second"] > 0
    assert result["predicted_seconds"] == plan.timeline.makespan_ns / 1e9
    assert result["plan"] == str((source / "plans/training.ssp").resolve())
    assert result["steady_steps"] == [2] and result["trace_step"] == steps
    trace = result["per_rank"][0]["trace"]
    assert trace["step"] == steps
    assert not any(
        trace[key]
        for key in (
            "runtime_event_overflow",
            "allocation_event_overflow",
            "allocator_overflow",
        )
    )
    diagnostics = json.loads((output / trace["diagnostics_path"]).read_text())
    assert diagnostics["tasks"]
    from shadowspill.diagnostics.step_file import read_step_events

    events = read_step_events(output / trace["diagnostics_path"])
    assert events.runtime and events.allocator
    phases = [
        json.loads(line)
        for line in (output / "rank-00000/phases.jsonl").read_text().splitlines()
    ]
    closed = next(
        i
        for i, row in enumerate(phases)
        if row["name"] == "runtime.close" and row["boundary"] == "end"
    )
    written = next(i for i, row in enumerate(phases) if row["name"] == "trace.write")
    assert closed < written
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not list(output.rglob("*.ssp"))
    timing = report(output)
    assert not timing["unfinished"]
    assert timing["monitors"][0]["status"] == "stopped"
    assert (
        next(row for row in timing["phases"] if row["name"] == "workers")["device"][
            "samples"
        ]
        > 0
    )
    if nsys is not None:
        assert len(result["nsys"]["reports"]) == 1
        capture = output / result["nsys"]["reports"][0]["report_path"]
        database = capture.with_suffix(".sqlite")
        exported = subprocess.run(
            [
                nsys.executable,
                "export",
                "--type=sqlite",
                f"--output={database}",
                str(capture),
            ],
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert exported.returncode == 0, exported.stdout + exported.stderr
        with sqlite3.connect(database) as db:
            strings = dict(db.execute("select id,value from StringIds"))
            labels = [
                text or strings.get(text_id)
                for text, text_id in db.execute(
                    'select text,textId from NVTX_EVENTS where "end" is not null'
                )
            ]
            assert sorted(
                label for label in labels if label and label.startswith("train/step=")
            ) == ["train/step=3/rank=0", "train/step=4/rank=0"]
            assert any(
                label and label.startswith("shadowspill.task.") for label in labels
            )
            assert (
                db.execute(
                    "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL"
                ).fetchone()[0]
                > 0
            )
            assert db.execute("select count(*) from GPU_METRICS").fetchone()[0] > 0
