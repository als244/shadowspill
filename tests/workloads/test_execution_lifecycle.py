"""Lifecycle observation preserves worker cleanup and does not execute extra steps."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from workloads.recipes.lifecycle import Lifecycle
from workloads.recipes.training import configuration, execute


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_timed_context_preserves_suppression_and_observes_real_cleanup(tmp_path):
    received = []

    @contextmanager
    def manager():
        try:
            yield 17
        except ValueError as error:
            received.append(str(error))
        finally:
            received.append("closed")

    path = tmp_path / "journal.jsonl"
    with Lifecycle(path) as timing, timing.managed("resource", manager()) as value:
        assert value == 17
        raise ValueError("original failure")
    assert received == ["original failure", "closed"]
    events = _events(path)
    assert [(e["name"], e["boundary"]) for e in events] == [
        ("resource.open", "begin"),
        ("resource.open", "end"),
        ("resource.close", "begin"),
        ("resource.close", "end"),
    ]
    assert all("error" not in e for e in events)
    assert all(e["timestamp_ns"] > 0 and e["monotonic_ns"] > 0 for e in events)


@pytest.mark.parametrize("fail_initialization", [False, True])
def test_recipe_records_before_and_after_training_including_failure_cleanup(
    tmp_path, monkeypatch, fail_initialization
):
    import shadowspill

    actions = []

    class Trainer:
        def __init__(self, plan, **options):
            actions.append("created")
            self.executor = SimpleNamespace(spill_capacity_bytes=1024)
            self.step_count = 0

        def __enter__(self):
            return self

        def __exit__(self, kind, error, traceback):
            actions.append(("closed", kind))

        def initialize(self, initialize):
            if fail_initialization:
                raise RuntimeError("initialization failed")
            actions.append("initialized")

        def fit(self, source, *, callbacks, steps, **options):
            assert steps == 2
            for step in range(1, steps + 1):
                self.step_count = step
                result = SimpleNamespace(
                    step=step,
                    seconds=0.1,
                    loss=1.0,
                    summary=SimpleNamespace(scalars={}),
                )
                actions.append(step)
                for callback in callbacks:
                    callback(self, result)

    @contextmanager
    def initializer(*args, **kwargs):
        yield None

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(shadowspill, "Trainer", Trainer)
    monkeypatch.setattr(shadowspill.Plan, "load", lambda path: None)
    monkeypatch.setattr(
        configuration,
        "adapter",
        lambda name: SimpleNamespace(
            initializer=initializer,
            trainer_options=lambda *args: {},
            inputs=lambda *args, **kwargs: SimpleNamespace(state_dict=lambda: {}),
        ),
    )
    options = dict(
        recipe="test",
        ranks=1,
        resume=None,
        eval_every=0,
        steps=2,
        wandb_project=None,
        checkpoint_every=0,
        keep_last=1,
    )
    for name, value in (("config", options), ("model", {}), ("selected", {})):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    if fail_initialization:
        with pytest.raises(RuntimeError, match="initialization failed"):
            execute.run(tmp_path)
        assert actions == ["created", ("closed", RuntimeError)]
        assert not (tmp_path / "completed.json").exists()
    else:
        execute.run(tmp_path)
        assert actions == ["created", "initialized", 1, 2, ("closed", None)]
        assert json.loads((tmp_path / "completed.json").read_text())["steps"] == 2
    events = _events(tmp_path / "logs/execution-rank-00000.jsonl")
    names = [e["name"] for e in events if e["boundary"] == "begin"]
    assert names[:5] == [
        "execution",
        "imports",
        "configuration",
        "control_group.open",
        "plans.load",
    ]
    assert names[-2:] == ["runtime.close", "control_group.close"]
    assert events[-1]["name"] == "execution" and events[-1]["boundary"] == "end"
    if fail_initialization:
        assert events[-1]["error"] == "RuntimeError"
        assert "training.fit" not in names
    else:
        assert "error" not in events[-1]
        assert [
            e["step"] for e in events if e["name"] == "training.step_completed"
        ] == [1, 2]
