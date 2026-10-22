"""Explicit prepare/fit/close owns only the resources the Trainer opens."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from shadowspill.training import DataParallel, Trainer
from shadowspill.training.backends import PyTorch
from shadowspill.training.schedules import WarmupCosine


class _Backend(PyTorch):
    def __init__(self, events, *, opened=False, fail=False):
        super().__init__(compile=False, device="cpu")
        self.events, self.is_open, self.fail = events, opened, fail

    def open(self, *, control_group=None):
        self.events.append("runtime-open")
        if self.fail:
            raise RuntimeError("pool refused")
        self.is_open = True

    def _distributed_binding(self, *args, **kwargs):
        return None

    def prepare_step(self, model, spec, examples):
        assert self.is_open
        self.spec = spec
        session, candidate = super().prepare_step(
            model, replace(spec, distributed=None), examples
        )
        return session, candidate

    def close(self):
        self.events.append("runtime-close")
        super().close()
        self.is_open = False


def _trainer(backend, **options):
    return Trainer(
        torch.nn.Linear(2, 1),
        objective=lambda model, data: model(data).square().mean(),
        optimizer=torch.optim.SGD,
        schedules={"lr": WarmupCosine(0.1, 0.01, 1, 3)},
        backend=backend,
        **options,
    )


@pytest.mark.parametrize("borrowed", [False, True])
def test_explicit_lifecycle_opens_lazily_and_preserves_borrowed_backend(borrowed):
    events = []
    backend = _Backend(events, opened=borrowed)
    trainer = _trainer(backend)
    assert events == []
    data = torch.ones(4, 2)
    trainer.prepare(data)
    assert backend.is_open
    last = trainer.fit([data] * 3, steps=3, logger=lambda _: None)
    assert last.hyperparams["lr"] == pytest.approx(0.01)
    trainer.close()
    trainer.close()
    assert backend.is_open is borrowed
    assert events == ([] if borrowed else ["runtime-open", "runtime-close"])


def test_failed_preparation_releases_its_owned_runtime():
    events = []
    backend = _Backend(events)
    trainer = _trainer(backend)
    # Fail after the runtime opened but before any execution can be installed.
    with pytest.raises(ValueError, match="initialize failed"):
        trainer.prepare(
            torch.ones(4, 2),
            initialize=lambda _: (_ for _ in ()).throw(ValueError("initialize failed")),
        )
    assert events == ["runtime-open", "runtime-close"]
    assert not backend.is_open


def test_named_search_candidates_are_explicit_and_cannot_conflict_with_fixed_split():
    backend = _Backend([])

    def split(data):
        yield data, 1.0

    trainer = _trainer(backend, microbatch_candidates={"whole_update": split})
    trainer.prepare(torch.ones(4, 2))
    assert trainer.selected_candidate == "whole_update"
    trainer.close()
    with pytest.raises(ValueError, match="not both"):
        _trainer(backend, microbatches=split, microbatch_candidates={"all": split})
    with pytest.raises(TypeError, match="map names"):
        _trainer(backend, microbatch_candidates=[1, 2, 4])


def _mock_groups(monkeypatch, events, *, borrowed=False, execution_failure=False):
    import shadowspill.training.parallel as module

    control, execution = object(), object()
    monkeypatch.setattr(module.dist, "is_initialized", lambda: borrowed)
    monkeypatch.setattr(module.dist, "group", SimpleNamespace(WORLD=control))
    monkeypatch.setattr(module.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(
        module.dist, "init_process_group", lambda *a, **k: events.append("control-open")
    )

    def create(**kwargs):
        assert events[-1] == "runtime-open"
        assert kwargs["backend"] == "nccl"
        events.append("execution-open")
        if execution_failure:
            raise RuntimeError("NCCL failed")
        return execution

    monkeypatch.setattr(module.dist, "new_group", create)
    monkeypatch.setattr(
        module.dist,
        "destroy_process_group",
        lambda group: events.append(
            "control-close" if group is control else "execution-close"
        ),
    )
    return control, execution


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("transfer", [False, True])
@pytest.mark.parametrize("shard_optimizer", [False, True])
def test_managed_groups_surround_runtime_and_keep_sharding_independent(
    monkeypatch, borrowed, transfer, shard_optimizer
):
    events = []
    control, execution = _mock_groups(monkeypatch, events, borrowed=borrowed)
    backend = _Backend(events)
    trainer = _trainer(
        backend,
        distributed=DataParallel(shard_weights=transfer, shard_gradients=transfer),
        **({} if shard_optimizer else {"shard_optimizer": False}),
    )
    trainer.prepare(torch.ones(4, 2))
    assert backend.spec.distributed.group is execution
    if transfer:
        assert backend.spec.distributed.transfer_sharding.host_group is control
    else:
        assert backend.spec.distributed.transfer_sharding is None
    assert backend.spec.distributed.shard_gradients is transfer
    assert backend.spec.distributed.symmetric_planning is transfer
    assert backend.spec.shard_optimizer is shard_optimizer
    trainer.close()
    assert events == (
        ([] if borrowed else ["control-open"])
        + ["runtime-open", "execution-open", "runtime-close", "execution-close"]
        + ([] if borrowed else ["control-close"])
    )


@pytest.mark.parametrize("failure", ["runtime", "execution"])
def test_failed_managed_startup_cleans_up_in_reverse_order(monkeypatch, failure):
    events = []
    _mock_groups(monkeypatch, events, execution_failure=failure == "execution")
    trainer = _trainer(
        _Backend(events, fail=failure == "runtime"), distributed=DataParallel()
    )
    with pytest.raises(RuntimeError, match="pool refused|NCCL failed"):
        trainer.prepare(torch.ones(4, 2))
    assert events == (
        ["control-open", "runtime-open"]
        + (["execution-open", "runtime-close"] if failure == "execution" else [])
        + ["control-close"]
    )


def test_parallel_data_coordinates_are_available_before_group_creation(monkeypatch):
    import shadowspill.training.parallel as module

    monkeypatch.setattr(module.dist, "is_initialized", lambda: False)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    parallel = DataParallel(shard_weights=True, shard_gradients=True)
    assert (parallel.rank, parallel.world_size) == (3, 8)
    with pytest.raises(ValueError, match="gradient sharding"):
        DataParallel(shard_gradients=True)
