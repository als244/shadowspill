"""Group binding cleans partial startup without destroying caller rendezvous."""

import pytest

from shadowspill.communication.binding import groups as module


@pytest.fixture
def backend(monkeypatch):
    owned, closed = [], []
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(module.dist, "get_rank", lambda *a: 0)
    monkeypatch.setattr(module.dist, "destroy_process_group", closed.append)

    def gather(outputs, value, **kwargs):
        outputs[:] = [value, value]

    def create(**kwargs):
        group = object()
        owned.append(group)
        return group

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    monkeypatch.setattr(module.dist, "new_group", create)
    return owned, closed, create


def bind(**options):
    return module.GroupBindings(
        identity="saved-plan",
        groups=[(0,), (0, 1)],
        rank_map={0: 0, 1: 1},
        topology={},
        backend="nccl",
        **options,
    )


def test_close_releases_owned_groups_once(backend):
    owned, closed, _ = backend
    bound = bind()
    assert len(owned) == 3 and not closed
    bound.close()
    bound.close()
    assert closed == owned[::-1]


@pytest.mark.parametrize("created", [0, 1, 2])
def test_creation_failure_releases_every_acquired_group(backend, monkeypatch, created):
    owned, closed, create = backend

    def fail(**options):
        if len(owned) == created:
            raise RuntimeError("group creation failed")
        return create(**options)

    monkeypatch.setattr(module.dist, "new_group", fail)
    with pytest.raises(RuntimeError, match="group creation failed"):
        bind()
    assert closed == owned[::-1]


def test_validation_failure_closes_control_before_backend_creation(backend):
    owned, closed, _ = backend
    with pytest.raises(ValueError, match="invalid distributed startup"):
        bind(startup_error="worker cannot execute this Plan")
    assert len(owned) == 1 and closed == owned
