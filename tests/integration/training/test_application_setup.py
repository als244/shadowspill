"""Shared setup retains recovery records and never prepares a resumed client."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from examples.training.application.records import record
from examples.training.application.setup import close_after, read_resume, start
from shadowspill_client.protocol.requests import ArtifactRef


@dataclass(frozen=True)
class Settings:
    steps: int = 3
    sizes: tuple = (1, 2)
    application = "custom.example@1"

    def __post_init__(self):
        object.__setattr__(self, "sizes", tuple(self.sizes))


def future(value):
    return SimpleNamespace(result=lambda: value)


@pytest.mark.parametrize("fail", [False, True])
def test_resume_uses_saved_plans_and_records_reconnect_ids_before_wait(tmp_path, fail):
    settings = Settings()
    data = {"generator": "custom", "seed": 17}
    checkpoint = ArtifactRef("checkpoint", "a" * 64, 1)
    plans = ArtifactRef("plans", "b" * 64, 1)
    saved = {
        "application": settings.application,
        "settings": record(settings),
        "checkpoint": checkpoint.to_dict(),
        "plans": plans.to_dict(),
    }
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(saved))
    assert read_resume(path, Settings) == (saved, settings)
    metadata = {**saved, "data": data, "next_update": 2}
    info = SimpleNamespace(
        to_dict=lambda: {"metadata": metadata}, completed_updates=2, plans=plans
    )
    handle = SimpleNamespace(manifest=lambda: future(SimpleNamespace(info=info)))
    training = SimpleNamespace(plans=SimpleNamespace(reference=plans))
    created = []

    def ready():
        assert json.loads((tmp_path / "run.json").read_text()) == {
            "application": settings.application,
            "settings": record(settings),
            "session_id": "session",
            "run_id": "run",
        }
        if fail:
            raise RuntimeError("initialization failed")
        return training

    def create(**options):
        created.append(options)
        return SimpleNamespace(session_id="session", run_id="run", result=ready)

    # There is deliberately no describe/build/plan operation on this service.
    service = SimpleNamespace(artifact=lambda _: handle, create_training_client=create)
    if fail:
        with pytest.raises(RuntimeError, match="initialization failed"):
            start(service, settings, tmp_path, data=data, resume=saved)
        assert not (tmp_path / "plans.json").exists()
    else:
        assert start(service, settings, tmp_path, data=data, resume=saved) == (
            training,
            2,
        )
        assert json.loads((tmp_path / "plans.json").read_text()) == plans.to_dict()
    assert created == [{"plans": plans, "checkpoint": handle}]


def test_resume_rejects_a_large_reference_before_decoding_settings(tmp_path):
    path = tmp_path / "saved.json"
    path.write_bytes(b" " * ((64 << 10) + 1))
    with pytest.raises(ValueError, match="64 KiB"):
        read_resume(path, Settings)


@pytest.mark.parametrize("application_fails", [False, True])
@pytest.mark.parametrize("close_fails", [False, True])
def test_cleanup_closes_once_and_keeps_the_original_failure(
    application_fails, close_fails
):
    calls = []
    original = RuntimeError("update failed")

    def close():
        calls.append("close")
        if close_fails:
            raise OSError("connection lost")

    training = SimpleNamespace(close=lambda: SimpleNamespace(result=close))

    def run():
        with close_after(training):
            if application_fails:
                raise original

    if application_fails:
        with pytest.raises(RuntimeError) as error:
            run()
        assert error.value is original
        assert getattr(original, "__notes__", []) == (
            ["Training close also failed: connection lost"] if close_fails else []
        )
    elif close_fails:
        with pytest.raises(OSError, match="connection lost"):
            run()
    else:
        run()
    assert calls == ["close"]
