"""Backend choice survives saved phase requests without opening a backend."""

import json
from argparse import Namespace

import pytest

from benchmarking.spill import resolve
from qualification.numerical.prepare import load_request, save_request
from qualification.numerical.request import CaseRequest, PlannedRequest
from qualification.numerical.tolerances import SPILL_BUDGET
from shadowspill.network import RemotePool
from shadowspill.ssd import SSDPool


@pytest.mark.parametrize("kind", ["host", "ssd", "remote"])
def test_numerical_request_preserves_backend_and_reference_identity(tmp_path, kind):
    pool = {
        "host": lambda: None,
        "ssd": lambda: SSDPool(
            SPILL_BUDGET,
            directory=tmp_path,
            staging_bytes=64 << 20,
            chunk_bytes=2 << 20,
            queue_depth=4,
        ),
        "remote": lambda: RemotePool(1 << 30, host="unavailable", port=19426),
    }[kind]()
    case = CaseRequest("llama3", "pytorch", 123, {}, None, None, {})
    request = PlannedRequest(
        case,
        tmp_path / "reference.pt",
        tmp_path / "result.json",
        2 << 30,
        3,
        False,
        spill_pool=pool,
    )
    save_request(request, tmp_path)
    restored = load_request(tmp_path)
    assert restored == request
    assert restored.case.identity() == case.identity()
    settings = json.loads((tmp_path / "settings.json").read_text())
    assert resolve(settings) == pool
    assert settings["spill_bytes"] == (SPILL_BUDGET if pool is None else pool.capacity)


@pytest.mark.parametrize("planning_gib,expected_gib", [(None, 8), (4, 4), (16, 8)])
def test_performance_workers_share_the_bounded_peer_capacity(
    tmp_path, monkeypatch, planning_gib, expected_gib
):
    from qualification.performance import run

    arguments = Namespace(
        family="llama3",
        implementation="mlops",
        output=tmp_path / "result.json",
        spill_budget_gib=None,
        execution_budget_gib=None,
        external_headroom_mib=None,
        planning_spill_budget_gib=planning_gib,
        remote_spill=f"unavailable:19426:{8 << 30}",
        plan_only=False,
        retain_execution_files=False,
    )
    phases = []

    def launch(directory, phase, *args):
        phases.append(phase)
        if phase == "execute":
            arguments.output.write_text('{"passed": true}')

    def profile_and_plan(directory):
        settings = json.loads((directory / "settings.json").read_text())
        assert settings["spill_bytes"] == expected_gib << 30
        assert resolve(settings) == RemotePool(
            expected_gib << 30,
            host="unavailable",
            port=19426,
        )
        phases.extend(["profile", "plan"])

    monkeypatch.setattr(run, "run_phase", launch)
    monkeypatch.setattr(run, "profile_and_plan", profile_and_plan)
    assert run._run(arguments) == {"passed": True}
    assert phases == ["build", "profile", "plan", "execute"]
