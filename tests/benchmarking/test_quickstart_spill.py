"""Pool configurations cross worker boundaries without opening a pool in build/plan."""

import json

import pytest

from benchmarking.quickstart import main, orchestrate, run
from benchmarking.quickstart.spill import record, resolve
from shadowspill.memory import PinnedHostPool
from shadowspill.network import RemotePool
from shadowspill.ssd import SSDPool


@pytest.mark.parametrize("kind", ["pinned", "ssd", "remote"])
def test_configuration_roundtrip(tmp_path, kind):
    pool = {
        "pinned": lambda: PinnedHostPool(1 << 30),
        "ssd": lambda: SSDPool(
            1 << 30,
            directory=tmp_path,
            staging_bytes=96 << 20,
            chunk_bytes=4 << 20,
            queue_depth=8,
        ),
        "remote": lambda: RemotePool(1 << 30, host="unavailable", port=19426),
    }[kind]()
    request = json.loads(json.dumps({"spill": record(pool)}))
    assert resolve(request) == pool


def test_cli_records_spill_and_reuses_it_without_opening_backend(tmp_path, monkeypatch):
    phases = []
    monkeypatch.setattr(
        orchestrate, "_launch", lambda phase, *a, **kw: phases.append(phase)
    )
    directory = tmp_path / "not-on-this-host"
    assert (
        main(
            [
                "--phase",
                "build",
                "--factory",
                "unused:factory",
                "--output-dir",
                str(tmp_path),
                "--ssd-spill",
                str(directory),
                "--spill-gib",
                "2",
                "--ssd-staging-mib",
                "96",
                "--ssd-chunk-mib",
                "4",
                "--ssd-queue-depth",
                "8",
            ]
        )
        == 0
    )
    assert not directory.exists()
    assert (
        main(
            [
                "--phase",
                "plan",
                "--output-dir",
                str(tmp_path),
                "--search-budget-gib",
                "3",
                "--spill-gib",
                "4",
            ]
        )
        == 0
    )
    request = json.loads((tmp_path / "request.json").read_text())
    assert request["spill"] == {
        "type": "shadowspill.ssd:SSDPool",
        "arguments": {
            "capacity": 4 << 30,
            "directory": str(directory),
            "staging_bytes": 96 << 20,
            "chunk_bytes": 4 << 20,
            "queue_depth": 8,
        },
    }
    assert phases == ["build", "plan"]


def test_backend_options_conflict_and_capacity_matches_budget(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "--output-dir",
                str(tmp_path),
                "--remote-spill",
                "host:1234",
                "--ssd-spill",
                str(tmp_path),
            ]
        )
    with pytest.raises(ValueError, match="capacity"):
        run(
            "unused:factory",
            output_dir=tmp_path,
            search_budget_gib=[2],
            spill_gib=1,
            spill=PinnedHostPool(2 << 30),
        )
    assert not (tmp_path / "request.json").exists()


@pytest.mark.parametrize("peer", ["host", "host:0", "host:65536", ":1234"])
def test_remote_address_rejected_before_workers(tmp_path, peer):
    with pytest.raises(SystemExit):
        main(
            [
                "--phase",
                "build",
                "--factory",
                "unused:factory",
                "--output-dir",
                str(tmp_path),
                "--remote-spill",
                peer,
            ]
        )
