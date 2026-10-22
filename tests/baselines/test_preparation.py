"""Twenty-five independent cases share five compatible preparations."""

import dataclasses

from baselines.common.preparation import groups
from baselines.engines.shadowspill.settings import Settings


def test_budget_grid_keeps_cases_and_groups_only_compatible_preparation(tmp_path):
    cases = [
        (
            f"mb{size}-ex{budget}",
            Settings.from_dict(
                {
                    "model": "llama3-tiny",
                    "sequence_length": 4096,
                    "tokens_per_step": 65536,
                    "tokens_per_microbatch": size,
                    "execution_gib": budget,
                    "spill_gib": 144,
                }
            ),
        )
        for size in (4096, 8192, 16384, 32768, 65536)
        for budget in (8, 12, 16, 24, 28)
    ]
    manifest = {"source": {"revision": "one"}, "environment": {"torch": "test"}}
    grouped = groups(
        cases, engine="shadowspill", world_size=1, manifest=manifest, output=tmp_path
    )
    assert len(cases) == len(grouped) == 25
    assert len({str(group["directory"]) for group in grouped.values()}) == 5
    assert all(len(group["members"]) == 5 for group in grouped.values())
    assert grouped["mb4096-ex8"] is grouped["mb4096-ex28"]
    assert grouped["mb4096-ex8"] is not grouped["mb8192-ex8"]
    changed = groups(
        cases,
        engine="shadowspill",
        world_size=1,
        manifest={**manifest, "source": {"revision": "two"}},
        output=tmp_path,
    )
    assert changed["mb4096-ex8"]["directory"] != grouped["mb4096-ex8"]["directory"]
    assert not groups(
        cases, engine="shadowspill", world_size=2, manifest=manifest, output=tmp_path
    )
    fresh = [
        (name, dataclasses.replace(settings, artifact_store="none"))
        for name, settings in cases
    ]
    assert not groups(
        fresh, engine="shadowspill", world_size=1, manifest=manifest, output=tmp_path
    )


def test_preparation_resumes_and_preserves_requests_when_extending_budgets(
    tmp_path, monkeypatch
):
    import json

    from baselines import sweep
    from baselines.common.preparation import ensure

    settings = Settings.from_dict({"model": "llama3-tiny", "execution_gib": 8})
    group = {
        "identity": {"test": True},
        "directory": tmp_path / "group",
        "members": [("first", settings)],
    }
    calls = []

    def run(command, worker, *args, **kwargs):
        calls.append(command)
        request = json.loads((worker / "request.json").read_text())
        artifact = group["directory"] / "artifact"
        artifact.mkdir(exist_ok=True)
        (artifact / "training.json").write_text("{}")
        (worker / "preparation-result.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "budgets": request["budgets"],
                    "artifact": str(artifact),
                }
            )
        )
        return kwargs["classify_result"](worker, 0, 1, settings)

    monkeypatch.setattr(sweep, "run_case", run)
    arguments = dict(engine="shadowspill", python="python", world_size=1, timeout=60)
    first = ensure(group, **arguments)
    request = (group["directory"] / "attempt-001" / "request.json").read_text()
    assert ensure(group, **arguments) == first and len(calls) == 1
    group["members"].append(("second", dataclasses.replace(settings, execution_gib=16)))
    assert ensure(group, **arguments) == first and len(calls) == 2
    assert (group["directory"] / "attempt-001" / "request.json").read_text() == request
    assert (group["directory"] / "attempt-002" / "request.json").is_file()
    assert not list(group["directory"].rglob("row.json"))
