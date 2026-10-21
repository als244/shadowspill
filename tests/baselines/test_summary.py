"""An experiment's summary: each engine's outcomes, the fastest case of each
within each GPU budget and each engine's frontier, from the rows its sweeps
hold, whichever run wrote them."""

import json

import pytest

from baselines import summary
from baselines.common import write_json
from baselines.common.results import memory_peaks, row, summarize_cycles

MODEL = {"model": "llama3-8b", "implementation": "mlops", "mode": "full"}
GEOMETRY = {"sequence_length": 4096, "world_size": 1}
HARDWARE = {"device": {"name": "RTX 5090", "count": 1}}


def completed(engine, tokens_per_second, gpu_gib, commit=None, **configuration):
    measurement = summarize_cycles([[4096 / tokens_per_second]], global_tokens=4096)
    peaks = {
        "peak_allocated_bytes": int(gpu_gib * 0.75 * 2**30),
        "peak_reserved_bytes": int(gpu_gib * 2**30),
        "host_peak_rss_bytes": 64 * 2**30,
    }
    return row(
        engine=engine,
        status="completed",
        model=MODEL,
        geometry=GEOMETRY,
        precision={"model_dtype": "bfloat16"},
        configuration=configuration,
        hardware=HARDWARE,
        provenance={"source": {"git_head": commit, "git_dirty": False}},
        measurement=measurement,
        memory=memory_peaks([peaks]),
    )


def boundary(engine, **configuration):
    return row(
        engine=engine,
        status="cuda_out_of_memory",
        model=MODEL,
        geometry=GEOMETRY,
        precision={"model_dtype": "bfloat16"},
        configuration=configuration,
        hardware=HARDWARE,
        provenance={},
        failure={"reason": "out of memory"},
    )


def sweep(directory, engine, cases):
    """A sweep's directory as the sweep leaves it: its manifest, its index of
    latest outcomes and each attempted case's row."""

    root = directory / "sweeps" / engine
    write_json(
        root / "manifest.json",
        {"engine": engine, "cases": [{"name": name} for name in cases]},
    )
    write_json(
        root / "results.json",
        {
            "engine": engine,
            "cases": [
                {"case": name, "status": status} for name, (status, _) in cases.items()
            ],
        },
    )
    for name, (_, value) in cases.items():
        if value is not None:
            write_json(root / "cases" / name / "attempt-001" / "row.json", value)


@pytest.fixture
def experiment(tmp_path):
    spec = {
        "sweeps": [
            {"name": name, "engine": name}
            for name in ("shadowspill", "fsdp2", "deepspeed")
        ],
        "summary_budgets_gib": [8, 12, 16, 32],
    }
    write_json(tmp_path / "experiment.json", {"spec": spec})
    sweep(
        tmp_path,
        "shadowspill",
        {
            "case-0001-a": (
                "completed",
                completed("shadowspill", 2000, 8, commit="fadc671c0000", budget=8),
            ),
            "case-0002-b": ("completed", completed("shadowspill", 3000, 16, budget=16)),
            "case-0003-c": ("completed", completed("shadowspill", 2900, 24, budget=24)),
        },
    )
    sweep(
        tmp_path,
        "fsdp2",
        {
            "case-0001-d": ("completed", completed("fsdp2", 1500, 12.8, recompute=1)),
            "case-0002-e": ("completed", completed("fsdp2", 1490, 17.2, recompute=0)),
            "case-0003-f": ("cuda_out_of_memory", boundary("fsdp2", recompute=0.5)),
            "case-0004-g": ("pending", None),
        },
    )
    return tmp_path


def test_the_fastest_case_of_each_engine_within_each_budget(experiment):
    result = summary.summarize(experiment)
    assert result["budgets_gib"] == [8, 12, 16, 32]
    (group,) = result["groups"]
    assert group["engines"] == ["shadowspill", "fsdp2"]
    fastest = {
        budget["budget_gib"]: {
            engine: None if entry is None else entry["tokens_per_second"]
            for engine, entry in budget["engines"].items()
        }
        for budget in group["best"]
    }
    assert fastest == {
        8: {"shadowspill": pytest.approx(2000), "fsdp2": None},
        12: {"shadowspill": pytest.approx(2000), "fsdp2": None},
        16: {"shadowspill": pytest.approx(3000), "fsdp2": pytest.approx(1500)},
        32: {"shadowspill": pytest.approx(3000), "fsdp2": pytest.approx(1500)},
    }
    best = group["best"][2]["engines"]["fsdp2"]
    assert best["settings"] == "recompute=1"
    assert best["gpu_in_use_gib"] == pytest.approx(12.8 * 0.75, abs=1e-6)
    assert best["row"] == "sweeps/fsdp2/cases/case-0001-d/attempt-001/row.json"


def test_each_engines_frontier_runs_fastest_first(experiment):
    (group,) = summary.summarize(experiment)["groups"]
    frontier = {
        engine: [round(entry["tokens_per_second"]) for entry in entries]
        for engine, entries in group["frontiers"].items()
    }
    assert frontier == {"shadowspill": [3000, 2000], "fsdp2": [1500]}


def test_every_engine_the_spec_names_is_counted(experiment):
    counted = {
        engine["sweep"]: summary.counts(engine)
        for engine in summary.summarize(experiment)["sweeps"]
    }
    assert counted == {
        "shadowspill": "3 cases: 3 completed",
        "fsdp2": "4 cases: 2 completed, 1 out of device memory, 1 not yet run",
        "deepspeed": "not run",
    }


def test_the_summary_is_printed_and_written(experiment, capsys):
    summary.main([str(experiment)])
    text = capsys.readouterr().out
    assert "fastest within each GPU budget:" in text and "nothing fits" in text
    assert "3,000.0 tok/s" in text and "budget=16" in text
    written = (experiment / "summary.md").read_text()
    assert "## Frontier of each engine, fastest first" in written
    assert "| Sweep | Engine | Outcomes |" in written
    data = json.loads((experiment / "summary.json").read_text())
    assert data["sweeps"][2] == {
        "sweep": "deepspeed",
        "engine": "deepspeed",
        "cases": None,
        "outcomes": {},
    }


def test_without_budgets_only_the_frontiers_are_given(experiment):
    (group,) = summary.summarize(experiment, budgets=[])["groups"]
    assert group["best"] == []
    assert "fastest within" not in summary.render(summary.summarize(experiment, []))


def test_the_summary_names_the_commits_and_embeds_the_reports_figures(experiment):
    from baselines.report import slug

    key = ("llama3-8b", "full", 4096, 0, "RTX 5090", 1)
    figures = experiment / "report" / "figures"
    figures.mkdir(parents=True)
    for name in ("device", "host"):
        (figures / f"{slug(key)}-{name}.png").write_bytes(b"png")
    result = summary.summarize(experiment)
    assert result["commits"][0] == {"commit": None, "dirty": False, "rows": 5}
    assert {"commit": "fadc671c0000", "dirty": False, "rows": 1} in result["commits"]
    (group,) = result["groups"]
    assert group["figures"] == [
        f"report/figures/{slug(key)}-device.png",
        f"report/figures/{slug(key)}-host.png",
    ]
    text = summary.markdown(result)
    for name in ("device", "host"):
        assert f"![{group['title']}](report/figures/{slug(key)}-{name}.png)" in text


def test_a_published_summary_keeps_its_figures_and_no_path(experiment, tmp_path):
    from baselines.report import slug

    key = ("llama3-8b", "full", 4096, 0, "RTX 5090", 1)
    figure = experiment / "report" / "figures" / f"{slug(key)}-device.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"png")
    target = tmp_path / "published"
    summary.publish(experiment, summary.summarize(experiment), target)
    assert (target / "figures" / f"{slug(key)}-device.png").read_bytes() == b"png"
    text = (target / "summary.md").read_text()
    assert f"(figures/{slug(key)}-device.png)" in text
    assert str(experiment) not in text and "row.json" not in text
    data = json.loads((target / "summary.json").read_text())
    assert "experiment" not in data and '"row":' not in json.dumps(data)
