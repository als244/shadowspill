"""Catalog results must match the displayed plan and link only finished movies."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

import pytest

from tools.pressurefit_trace.library import build, finish_record, write


def save(path, value):
    write(path, json.dumps(value))
    return path


def saved_plan(tmp_path):
    return save(
        tmp_path / "plan.json",
        {
            "simulation": {
                "result": {
                    "makespan_ns": 2_000_000_000,
                    "task_intervals": [
                        {
                            "resource_kind": "control",
                            "ready_ns": 0,
                            "start_ns": 0,
                            "end_ns": 0,
                        },
                        {
                            "resource_kind": "compute",
                            "ready_ns": 0,
                            "start_ns": 500_000_000,
                            "end_ns": 1_500_000_000,
                        },
                    ],
                    "transfer_intervals": [
                        {"direction": "fetch", "bytes": 1024},
                        {"direction": "evict", "bytes": 512},
                    ],
                    "device_peaks": [{"total_bytes": 1234}],
                    "spill_peak_bytes": 1024,
                    "capacity_violation_count": 0,
                }
            },
            "physical_admission": {"fixed_layout": {"required_bytes": 1400}},
        },
    )


def entry(plan):
    return {
        "id": "example",
        "model": "Model & <example>",
        "budget_gib": 16,
        "transfer": "Measured bandwidth",
        "tokens_per_step": 1000,
        "directory": "Example model/16 GiB/Measured bandwidth",
        "plan": str(plan),
        "facts": {"makespan_ns": 1_000_000_000, "search_makespan_ns": 1_000_000_000},
        "sources": [{"label": "Plan.json", "path": str(plan)}],
    }


def test_certified_makespan_and_waits_override_logical_search(tmp_path):
    row = finish_record(entry(saved_plan(tmp_path)), tmp_path, tmp_path / "library", {})
    assert row["step_seconds"] == 2
    assert row["tokens_per_second"] == 500
    assert row["search_makespan_ns"] == 1_000_000_000
    assert row["compute_seconds"] == 1
    # Final copy drain is not a task wait: step - compute would give 1 second.
    assert row["task_wait_seconds"] == 0.5
    assert row["fetch_bytes"] == 1024 and row["evict_bytes"] == 512


def test_measured_throughput_stays_separate_from_certified_simulation(tmp_path):
    spec = entry(saved_plan(tmp_path))
    spec["measurement"] = {
        "step_seconds": 4,
        "aggregation": "Median of steps 2-6; slowest rank per step across eight ranks",
        "useful_tflops_per_gpu": 300,
    }
    root = tmp_path / "library"
    document = build(
        save(tmp_path / "sources.json", {"schema": 1, "entries": [spec]}), root
    )
    row = document["entries"][0]
    assert row["step_seconds"] == 2 and row["tokens_per_second"] == 500
    assert row["measured_step_seconds"] == 4
    assert row["measured_tokens_per_second"] == 250
    assert row["simulation_error"] == -0.5
    assert row["useful_tflops_per_gpu"] == 300
    detail = root / spec["directory"]
    for path in (detail / "index.html", detail / "README.md", root / "index.html"):
        content = path.read_text()
        assert "Measured throughput" in content and "250" in content
        assert "All throughput values are simulated" not in content
    assert "slowest rank per step" in (detail / "README.md").read_text()
    assert "measured_tokens_per_second" in (root / "results.csv").read_text()


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_measurement_is_not_published(tmp_path, seconds):
    spec = entry(saved_plan(tmp_path))
    spec["measurement"] = {"step_seconds": seconds, "aggregation": "Median"}
    with pytest.raises(ValueError, match="measured step time"):
        finish_record(spec, tmp_path, tmp_path / "library", {})


def test_measurement_requires_aggregation_provenance(tmp_path):
    spec = entry(saved_plan(tmp_path))
    spec["measurement"] = {"step_seconds": 2}
    with pytest.raises(ValueError, match="aggregation"):
        finish_record(spec, tmp_path, tmp_path / "library", {})


def test_partial_movies_are_never_linked_and_infeasible_has_no_rate(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    (root / "partial.mp4").write_bytes(b"partial")
    spec = entry(saved_plan(tmp_path))
    spec["status"] = "infeasible"
    spec["videos"] = [
        {"kind": "execution", "path": "partial.mp4", "status": "running"},
        {"kind": "cuts", "path": "missing.mp4", "status": "complete"},
    ]
    row = finish_record(spec, tmp_path, root, {})
    assert row["step_seconds"] is None and row["tokens_per_second"] is None
    assert all("url" not in v for v in row["videos"])
    assert [v["status"] for v in row["videos"]] == ["running", "missing"]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("href", "src", "data-src") and value:
                self.targets.append(value)


def test_library_links_and_escaping_and_rebuild(tmp_path):
    root = tmp_path / "library"
    spec = entry(saved_plan(tmp_path))
    movie = root / spec["directory"] / "movie & study.mp4"
    movie.parent.mkdir(parents=True)
    movie.write_bytes(b"published")
    spec["videos"] = [{"kind": "execution", "path": str(movie.relative_to(root))}]
    config = save(tmp_path / "sources.json", {"schema": 1, "entries": [spec]})
    first = build(config, root)
    second = build(config, root)
    assert first["entries"] == second["entries"]
    assert first["counts"] == {"complete": 1}
    assert "Model &amp; &lt;example&gt;" in (root / "index.html").read_text()
    assert "%26" in (root / "index.html").read_text()
    for page in root.rglob("*.html"):
        parser = Links()
        parser.feed(page.read_text())
        for target in parser.targets:
            path = urlsplit(target).path
            assert (page.parent / unquote(path)).exists(), (page, target)
    assert (root / spec["directory"] / "Data/Plan.json").is_symlink()
    assert "500.0" in (root / "results.csv").read_text()


def test_matrix_pending_counts_and_renderer_readme_ownership(tmp_path):
    plan = saved_plan(tmp_path)
    raw = tmp_path / "raw"
    (raw / "ready/trace").mkdir(parents=True)
    (raw / "infeasible").mkdir()
    (raw / "ready/plan.json").write_bytes(plan.read_bytes())
    (raw / "ready/semantics.json").write_text("{}")
    save(
        raw / "matrix.json",
        {
            "cases": [
                {
                    "path": "ready",
                    "status": "ready",
                    "plan": "plan.json",
                    "semantics": "semantics.json",
                    "trace": "trace",
                    "makespan_ns": 1,
                },
                {"path": "infeasible", "status": "infeasible", "makespan_ns": None},
            ]
        },
    )
    defaults = {
        "model": "Test",
        "budget_gib": 16,
        "transfer": "Measured",
        "tokens_per_step": 1000,
    }
    config = save(
        tmp_path / "sources.json",
        {
            "schema": 1,
            "matrices": [
                {
                    "id": "matrix",
                    "directory": "Test/16 GiB/Matrix",
                    "matrix": str(raw),
                    "progress": str(tmp_path / "progress.json"),
                    "defaults": defaults,
                }
            ],
        },
    )
    root = tmp_path / "library"
    batch_readme = root / "Test/16 GiB/Matrix/README.md"
    write(batch_readme, "Renderer owns this file")
    result = build(config, root)
    assert result["matrix_counts"] == {"queued": 3}
    assert result["entries"][0]["tokens_per_second"] == 500
    assert result["entries"][1]["tokens_per_second"] is None
    assert batch_readme.read_text() == "Renderer owns this file"


def test_rejects_destination_outside_library(tmp_path):
    spec = entry(saved_plan(tmp_path))
    spec["directory"] = "../escape"
    with pytest.raises(ValueError, match="must be relative"):
        finish_record(spec, tmp_path, tmp_path / "library", {})


def test_missing_cut_counts_stay_unknown(tmp_path):
    spec = entry(saved_plan(tmp_path))
    spec["facts"]["answer_reduction"] = 2
    row = finish_record(spec, tmp_path, tmp_path / "library", {})
    assert row.get("answer_cuts") is None
