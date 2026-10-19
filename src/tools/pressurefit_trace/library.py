"""Build a browsable, offline video library from saved plans and render reports.

This only catalogs existing artifacts. It neither renders nor changes a plan.
All workload names and run locations come from the input configuration.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import os
import shlex
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from .library_html import detail_html, index_html

VIDEO_NAMES = {
    "cuts": "Cuts · compute time",
    "cuts-task": "Cuts · task indices",
    "emission-placement": "Emission + placement",
    "execution": "Execution timelines",
}
MATRIX_FILES = {
    "cuts": "cuts-compute_4k.mp4",
    "emission-placement": "emission-placement_4k.mp4",
    "execution": "execution_4k.mp4",
}
FACT_KEYS = (
    "geometry",
    "microbatches",
    "ordering",
    "resolution",
    "candidate_id",
    "makespan_ns",
    "search_makespan_ns",
    "schedule_digest",
    "answer_reduction",
    "reductions",
    "cuts_per_reduction",
    "answer_cuts",
    "profiled_compute_seconds",
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, text):
    """Publish a complete document so a browser never sees half-written JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def href(target, parent):
    # Do not resolve symlinks: links through the library's data folders should
    # remain within the browsable library, including when served over HTTP.
    return quote(os.path.relpath(target, parent), safe="/")


def local_path(base, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def output_path(root, value):
    path = root / value
    if Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"library destination must be relative: {value}")
    return path


def plan_metrics(path, cache):
    """Use the certified simulation, not the logical search's makespan."""
    path = Path(path)
    stamp = [path.stat().st_mtime_ns, path.stat().st_size]
    key = str(path.resolve())
    cached = cache.get(key)
    if cached and cached["stamp"] == stamp:
        return copy.deepcopy(cached["metrics"])
    raw = read(path)
    result = raw["simulation"]["result"]
    tasks = result["task_intervals"]
    transfers = result["transfer_intervals"]
    layout = (raw.get("physical_admission") or {}).get("fixed_layout") or {}
    metrics = {
        "makespan_ns": result["makespan_ns"],
        "timing_basis": "Certified physical-plan simulation",
        "tasks": len(tasks),
        "compute_seconds": sum(
            t["end_ns"] - t["start_ns"]
            for t in tasks
            if t["resource_kind"] == "compute"
        )
        / 1e9,
        "task_wait_seconds": sum(t["start_ns"] - t["ready_ns"] for t in tasks) / 1e9,
        "execution_peak_bytes": max(
            (p["total_bytes"] for p in result.get("device_peaks", [])), default=0
        ),
        "spill_peak_bytes": result.get("spill_peak_bytes"),
        "layout_required_bytes": layout.get("required_bytes"),
        "pool_capacity_bytes": layout.get("pool_capacity_bytes"),
        "transfer_bandwidths": raw.get("transfer_bandwidths"),
        "capacity_violation_count": result.get("capacity_violation_count", 0),
    }
    for direction in ("fetch", "evict"):
        selected = [t for t in transfers if t["direction"] == direction]
        metrics[f"{direction}_bytes"] = sum(t["bytes"] for t in selected)
        metrics[f"{direction}_count"] = len(selected)
    cache[key] = {"stamp": stamp, "metrics": metrics}
    return copy.deepcopy(metrics)


def movie_record(spec, root):
    video = dict(spec)
    video.setdefault("label", VIDEO_NAMES.get(video.get("kind"), "Video"))
    path = output_path(root, video["path"])
    video.setdefault("status", "complete")
    if video["status"] == "complete" and not path.is_file():
        video["status"] = "missing"
    # Never make a partially written MP4 clickable.
    if video["status"] == "complete":
        video["url"] = href(path, root)
    else:
        video.pop("url", None)
    return video


def source_links(spec, base, root, page_dir):
    links = []
    for source in spec.get("sources", []):
        target = local_path(base, source["path"])
        if not target.exists():
            raise FileNotFoundError(target)
        label = source["label"]
        if Path(label).name != label or label in (".", ".."):
            raise ValueError(f"source label must be one filename: {label}")
        # Convenient local entry points; raw data remains in its canonical run.
        alias = page_dir / "Data" / label
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.is_symlink():
            if alias.resolve() != target.resolve():
                alias.unlink()
        elif alias.exists():
            raise FileExistsError(f"refusing to replace data: {alias}")
        if not alias.is_symlink():
            alias.symlink_to(target.resolve(), target_is_directory=target.is_dir())
        links.append(
            {
                "label": label,
                "url": href(alias, root),
                "source": str(target.resolve()),
            }
        )
    return links


def finish_record(spec, base, root, cache):
    row = copy.deepcopy(spec)
    row.setdefault("category", "Winner")
    row.setdefault("status", "ready")
    facts = dict(row.pop("facts", {}))
    if winner := row.pop("winner", None):
        saved = read(local_path(base, winner))
        facts = {**{k: saved[k] for k in FACT_KEYS if k in saved}, **facts}
    if plan := row.pop("plan", None):
        facts.update(plan_metrics(local_path(base, plan), cache))
    if row["status"] == "infeasible":
        facts["makespan_ns"] = None
    row.update(facts)
    ns = row.get("makespan_ns")
    row["step_seconds"] = ns / 1e9 if ns and ns > 0 else None
    units = row.get("tokens_per_step")
    row["tokens_per_second"] = (
        units / row["step_seconds"] if units and row["step_seconds"] else None
    )
    row.setdefault("timing_basis", "Recorded search simulation")
    if measurement := row.get("measurement"):
        seconds = measurement["step_seconds"]
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("measured step time must be finite and positive")
        if not measurement.get("aggregation"):
            raise ValueError("measurement must describe its timing aggregation")
        row["measured_step_seconds"] = seconds
        row["measured_tokens_per_second"] = units / seconds if units else None
        row["simulation_error"] = (
            row["step_seconds"] / seconds - 1 if row["step_seconds"] else None
        )
        row["useful_tflops_per_gpu"] = measurement.get("useful_tflops_per_gpu")
    reduction = row.get("answer_reduction")
    cuts = row.get("cuts_per_reduction") or []
    if (
        row.get("answer_cuts") is None
        and reduction is not None
        and 0 <= reduction < len(cuts)
    ):
        row["answer_cuts"] = cuts[reduction]
    if "profiled_compute_seconds" in row and "compute_seconds" not in row:
        row["compute_seconds"] = row["profiled_compute_seconds"]
    page_dir = output_path(root, row["directory"])
    row["page"] = href(page_dir / "index.html", root)
    row["readme"] = href(page_dir / "README.md", root)
    row["sources"] = source_links(row, base, root, page_dir)
    row["videos"] = [movie_record(v, root) for v in row.get("videos", [])]
    statuses = Counter(v["status"] for v in row["videos"])
    row["ready_videos"] = statuses["complete"]
    row["expected_videos"] = len(row["videos"])
    return row


def matrix_records(spec, base, root, cache):
    directory = local_path(base, spec["matrix"])
    matrix = read(directory / "matrix.json")
    progress_path = local_path(base, spec["progress"])
    progress = read(progress_path) if progress_path.exists() else {"movies": []}
    progress_by_key = {v["key"]: v for v in progress["movies"]}
    rows = []
    for case in matrix["cases"]:
        case_path = directory / case["path"]
        row = copy.deepcopy(spec["defaults"])
        row.update(
            {
                "id": spec["id"] + "/" + case["path"],
                "directory": str(Path(spec["directory"]) / case["path"]),
                "category": "Matrix",
                "status": case["status"],
                "facts": {k: case[k] for k in FACT_KEYS if k in case},
                "videos": [],
                "failures": case.get("failures", []),
                "sources": [{"label": "Raw configuration", "path": str(case_path)}],
            }
        )
        if case["status"] == "ready":
            row["plan"] = str(case_path / case["plan"])
            row["sources"] += [
                {"label": "Plan.json", "path": row["plan"]},
                {"label": "Cut trace", "path": str(case_path / case["trace"])},
                {"label": "Semantics.json", "path": str(case_path / case["semantics"])},
            ]
            for kind, filename in spec.get("video_files", MATRIX_FILES).items():
                rendered = progress_by_key.get(f"{case['path']}/{kind}", {})
                validation = rendered.get("validation", {})
                row["videos"].append(
                    {
                        "kind": kind,
                        "path": str(Path(row["directory"]) / filename),
                        "status": rendered.get("status", "queued"),
                        **{
                            k: validation[k]
                            for k in (
                                "seconds",
                                "width",
                                "height",
                                "fps",
                                "sha256",
                                "decode",
                            )
                            if k in validation
                        },
                    }
                )
        rows.append(finish_record(row, base, root, cache))
    return rows


def number(value, digits=0):
    return "—" if value is None else f"{value:,.{digits}f}"


def duration(seconds):
    if seconds is None:
        return "—"
    seconds = round(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def relative_url(url, root, parent):
    # URLs stored in results.json are relative to the library root.
    prefix = href(root, parent)
    return url if prefix == "." else prefix + "/" + url


def metric_rows(row):
    reduction = row.get("answer_reduction")
    measured = []
    if measurement := row.get("measurement"):
        measured = [
            ("Measured step time", number(row["measured_step_seconds"], 6) + " s"),
            (
                "Measured throughput",
                number(row["measured_tokens_per_second"], 2) + " tokens/s",
            ),
            (
                "Simulation error (simulated / measured - 1)",
                "—"
                if row["simulation_error"] is None
                else f"{row['simulation_error']:+.2%}",
            ),
            ("Useful TFLOP/s/GPU", number(row.get("useful_tflops_per_gpu"), 2)),
            ("Measurement aggregation", measurement["aggregation"]),
        ]
    return [
        ("Model", row["model"]),
        ("Training", row.get("training", "—")),
        ("Sequence length", number(row.get("sequence_length"))),
        ("Tokens per step", number(row.get("tokens_per_step"))),
        ("Device budget", f"{row['budget_gib']:g} GiB"),
        (
            "Spill pool",
            "—" if row.get("spill_gib") is None else f"{row['spill_gib']:g} GiB",
        ),
        ("Transfer assumption", row["transfer"]),
        ("Sequences per microbatch", str(row.get("geometry", "—"))),
        ("Microbatches per step", number(row.get("microbatches"))),
        ("Ordering", row.get("ordering", "—")),
        ("Recompute resolution", str(row.get("resolution", "—"))),
        ("Best policy", row.get("candidate_id") or "—"),
        ("Simulated step time", number(row.get("step_seconds"), 6) + " s"),
        ("Simulated throughput", number(row.get("tokens_per_second"), 2) + " tokens/s"),
        ("Timing evidence", row["timing_basis"]),
        *measured,
        ("GPU compute time", number(row.get("compute_seconds"), 6) + " s"),
        ("Sum of task waits", number(row.get("task_wait_seconds"), 6) + " s"),
        ("Total tasks", number(row.get("tasks"))),
        (
            "Answer reduction (1-based)",
            "—"
            if reduction is None
            else f"{reduction + 1} / {row.get('reductions', '—')}",
        ),
        ("Cuts in answer reduction", number(row.get("answer_cuts"))),
        (
            "Execution peak (includes task workspace)",
            number(row.get("execution_peak_bytes", 0) / 2**30, 3) + " GiB"
            if row.get("execution_peak_bytes") is not None
            else "—",
        ),
        (
            "Fetched",
            number(row.get("fetch_bytes", 0) / 2**30, 3) + " GiB"
            if row.get("fetch_bytes") is not None
            else "—",
        ),
        (
            "Evicted",
            number(row.get("evict_bytes", 0) / 2**30, 3) + " GiB"
            if row.get("evict_bytes") is not None
            else "—",
        ),
    ]


def detail_readme(row, root, directory):
    link = lambda url: relative_url(url, root, directory)  # noqa: E731
    lines = [
        f"# {row['model']} · {row['budget_gib']:g} GiB · {row['transfer']}",
        "",
        f"[Open video page]({link(row['page'])}) · "
        f"[Library index]({href(root / 'index.html', directory)})",
        "",
        f"{row['category']} · {row.get('edition', '')}",
        "",
        "Simulated throughput uses the saved plan's step time. Measured throughput, "
        "when available, uses separately recorded training steps. Movie playback "
        "follows teaching pace.",
        "",
    ]
    if row.get("note"):
        lines += [row["note"], ""]
    lines += ["| Statistic | Value |", "| --- | --- |"]
    lines += [f"| {label} | {value} |" for label, value in metric_rows(row)]
    lines += [
        "",
        "## Videos",
        "",
        "| Video | Status | Length | Resolution |",
        "| --- | --- | ---: | --- |",
    ]
    for v in row["videos"]:
        label = f"[{v['label']}]({link(v['url'])})" if v.get("url") else v["label"]
        lines.append(
            f"| {label} | {v['status']} | {duration(v.get('seconds'))} | "
            f"{v.get('width', '—')} x {v.get('height', '—')} |"
        )
    if row["status"] == "infeasible":
        lines += ["", "No successful plan or videos for this configuration.", ""]
        for failure in row.get("failures", []):
            detail = failure.get("detail", failure.get("kind"))
            lines.append(f"- **{failure['candidate_id']}**: {detail}")
    lines += ["", "## Saved inputs", ""]
    lines += [f"- [{s['label']}]({link(s['url'])})" for s in row["sources"]]
    if row.get("related"):
        lines += ["", "## Also in this folder", ""]
        lines += [f"- [{s['label']}]({link(s['url'])})" for s in row["related"]]
    lines += [
        "",
        "Data links point to preserved local run directories. Keep those "
        "directories when moving this library, or copy with symlinks dereferenced.",
        "",
    ]
    return "\n".join(lines)


def results_table(rows, root, parent):
    lines = [
        "| Configuration | Seqs/mb | Order | Recompute | Sim. step (s) | "
        "Sim. tokens/s | Measured tokens/s | Videos |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        label = f"{r['model']} · {r['budget_gib']:g} GiB · {r['transfer']}"
        if r.get("edition"):
            label += f" · {r['edition']}"
        videos = []
        for v in r["videos"]:
            name = v["label"]
            if v.get("url"):
                videos.append(f"[{name}]({relative_url(v['url'], root, parent)})")
            else:
                videos.append(f"{name}: {v['status']}")
        if r["status"] == "infeasible":
            videos = ["Infeasible; see configuration for reasons"]
        page_url = relative_url(r["page"], root, parent)
        lines.append(
            f"| [{label}]({page_url}) | {r.get('geometry', '—')} | "
            f"{r.get('ordering', '—')} | {r.get('resolution', '—')} | "
            f"{number(r.get('step_seconds'), 3)} | "
            f"{number(r.get('tokens_per_second'))} | "
            f"{number(r.get('measured_tokens_per_second'))} | "
            + " · ".join(videos)
            + " |"
        )
    return "\n".join(lines)


def build(config_path, root):
    config_path, root = Path(config_path).resolve(), Path(root).absolute()
    config = read(config_path)
    if config.get("schema") != 1:
        raise ValueError("library config must have schema: 1")
    base = config_path.parent
    cache_path = root / "_catalog" / "metrics_cache.json"
    cache = read(cache_path) if cache_path.exists() else {}
    rows = [finish_record(s, base, root, cache) for s in config.get("entries", [])]
    for spec in config.get("matrices", []):
        rows.extend(matrix_records(spec, base, root, cache))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate configuration IDs")
    if len({r["directory"] for r in rows}) != len(rows):
        raise ValueError("each configuration needs its own page directory")
    for row in rows:
        prefix = row["directory"] + "/"
        children = sorted(
            {
                r["directory"][len(prefix) :].split("/")[0]
                for r in rows
                if r["directory"].startswith(prefix)
            }
        )
        row["related"] = [
            {
                "label": name,
                "url": href(root / row["directory"] / name / "index.html", root),
            }
            for name in children
        ]
    counts = Counter(v["status"] for r in rows for v in r["videos"])
    matrix_counts = Counter(
        v["status"] for r in rows if r["category"] == "Matrix" for v in r["videos"]
    )
    document = {
        "schema": 1,
        "title": config.get("title", "Video library"),
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "counts": dict(counts),
        "matrix_counts": dict(matrix_counts),
        "notes": config.get("notes", []),
        "entries": rows,
    }
    assets = Path(__file__).with_name("library_assets")
    for name in ("style.css", "index.js"):
        write(root / "_catalog" / name, (assets / name).read_text())
    write(cache_path, json_text(cache))
    write(root / "results.json", json_text(document))
    fieldnames = [
        "id",
        "model",
        "category",
        "edition",
        "budget_gib",
        "spill_gib",
        "transfer",
        "training",
        "sequence_length",
        "tokens_per_step",
        "geometry",
        "microbatches",
        "ordering",
        "resolution",
        "candidate_id",
        "status",
        "timing_basis",
        "makespan_ns",
        "search_makespan_ns",
        "step_seconds",
        "tokens_per_second",
        "measured_step_seconds",
        "measured_tokens_per_second",
        "simulation_error",
        "useful_tflops_per_gpu",
        "compute_seconds",
        "task_wait_seconds",
        "tasks",
        "answer_reduction",
        "reductions",
        "answer_cuts",
        "execution_peak_bytes",
        "fetch_bytes",
        "evict_bytes",
        "ready_videos",
        "expected_videos",
        "page",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    write(root / "results.csv", stream.getvalue())
    write(root / "index.html", index_html(document, root, root))
    groups = {}
    for row in rows:
        directory = output_path(root, row["directory"])
        write(directory / "index.html", detail_html(row, document, root, directory))
        write(directory / "README.md", detail_readme(row, root, directory))
        write(directory / "result.json", json_text(row))
        parent = directory.parent
        while parent != root:
            groups.setdefault(parent, []).append(row)
            parent = parent.parent
    occupied = {output_path(root, r["directory"]) for r in rows}
    for directory, children in groups.items():
        if directory in occupied:
            continue
        subdocument = {**document, "title": directory.name, "entries": children}
        write(directory / "index.html", index_html(subdocument, root, directory))
        # The renderer owns matrix README.md; never compete with that writer.
        write(
            directory / "CONTENTS.md",
            f"# {directory.name}\n\n" + results_table(children, root, directory) + "\n",
        )
    lines = [
        f"# {document['title']}",
        "",
        "**[Open the searchable video catalog](index.html)** · "
        "[Results CSV](results.csv) · [Results JSON](results.json)",
        "",
        f"Updated {document['updated_at']}. **{counts['complete']} videos ready**. "
        f"Expanded matrix: **{matrix_counts['complete']} / "
        f"{sum(matrix_counts.values())} ready**, {matrix_counts['running']} rendering, "
        f"{matrix_counts['queued']} queued, {matrix_counts['failed']} failed.",
        "",
        "Browse by model → budget → transfer assumption. Each configuration has "
        "an HTML video page, README, statistics, and links to its raw inputs. "
        "The catalog works offline; open index.html in a browser. Refresh the "
        "page for new videos while the background catalog updater is running.",
        "",
        "Simulated throughput uses certified physical-plan timing wherever "
        "available; older cut-only traces identify their search-simulation basis. "
        "Measured throughput is shown separately when training steps were recorded. "
        "Movie length is teaching pace, not execution time.",
        "",
    ]
    lines += [f"- {note}" for note in document["notes"]]
    if (root / "Data and rerendering.md").exists():
        lines += ["", "[Saved data and rerendering guide](Data%20and%20rerendering.md)"]
    for category in ("Winner", "Matrix", "Earlier cut", "Archive"):
        selected = [r for r in rows if r["category"] == category]
        if selected:
            lines += [
                "",
                f"## {category} configurations",
                "",
                results_table(selected, root, root),
            ]
    lines += [
        "",
        "## Rebuild this index",
        "",
        "The configuration is saved in "
        "`_catalog/sources.json`. From the repository with the shadowspill "
        "environment active:",
        "",
        "```sh",
        "PYTHONPATH=src python -m "
        "tools.pressurefit_trace.library " + shlex.quote(str(config_path)) + " \\",
        "  --output " + shlex.quote(str(root)) + " --watch 30",
        "```",
        "",
        "The updater exits when every requested matrix video has a final "
        "status. It does not render videos or alter raw plans/traces. "
        "Data folders are local symlinks to preserved run directories; "
        "keep their targets or dereference the links when copying the library.",
        "",
    ]
    write(root / "README.md", "\n".join(lines))
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Library source configuration JSON")
    parser.add_argument(
        "--output", type=Path, required=True, help="Published video library"
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=0,
        metavar="SECONDS",
        help="Refresh on progress changes; exit when matrix jobs finish",
    )
    args = parser.parse_args()
    if args.watch and args.watch < 1:
        parser.error("--watch must be at least one second")
    while True:
        document = build(args.config, args.output)
        print(
            json.dumps(
                {
                    "updated_at": document["updated_at"],
                    "counts": document["counts"],
                    "matrix_counts": document["matrix_counts"],
                }
            ),
            flush=True,
        )
        pending = any(
            document["matrix_counts"].get(k, 0)
            for k in ("queued", "running", "missing")
        )
        if not args.watch or not pending:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
