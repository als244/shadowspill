"""Render case progress and the sweep result index."""

from __future__ import annotations

import datetime
import json
from collections import Counter

from baselines.common import write_json
from baselines.common.results import varied
from baselines.report import DERIVED_GEOMETRY

from .results import described, latest

OUTCOME_WORDS = {
    "completed": "completed",
    "cuda_out_of_memory": "out of device memory",
    "failed": "failed",
    "timeout": "timed out",
    "interrupted": "interrupted",
    "pending": "not yet run",
}


INDEX_COLUMNS = (
    "Status",
    "s/update",
    "tok/s",
    "TFLOP/s/GPU",
    "GPU peak GiB",
    "Host peak GiB",
    "Evidence",
)


RULE_WIDTH = 72


def clock():
    """The local time, as the console shows it."""

    return datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def duration(seconds):
    """``1h 02m 03s`` or ``2m 03s``."""

    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def tally(statuses):
    """``3 completed, 1 out of device memory, 8 not yet run``."""

    order = list(OUTCOME_WORDS)
    counts = Counter(statuses)
    return ", ".join(
        f"{counts[status]} {OUTCOME_WORDS.get(status, status)}"
        for status in sorted(
            counts, key=lambda item: order.index(item) if item in order else len(order)
        )
    )


def labels(cases, world_size):
    """The settings the matrix varies, those the geometry derives from them
    left out, and per case ``key=value`` for each."""

    values = described(cases, world_size)
    knobs = [key for key in varied(values) if key not in DERIVED_GEOMETRY]
    return knobs, {
        name: " ".join(
            f"{key}={value if isinstance(value, str) else json.dumps(value)}"
            for key in knobs
            for value in [values[name].get(key)]
        )
        for name in values
    }


def rule(title):
    """A section's heading: ``── title ───``."""

    head = f"── {title} "
    return head + "─" * max(0, RULE_WIDTH - len(head))


def field(name, value):
    """One labeled line of a block, its values aligned."""

    return f"  {name:<13}{value}"


def geometry_text(settings, world_size):
    """A case's microbatches as the console describes them."""

    geometry = settings.geometry(world_size)
    if getattr(settings, "tokens_per_microbatch", 0) is None:
        sizes = settings.microbatch_candidates(world_size)
        return (
            f"searched: {sizes[-1]:,} to {sizes[0]:,} tokens per microbatch,"
            f" {geometry.sequence_length:,}-token sequences"
        )
    return (
        f"{geometry.sequences_per_microbatch} x {geometry.sequence_length:,}-token"
        f" sequences per microbatch, {geometry.accumulation} microbatches per update"
    )


def started_block(position, name, label, settings, world_size, attempt):
    """A case as it starts: what it runs and where its output goes."""

    lines = [
        f"CASE START {position} {name}",
        field("settings", label or "the matrix's defaults"),
        field("geometry", geometry_text(settings, world_size)),
    ]
    if attempt.name != "attempt-001":
        lines.append(field("attempt", f"{attempt.name}, a retry"))
    lines.extend(
        [
            field("log", attempt / "console.log"),
            field("start", clock()),
        ]
    )
    return lines


def ended_block(position, name, outcome, attempt):
    """A case as it ends: its outcome and, when it completed, what it
    measured."""

    status = outcome["status"]
    lines = [f"CASE {OUTCOME_WORDS.get(status, status).upper()} {position} {name}"]
    if status == "completed":
        host = outcome.get("host_peak_bytes", outcome["host_peak_rss_bytes"])
        lines.append(
            field(
                "throughput",
                f"{outcome['global_tokens_per_second']:,.1f} tok/s,"
                f" {outcome['step_seconds']:.2f} s/update",
            )
        )
        lines.append(
            field(
                "memory",
                f"GPU {outcome['peak_reserved_bytes'] / 2**30:.2f} GiB held"
                f" ({outcome['peak_allocated_bytes'] / 2**30:.2f} in use),"
                f" host {host / 2**30:.2f} GiB",
            )
        )
        if not outcome.get("performance_result", True):
            lines.append(
                field("note", "a qualification or profile, not a throughput result")
            )
    elif outcome.get("reason"):
        lines.append(field("reason", outcome["reason"]))
    if "wall_seconds" in outcome:
        lines.append(
            field("duration", f"{duration(outcome['wall_seconds'])}, ended {clock()}")
        )
    if status != "completed":
        lines.append(field("log", attempt / "console.log"))
    if (attempt / "row.json").is_file():
        lines.append(field("row", attempt / "row.json"))
    return lines


def _cell(value, form):
    return "—" if value is None else form.format(value)


def report(output, cases, engine, world_size):
    """``results.json`` and the ``README.md`` index of every case's latest
    outcome, with a column per setting the matrix varies."""

    rows, values = [], described(cases, world_size)
    knobs = varied(values)
    text = [
        f"# {engine} sweep",
        "",
        "[Exact manifest and commands](manifest.json). Throughput is global; "
        "device peaks are PyTorch allocator measurements; a completed run that "
        "was a qualification or a profile is not a throughput result.",
        "",
        "| Case | " + " | ".join([*knobs, *INDEX_COLUMNS]) + " |",
        "|---|" + "---|" * (len(knobs) + len(INDEX_COLUMNS)),
    ]
    for name, settings in cases:
        outcome = latest(output / "cases" / name)
        host = outcome.get("host_peak_bytes", outcome.get("host_peak_rss_bytes"))
        rows.append({"case": name, "settings": settings.as_dict(), **outcome})
        status = outcome["status"]
        eligible = status == "completed" and outcome.get("performance_result", False)
        if status == "completed" and not eligible:
            status += " (qualification/profile)"
        evidence = f"[config](cases/{name}/config.json)"
        if "attempt" in outcome:
            path = f"cases/{name}/{outcome['attempt']}"
            evidence += f" · [log]({path}/console.log) · [outcome]({path}/outcome.json)"
            if (output / path / "row.json").is_file():
                evidence += f" · [row]({path}/row.json)"
        cells = [
            name,
            *(json.dumps(values[name].get(key)) for key in knobs),
            status,
            _cell(outcome.get("step_seconds") if eligible else None, "{:.4f}"),
            _cell(
                outcome.get("global_tokens_per_second") if eligible else None, "{:,.0f}"
            ),
            _cell(outcome.get("useful_tflops_per_gpu") if eligible else None, "{:.1f}"),
            _cell(outcome.get("peak_allocated_bytes"), "{:.2f}")
            if outcome.get("peak_allocated_bytes") is None
            else f"{outcome['peak_allocated_bytes'] / 2**30:.2f}",
            _cell(None if host is None else host / 2**30, "{:.2f}"),
            evidence,
        ]
        text.append("| " + " | ".join(cells) + " |")
    write_json(
        output / "results.json",
        {"engine": engine, "world_size": world_size, "cases": rows},
    )
    (output / "README.md").write_text("\n".join(text) + "\n")
