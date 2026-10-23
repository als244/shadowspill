"""Publish matched throughput curves and grouped runtime bars for three phases."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

from baselines.common import write_json
from baselines.moe_layer.config import MODEL_SHAPES, summarize
from baselines.moe_layer.implementations.registry import BACKENDS

PHASES = {"forward": "Forward", "backward": "Backward", "total": "Forward + backward"}


def phase_timings(result, attempt):
    """Recover event timings from legacy raw samples without mixing in wall time."""
    if "phase_timings" in result:
        return result["phase_timings"]
    ranks = [
        json.loads(path.read_text())
        for path in sorted(attempt.glob("rank-*/result.json"))
    ]
    phases = {}
    for phase, key in (
        ("forward", "forward_gpu_seconds"),
        ("backward", "backward_gpu_seconds"),
        ("total", "gpu_seconds"),
    ):
        if ranks and all(
            all(key in sample for sample in rank["samples"]) for rank in ranks
        ):
            timing = summarize(
                [[sample[key] for sample in rank["samples"]] for rank in ranks]
            )
            phases[phase] = {
                **timing,
                "effective_tflops_per_gpu": result["request"]["flops_per_rank"][phase]
                / timing["median_seconds"]
                / 1e12,
            }
    return phases


def collect(root):
    rows = []
    for case in sorted((root / "cases").glob("*/*/*/tokens-*")):
        attempts = sorted(case.glob("attempt-*"))
        if not attempts:
            continue
        attempt = attempts[-1]
        outcome_path = attempt / "outcome.json"
        outcome = (
            json.loads(outcome_path.read_text())
            if outcome_path.exists()
            else {"status": "running_or_interrupted"}
        )
        backend, precision, training, tokens = case.relative_to(root / "cases").parts
        row = dict(
            backend=backend,
            precision=precision,
            training=training,
            tokens_per_rank=int(tokens.removeprefix("tokens-")),
            status=outcome["status"],
            path=str(attempt.relative_to(root)),
        )
        if row["status"] != "completed":
            failures = [
                json.loads(p.read_text()) for p in attempt.glob("rank-*/failure.json")
            ]
            row["failure_stages"] = ", ".join(
                sorted({f.get("stage", "unknown") for f in failures})
            )
        if row["status"] == "completed":
            result = json.loads((attempt / "result.json").read_text())
            if not result["performance_result"]:
                continue
            request = result["request"]
            row.update(
                world_size=request["world_size"],
                **request["shape"],
                wall_milliseconds=result["timing"]["median_seconds"] * 1000,
                wall_tflops_per_gpu=result["effective_tflops_per_gpu"],
                peak_allocated_gib=result["peak_allocated_bytes"] / 2**30,
            )
            for phase, timing in phase_timings(result, attempt).items():
                row.update(
                    {
                        f"{phase}_milliseconds": timing["median_seconds"] * 1000,
                        f"{phase}_tflops_per_gpu": timing["effective_tflops_per_gpu"],
                        f"{phase}_min_milliseconds": timing["min_seconds"] * 1000,
                        f"{phase}_max_milliseconds": timing["max_seconds"] * 1000,
                    }
                )
        rows.append(row)
    order = {backend: index for index, backend in enumerate(BACKENDS)}
    return sorted(
        rows,
        key=lambda row: (
            row["precision"],
            row["training"],
            row["tokens_per_rank"],
            order.get(row["backend"], len(order)),
        ),
    )


def collect_memory(root, rows):
    """Flat per-rank/per-phase memory table, retaining allocator/device distinctions."""
    memory_rows = []
    for row in rows:
        attempt = root / row["path"]
        if row["status"] == "completed":
            result = json.loads((attempt / "result.json").read_text())
            measurements = result.get("memory_by_rank", {})
        else:
            # A validation failure must not discard completed memory observations.
            measurements = {
                str(int(p.parent.name.removeprefix("rank-"))): json.loads(
                    p.read_text()
                )["memory"]
                for p in attempt.glob("rank-*/measurements.json")
            }
        for rank, phases in measurements.items():
            for phase, values in phases.items():
                item = {
                    key: row[key]
                    for key in ("backend", "precision", "training", "tokens_per_rank")
                }
                item.update(rank=int(rank), phase=phase, status=row["status"])
                for key, value in values.items():
                    if key in ("before", "after"):
                        item.update(
                            {
                                f"{key}_{name.removesuffix('_bytes')}_gib": count
                                / 2**30
                                for name, count in value.items()
                            }
                        )
                    elif key.endswith("_bytes"):
                        item[key.removesuffix("_bytes") + "_gib"] = (
                            None if value is None else value / 2**30
                        )
                    else:
                        item[key] = value
                memory_rows.append(item)
    return memory_rows


def validate_inputs(root, rows):
    """Reject a comparison if matching ranks received different seeded data."""
    seen = {}
    for row in rows:
        if row["status"] != "completed":
            continue
        for path in (root / row["path"]).glob("rank-*/request.json"):
            request = json.loads(path.read_text())
            if "input_fingerprint" not in request:
                continue  # Legacy results predate the explicit fingerprint record.
            key = (
                request["seed"],
                request["rank"],
                request["tokens"],
                request["shape"]["width"],
            )
            value = (
                request["input_seed"],
                request["gradient_seed"],
                request["input_fingerprint"],
                request["gradient_fingerprint"],
            )
            if key in seen and seen[key] != value:
                raise ValueError(f"seeded DP/EP input mismatch for {key}: {path}")
            seen[key] = value
    return len(seen)


def plot_options(root, rows):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    options = [
        item.get("options", {})
        for item in manifest.get("backend_manifests", {"sweep": manifest}).values()
    ]
    tokens = sorted(
        {row["tokens_per_rank"] for row in rows}
        | {token for item in options for token in item.get("tokens", [])}
    )
    tokens = tokens or [2048, 4096, 8192, 16384, 32768, 65536]
    selected = {row["backend"] for row in rows} | {
        backend for item in options for backend in item.get("backends", [])
    }
    backends = [backend for backend in BACKENDS if backend in selected]
    backends = backends or ["sonicmoe", "deepep", "hybridep", "ncclep"]
    dimensions = ("width", "hidden", "experts", "top_k")
    shapes = {tuple(row[key] for key in dimensions) for row in rows if "width" in row}
    if len(shapes) > 1:
        raise ValueError("a plot comparison requires one shared MoE layer shape")
    if shapes:
        shape = next(iter(shapes))
    else:
        entry = options[0] if options else {}
        preset = MODEL_SHAPES[entry.get("model", "glm53-flash")]
        shape = tuple(
            entry.get(key) or default
            for key, default in zip(dimensions, preset, strict=True)
        )
    ranks = {
        row["lora_rank"]
        for row in rows
        if row["training"] == "lora" and "lora_rank" in row
    }
    ranks |= {item["lora_rank"] for item in options if "lora_rank" in item}
    if len(ranks) > 1:
        raise ValueError("a plot comparison requires one shared LoRA rank")
    return tokens, backends, shape, next(iter(ranks), 32)


def draw_panel(axis, rows, *, precision, training, tokens, backends, phase, metric):
    labels = [
        f"{token // 1024}K" if token % 1024 == 0 else str(token) for token in tokens
    ]
    width = 0.82 / len(backends)
    available = False
    for index, backend in enumerate(backends):
        entry = BACKENDS[backend]
        points = {
            row["tokens_per_rank"]: row
            for row in rows
            if row["precision"] == precision
            and row["training"] == training
            and row["backend"] == backend
        }
        key = (
            f"{phase}_{'tflops_per_gpu' if metric == 'throughput' else 'milliseconds'}"
        )
        values = [points.get(token, {}).get(key, math.nan) for token in tokens]
        available |= any(math.isfinite(value) for value in values)
        if metric == "throughput":
            axis.plot(
                tokens,
                values,
                label=entry.label,
                color=entry.color,
                marker=entry.marker,
                linewidth=2.2,
                markersize=6,
            )
        else:
            positions = [
                group + (index - (len(backends) - 1) / 2) * width
                for group in range(len(tokens))
            ]
            axis.bar(
                positions,
                values,
                width=width * 0.92,
                label=entry.label,
                color=entry.color,
            )
            for position, token in zip(positions, tokens, strict=True):
                state = points.get(token, {}).get("status")
                if state and state != "completed":
                    label = (
                        "OOM" if state == "out_of_memory" else state.replace("_", " ")
                    )
                    axis.text(
                        position,
                        0.02,
                        label,
                        transform=axis.get_xaxis_transform(),
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=entry.color,
                    )
    if metric == "throughput":
        axis.set_xscale("log", base=2)
        axis.set_xlim(min(tokens) / 1.15, max(tokens) * 1.15)
        axis.set_xticks(tokens, labels)
    else:
        axis.set_xticks(range(len(tokens)), labels)
        axis.set_xlim(-0.55, len(tokens) - 0.45)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#e3e8ef", linewidth=0.8)
    axis.set_axisbelow(True)
    if not available:
        axis.text(
            0.5,
            0.5,
            "Measurements pending",
            transform=axis.transAxes,
            ha="center",
            color="#617087",
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args(argv)
    root = args.results.resolve()
    rows = collect(root)
    validated_input_groups = validate_inputs(root, rows)
    memory_rows = collect_memory(root, rows)
    tokens, backends, (width, hidden, experts, top_k), rank = plot_options(root, rows)
    shape_label = (
        f"{experts} experts · top-{top_k} · model width {width} · "
        f"expert hidden {hidden} · LoRA rank {rank}"
    )
    write_json(
        root / "results.json",
        {
            "rows": rows,
            "memory_by_rank_and_phase": memory_rows,
            "validated_input_groups": validated_input_groups,
            "plot_denominator": (
                "GPU events; median of per-iteration maximum across ranks"
            ),
        },
    )
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (root / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    if memory_rows:
        fields = list(dict.fromkeys(key for row in memory_rows for key in row))
        with (root / "memory.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(memory_rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#8190a5",
            "text.color": "#14263f",
            "axes.labelcolor": "#14263f",
            "savefig.facecolor": "white",
        }
    )
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    names = []
    panels = [
        ("bf16", "BF16", "full", "Full training"),
        ("bf16", "BF16", "lora", f"LoRA (rank {rank})"),
        ("fp8_current", "FP8", "full", "Full training"),
        ("fp8_current", "FP8", "lora", f"LoRA (rank {rank})"),
    ]
    for metric in ("throughput", "runtime"):
        for phase, phase_label in PHASES.items():
            figure, axes = plt.subplots(
                2, 2, figsize=(16, 11.5), sharex=True, sharey=True
            )
            figure.subplots_adjust(
                left=0.085, right=0.98, top=0.78, bottom=0.17, hspace=0.28, wspace=0.14
            )
            figure.suptitle(
                f"MoE layer · {phase_label} · {metric}", y=0.975, fontsize=24
            )
            figure.text(0.5, 0.925, shape_label, ha="center", fontsize=15)
            unit = "tflops_per_gpu" if metric == "throughput" else "milliseconds"
            value_key = f"{phase}_{unit}"
            maximum = max(
                (row[value_key] for row in rows if value_key in row), default=1
            )
            for index, (axis, (precision, label, training, mode)) in enumerate(
                zip(axes.flat, panels, strict=True)
            ):
                draw_panel(
                    axis,
                    rows,
                    precision=precision,
                    training=training,
                    tokens=tokens,
                    backends=backends,
                    phase=phase,
                    metric=metric,
                )
                axis.set_title(
                    f"({chr(97 + index)}) {label} · {mode}",
                    loc="left",
                    fontsize=17,
                    pad=12,
                )
                axis.set_ylim(0, max(1, maximum * 1.12))
                axis.tick_params(labelsize=13)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            figure.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.89),
                ncol=2,
                frameon=False,
                fontsize=15,
            )
            ylabel = (
                "Effective TFLOP/s per GPU"
                if metric == "throughput"
                else "Runtime (ms)"
            )
            figure.text(
                0.02, 0.48, ylabel, rotation="vertical", va="center", fontsize=18
            )
            figure.text(0.53, 0.11, "Tokens per rank", ha="center", fontsize=18)
            figure.text(
                0.085,
                0.065,
                "GPU-event time · median of the slowest rank per iteration · "
                "combined phase measured directly",
                fontsize=11,
                color="#617087",
            )
            figure.text(
                0.085,
                0.035,
                "EP includes dispatch/combine. DP measures replicated local compute. "
                "Optimizer and DP gradient sync excluded.",
                fontsize=11,
                color="#617087",
            )
            name = f"moe-layer-{phase}-{metric}"
            names.append(name)
            for extension in ("png", "pdf", "svg"):
                figure.savefig(figures / f"{name}.{extension}", dpi=args.dpi)
            plt.close(figure)
    lines = [
        "# MoE layer results",
        "",
        shape_label,
        "",
        "Six figures show the same samples: three throughput line plots and "
        "three grouped runtime bar plots. Each figure has BF16/FP8 rows and "
        "full-training/LoRA columns. Missing results are not treated as zero.",
        "",
    ]
    for name in names:
        lines += [
            f"[{name}](figures/{name}.png) · [PDF](figures/{name}.pdf) · "
            f"[SVG](figures/{name}.svg)",
            "",
        ]
    lines += [
        "Values: [CSV](results.csv) · [JSON](results.json) · "
        "[Per-rank memory](memory.csv). GPU event intervals "
        "include launch gaps and communication dependencies. Wall-clock totals "
        "remain in the table. Phase maxima can belong to different ranks; "
        "combined runtime is measured directly, not calculated by adding medians.",
        "",
        "| Backend | Precision / training | Tokens/rank | Forward ms | "
        "Backward ms | Combined ms | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        values = [
            f"{row[f'{phase}_milliseconds']:.3f}"
            if f"{phase}_milliseconds" in row
            else "—"
            for phase in PHASES
        ]
        lines.append(
            f"| {BACKENDS[row['backend']].label} | "
            f"{row['precision']} / {row['training']} | {row['tokens_per_rank']} | "
            f"{' | '.join(values)} | [{row['status']}]({row['path']}/) |"
        )
    (root / "README.md").write_text("\n".join(lines) + "\n")
    images = "\n".join(
        f'<h2>{html.escape(name)}</h2><a href="figures/{name}.pdf">'
        f'<img style="width:100%;max-width:1400px" src="figures/{name}.png"></a>'
        for name in names
    )
    (root / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>MoE layer benchmark</title>'
        '<body style="font-family:system-ui;margin:2rem">'
        f"<h1>MoE layer benchmark</h1><p>{html.escape(shape_label)}</p>"
        '<p><a href="results.csv">CSV</a> · <a href="results.json">JSON</a> · '
        '<a href="memory.csv">Per-rank memory</a> · '
        f'<a href="README.md">Case status and raw logs</a></p>{images}</body>'
    )
    print(f"Wrote six four-panel figures to {figures}")


if __name__ == "__main__":
    main()
