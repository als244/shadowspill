"""Publish per-GPU throughput plots, a four-panel summary, and a linked table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path

from baselines.common import write_json
from baselines.moe_layer.config import Shape


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
        row = {
            "backend": backend,
            "precision": precision,
            "training": training,
            "tokens_per_rank": int(tokens.removeprefix("tokens-")),
            "status": outcome["status"],
            "path": str(attempt.relative_to(root)),
        }
        if row["status"] == "completed":
            result = json.loads((attempt / "result.json").read_text())
            if not result["performance_result"]:
                continue
            row.update(
                world_size=result["request"]["world_size"],
                lora_rank=result["request"]["shape"]["lora_rank"],
                **{
                    key: result["request"]["shape"][key]
                    for key in ("width", "hidden", "experts", "top_k")
                },
                milliseconds=result["timing"]["median_seconds"] * 1000,
                tflops_per_gpu=result["effective_tflops_per_gpu"],
                gpu_event_tflops_per_gpu=result["gpu_event_tflops_per_gpu"],
                peak_allocated_gib=result["peak_allocated_bytes"] / 2**30,
                spread_min_seconds=result["timing"]["min_seconds"],
                spread_max_seconds=result["timing"]["max_seconds"],
            )
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["precision"],
            row["training"],
            row["tokens_per_rank"],
            {"sonicmoe": 0, "quackmoe": 1}[row["backend"]],
        ),
    )


def draw_curves(axis, rows, *, precision, training, tokens):
    available = False
    for backend, color, marker in (
        ("sonicmoe", "#3268db", "o"),
        ("quackmoe", "#008b83", "D"),
    ):
        points = sorted(
            [
                row
                for row in rows
                if row["precision"] == precision
                and row["training"] == training
                and row["backend"] == backend
                and row["status"] == "completed"
            ],
            key=lambda row: row["tokens_per_rank"],
        )
        if not points:
            continue
        available = True
        world = points[0]["world_size"]
        line = (
            "DP · SonicMoE (1 GPU)"
            if backend == "sonicmoe"
            else f"EP · QuackMoE ({world} GPUs)"
        )
        axis.plot(
            [point["tokens_per_rank"] for point in points],
            [point["tflops_per_gpu"] for point in points],
            label=line,
            color=color,
            marker=marker,
            linewidth=2.5,
            markersize=7,
        )
    axis.set_xscale("log", base=2)
    axis.set_xlim(min(tokens) / 1.12, max(tokens) * 1.12)
    axis.set_xticks(
        tokens,
        [f"{token // 1024}K" if token % 1024 == 0 else str(token) for token in tokens],
    )
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#e3e8ef", linewidth=0.8)
    axis.set_axisbelow(True)
    if not available:
        axis.text(
            0.5, 0.5, "Measurements pending", transform=axis.transAxes, ha="center"
        )
    return available


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args(argv)
    root = args.results.resolve()
    rows = collect(root)
    tokens = sorted({row["tokens_per_rank"] for row in rows})
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifests = manifest.get("backend_manifests", {"sweep": manifest})
        requested = {
            token
            for entry in manifests.values()
            for token in entry.get("options", {}).get("tokens", [])
        }
        tokens = sorted(set(tokens) | requested)
    tokens = tokens or [4096, 8192, 16384, 32768, 65536, 131072]
    ranks = sorted(
        {
            row["lora_rank"]
            for row in rows
            if row["training"] == "lora" and "lora_rank" in row
        }
    )
    rank_label = ", ".join(map(str, ranks)) if ranks else "16"
    shapes = {
        tuple(row[key] for key in ("width", "hidden", "experts", "top_k"))
        for row in rows
        if "width" in row
    }
    if len(shapes) > 1:
        raise ValueError("a plot comparison requires one shared MoE layer shape")
    shape = Shape(*next(iter(shapes))) if shapes else Shape()
    shape_label = (
        f"{shape.experts} experts · top-{shape.top_k} · "
        rf"$d_{{\mathrm{{model}}}}={shape.width}$ · "
        rf"$d_{{\mathrm{{FFN}}}}={shape.hidden}$"
    )
    write_json(root / "results.json", {"rows": rows})
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (root / "results.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 14,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#8190a5",
            "text.color": "#14263f",
            "axes.labelcolor": "#14263f",
            "xtick.color": "#45566e",
            "ytick.color": "#45566e",
            "savefig.facecolor": "white",
        }
    )
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    names = []
    for precision, label in (("bf16", "BF16"), ("fp8_current", "FP8")):
        for training, mode in (
            ("full", "Full training"),
            ("lora", f"LoRA · rank {rank_label}, BF16 adapters"),
        ):
            figure, axis = plt.subplots(figsize=(9.5, 6.2))
            figure.subplots_adjust(left=0.12, right=0.97, top=0.76, bottom=0.23)
            figure.suptitle(
                "MoE layer · Router and expert FFNs",
                x=0.12,
                ha="left",
                y=0.97,
                fontsize=21,
            )
            figure.text(0.12, 0.89, shape_label, fontsize=15, color="#45566e")
            figure.text(
                0.12, 0.83, f"{label}   |   {mode}", fontsize=16, color="#45566e"
            )
            available = draw_curves(
                axis, rows, precision=precision, training=training, tokens=tokens
            )
            axis.set_xlabel("Tokens per rank", labelpad=10)
            axis.set_ylabel("Effective TFLOP/s per GPU", labelpad=9)
            if available:
                axis.legend(frameon=False, loc="best", fontsize=13)
            figure.text(
                0.12,
                0.105,
                "Forward + backward · real top-k routing · "
                "median wall time, slowest EP rank",
                fontsize=11,
                color="#617087",
            )
            figure.text(
                0.12,
                0.065,
                "EP includes dispatch/combine. DP is single-GPU compute; "
                "optimizer and DP gradient sync excluded.",
                fontsize=10,
                color="#617087",
            )
            name = f"{precision}-{training}"
            names.append(name)
            for extension in ("png", "pdf", "svg"):
                figure.savefig(figures / f"{name}.{extension}", dpi=args.dpi)
            plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    figure.subplots_adjust(
        left=0.095, right=0.975, top=0.80, bottom=0.16, hspace=0.33, wspace=0.17
    )
    figure.suptitle("MoE layer · Router and expert FFNs", y=0.975, fontsize=27)
    figure.text(0.5, 0.925, shape_label, ha="center", fontsize=20, color="#45566e")
    max_tflops = max(
        (row["tflops_per_gpu"] for row in rows if "tflops_per_gpu" in row),
        default=1,
    )
    y_max = math.ceil(max_tflops * 1.08 / 100) * 100
    legend = {}
    for index, (axis, (precision, label, training, mode)) in enumerate(
        zip(
            axes.flat,
            (
                ("bf16", "BF16", "full", "Full training"),
                ("bf16", "BF16", "lora", f"LoRA (rank {rank_label})"),
                ("fp8_current", "FP8", "full", "Full training"),
                ("fp8_current", "FP8", "lora", f"LoRA (rank {rank_label})"),
            ),
            strict=True,
        )
    ):
        draw_curves(axis, rows, precision=precision, training=training, tokens=tokens)
        axis.set_title(
            f"({chr(97 + index)})  {label} · {mode}",
            loc="left",
            fontsize=21,
            pad=16,
        )
        axis.set_ylim(0, y_max)
        axis.tick_params(labelsize=17, pad=6)
        handles, labels = axis.get_legend_handles_labels()
        legend.update(zip(labels, handles, strict=True))
    if legend:
        figure.legend(
            legend.values(),
            legend.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.895),
            ncol=2,
            frameon=False,
            fontsize=18,
            columnspacing=3,
        )
    figure.text(
        0.025,
        0.48,
        "Effective TFLOP/s per GPU",
        rotation="vertical",
        va="center",
        fontsize=22,
    )
    figure.text(0.535, 0.095, "Tokens per rank", ha="center", fontsize=22)
    figure.text(
        0.095,
        0.052,
        "Forward + backward · real top-k routing · median wall time, slowest EP rank",
        fontsize=14,
        color="#617087",
    )
    figure.text(
        0.095,
        0.027,
        "EP includes dispatch/combine; DP excludes gradient sync. "
        "No optimizer. LoRA adapters: BF16.",
        fontsize=14,
        color="#617087",
    )
    for extension in ("png", "pdf", "svg"):
        figure.savefig(figures / f"moe-layer-four-panel.{extension}", dpi=args.dpi)
    plt.close(figure)
    lines = [
        "# MoE layer microbenchmark",
        "",
        "[Open plots and table](index.html) · [Raw table](results.csv) · "
        "[JSON](results.json) · [Source/configuration](manifest.json)",
        "",
        "[Four-panel summary: PNG](figures/moe-layer-four-panel.png) · "
        "[PDF](figures/moe-layer-four-panel.pdf) · "
        "[SVG](figures/moe-layer-four-panel.svg)",
        "",
        f"Router plus expert FFNs: model dimension {shape.width}, expert FFN "
        f"dimension {shape.hidden}, {shape.experts} experts, "
        f"top-{shape.top_k} routing. "
        "SonicMoE uses one GPU; QuackMoE's group size is shown in each plot. "
        "TFLOP/s is per GPU.",
        "",
        "Forward/backward includes routing and EP dispatch/combine, but excludes "
        "optimizer updates, DP gradient synchronization, and the replicated "
        "EP router's optimizer-step reduction. FP8 uses E4M3 current per-row "
        "scaling, BF16 activation transport, and BF16 LoRA adapters. "
        "Gradients accumulate into FP32 parameters; SonicMoE BF16 autograd casts "
        "can round intermediate weight gradients to BF16.",
        "",
        "Useful FLOPs count a multiply-add as two: "
        "full training = 18 T K D H + 6 T D E; "
        "LoRA = 12 T K D H + 4 T D E + 6 T K r (2D + 3H). "
        "Padding, activation/routing elementwise work and internal "
        "recomputation are excluded from the numerator.",
        "",
    ]
    for name in names:
        lines += [f"[![{name}](figures/{name}.png)](figures/{name}.pdf)", ""]
    lines += [
        "| Precision | Training | Backend | Tokens/rank | Status | "
        "F+B ms | TFLOP/s/GPU | Peak PyTorch GiB | Raw evidence |",
        "|---|---|---|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = [
            f"{row[key]:.2f}" if key in row else "—"
            for key in ("milliseconds", "tflops_per_gpu", "peak_allocated_gib")
        ]
        lines.append(
            f"| {row['precision']} | {row['training']} | {row['backend']} | "
            f"{row['tokens_per_rank']} | {row['status']} | {' | '.join(metrics)} | "
            f"[logs/data]({row['path']}/) |"
        )
    (root / "README.md").write_text("\n".join(lines) + "\n")
    cards = "".join(
        f'<a href="figures/{name}.pdf"><img src="figures/{name}.png" alt="{name}"></a>'
        for name in names
    )
    headers = [
        "Precision",
        "Mode",
        "Backend",
        "Tokens/rank",
        "Status",
        "F+B ms",
        "TFLOP/s/GPU",
        "Evidence",
    ]
    table = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(value))}</td>"
            for value in [
                r["precision"],
                r["training"],
                r["backend"],
                r["tokens_per_rank"],
                r["status"],
                f"{r['milliseconds']:.2f}" if "milliseconds" in r else "—",
                f"{r['tflops_per_gpu']:.2f}" if "tflops_per_gpu" in r else "—",
            ]
        )
        + f'<td><a href="{r["path"]}/console.log">log</a> · '
        f'<a href="{r["path"]}/result.json">data</a></td></tr>'
        for r in rows
    )
    (root / "index.html").write_text(
        (Path(__file__).parent / "index_header.html").read_text()
        + cards
        + "</div><table><tr>"
        + "".join(f"<th>{x}</th>" for x in headers)
        + "</tr>"
        + table
        + "</table>"
    )
    write_json(
        root / "render-manifest.json",
        {
            "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "table_sha256": hashlib.sha256(
                (root / "results.json").read_bytes()
            ).hexdigest(),
            "matplotlib": matplotlib.__version__,
            "dpi": args.dpi,
        },
    )


if __name__ == "__main__":
    main()
