"""What a faster interconnect would be worth: throughput against budget, per lane speed.

The quickstart answers what an execution budget buys on the machine it runs
on. This asks the same question of interconnects the machine does not have:
the same model, the same geometries and the same budgets, planned as though
the lanes carried bytes at another rate. Nothing executes, so a line is what
the simulator predicts for the plan the search chose, and the distance
between two lines is the transfer calibration and nothing else.

    python -m benchmarking.bandwidth_frontier mlops_qwen35 \
        --sequence-length 32768 --sequences-per-step 2 \
        --min-tokens-per-microbatch 16384 \
        --budget-gib 14,16,18,20 \
        --sim-transfer-bandwidths 2.75/2.75,25/25,100/100 \
        --build-store <a store with this model's builds> \
        --output-dir <where the raw data and figures go>

Each calibration is planned in a subprocess of its own, so one that fails
costs its own line and no other, and `--resume` skips the ones already
answered. The planning is a full geometry search per budget, exactly what
the quickstart's search phase does, so a line here and the quickstart's
simulated line are the same quantity.

**The raw data is the record.** Every calibration's complete search report is
written under `raw_data/`, with a tidy table beside it, and the figures are
drawn from those files alone -- so a figure can be redrawn, narrowed or
restyled without planning anything again:

    python -m benchmarking.bandwidth_frontier --replot <raw_data> --output-dir <new>

A GPU is needed to build the programs, which is once per calibration and hits
the build store after the first; the searching itself is CPU work.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from shadowspill.store import STORE_MODES

GIB = 1 << 30


def _bandwidth_pairs(value: str) -> tuple[tuple[float, float], ...]:
    """`fetch/evict,fetch/evict` in GB/s, in the order the lines are drawn."""

    pairs = []
    for item in value.split(","):
        if not item.strip():
            continue
        fetch, _, evict = item.partition("/")
        if not evict:
            raise argparse.ArgumentTypeError(
                f"{item}: a calibration is fetch/evict in GB/s, such as 25/25"
            )
        try:
            pairs.append((float(fetch), float(evict)))
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{item}: {error}") from error
    if not pairs:
        raise argparse.ArgumentTypeError("no calibrations given")
    return tuple(pairs)


#: Named resolution options, as the quickstart names them, so a sweep and a
#: tour asked for the same word search the same shares.
NAMED_RESOLUTION_OPTIONS: dict[str, tuple[str, ...]] = {
    "quarters": ("0", "1/4", "1/2", "3/4", "1"),
    "eighths": tuple(f"{numerator}/8" for numerator in range(9)),
    "halves": ("0", "1/2", "1"),
}


def named_resolution_options(value: str) -> tuple[str, ...]:
    """A named set, or a comma-separated list of exact fractions."""

    if value in NAMED_RESOLUTION_OPTIONS:
        return NAMED_RESOLUTION_OPTIONS[value]
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def label_for(fetch: float, evict: float) -> str:
    """How one calibration is named in a file name and on the legend."""

    def part(value: float) -> str:
        return f"{value:g}".replace(".", "p")

    return f"{part(fetch)}_{part(evict)}"


def legend_for(fetch: float, evict: float) -> str:
    if fetch == evict:
        return f"{fetch:g} GB/s"
    return f"{fetch:g} / {evict:g} GB/s"


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("identity", nargs="?", help="a quickstart model identity")
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--sequences-per-step", type=int, default=None)
    parser.add_argument("--min-tokens-per-microbatch", type=int, default=None)
    parser.add_argument("--max-tokens-per-microbatch", type=int, default=None)
    parser.add_argument(
        "--budget-gib",
        type=_floats,
        default=None,
        help="the execution budgets to plan, in gibibytes",
    )
    parser.add_argument(
        "--spill-gib",
        type=float,
        default=112.0,
        help="the spill pool this process opens, which holds the model while"
        " its programs are built and so must fit in host memory. It is not"
        " what the plans are given: that is --sim-spill-gib",
    )
    parser.add_argument(
        "--sim-spill-gib",
        type=_floats,
        default=None,
        help="the spill budgets to plan against, comma separated, one figure"
        " each. A plan is never executed here, so a budget larger than this"
        " machine's memory is a question it can still answer. Defaults to"
        " --spill-gib",
    )
    parser.add_argument(
        "--physical-capacity-gib",
        type=float,
        default=30.0,
        help="the device pool the programs are built in; profiling runs real"
        " kernels, so this is the device the build needs rather than a budget",
    )
    parser.add_argument(
        "--sim-transfer-bandwidths",
        type=_bandwidth_pairs,
        default=None,
        help="one line per calibration: FETCH/EVICT in GB/s, comma separated",
    )
    parser.add_argument(
        "--latency-us",
        type=_floats,
        default=(20.5, 14.5),
        help="fetch and evict per-transfer latency, held the same on every"
        " line so bandwidth is the only thing that moves",
    )
    parser.add_argument(
        "--orderings",
        choices=("factors", "depth-first"),
        default="factors",
        help="which microbatch orderings the search tries per geometry:"
        " every depth x breadth factor pair (the default), or only the"
        " depth-first walk",
    )
    parser.add_argument(
        "--initial-placement",
        choices=("greedy", "required"),
        default=None,
        help="how objects the declaration leaves in spill may be placed"
        " before the first task; defaults to whichever the library chooses",
    )
    parser.add_argument(
        "--resolution-options",
        type=named_resolution_options,
        default=None,
        help="which resolutions every point is searched over: 'quarters'"
        " (the library default), 'eighths', 'halves', or a comma-separated"
        " list of exact fractions such as 0,1/2,7/8,1",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="make the search reproduce exactly at any worker count. On by"
        " default here, because the lines are compared against each other",
    )
    parser.add_argument(
        "--incumbents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="hand each budget the best plan found at a smaller budget as the"
        " plan to beat, so no line falls with more memory",
    )
    parser.add_argument("--export-bypass-key", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="everything this sweep produces goes under it: raw_data/ with a"
        " search report per calibration, logs/ with a planning log per"
        " calibration, figures/, and the stores unless they are placed"
        " elsewhere",
    )
    parser.add_argument(
        "--artifact-store",
        type=Path,
        default=None,
        help="roots both stores under one directory. Defaults to"
        " <output-dir>/artifact_store",
    )
    parser.add_argument(
        "--build-store",
        type=Path,
        default=None,
        help="the captures, graph pairs, profiles and compiled artifacts to"
        " read and write; point it at another run's store so no calibration"
        " profiles a kernel again. Overrides --artifact-store for the build"
        " tree",
    )
    parser.add_argument(
        "--plan-store",
        type=Path,
        default=None,
        help="where this sweep's plans go, kept apart from the build store."
        " Every calibration shares it: a plan's key carries the calibration"
        " it was planned against, so they cannot be confused. Overrides"
        " --artifact-store for the planning tree. Defaults to"
        " <output-dir>/plan_store",
    )
    for tree in ("build", "plan"):
        parser.add_argument(
            f"--{tree}-store-mode",
            choices=STORE_MODES,
            default="contribute",
            help=f"what this sweep may do about a {tree} artifact the store"
            " does not hold: contribute builds it and writes it back, reuse"
            " builds it and persists nothing, require refuses and names it",
        )
    parser.add_argument(
        "--measured",
        default=None,
        help="what this machine, or another, actually measured -- drawn over"
        " the simulated lines as a second figure. Either a quickstart run"
        " directory, which is read as one measured set on the first"
        " calibration, or a JSON file naming one set per calibration:"
        ' {"measured": [{"fetch_gb_per_second": 2.75, "evict_gb_per_second":'
        ' 2.75, "run": "<a quickstart run>"}, {"fetch_gb_per_second": 25,'
        ' "evict_gb_per_second": 25, "label": "A100 over NVLink",'
        ' "tokens_per_second": {"16": 2100, "24": 2600}}]}. Each set either'
        " names a run whose measured throughput is read, or gives budgets in"
        " gibibytes against tokens per second directly",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--replot",
        default=None,
        help="draw from an existing raw_data directory and plan nothing",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--plan-one",
        default=None,
        help=argparse.SUPPRESS,  # the per-calibration subprocess
    )
    parser.add_argument(
        "--plan-spill",
        type=float,
        default=None,
        help=argparse.SUPPRESS,  # the spill budget that subprocess plans
    )
    parsed = parser.parse_args(argv)
    if parsed.sim_spill_gib is None and parsed.spill_gib is not None:
        parsed.sim_spill_gib = (parsed.spill_gib,)
    if parsed.replot is None:
        missing = [
            name
            for name, value in (
                ("identity", parsed.identity),
                ("--sequence-length", parsed.sequence_length),
                ("--sequences-per-step", parsed.sequences_per_step),
                ("--budget-gib", parsed.budget_gib),
                ("--sim-transfer-bandwidths", parsed.sim_transfer_bandwidths),
            )
            if value is None
        ]
        if missing:
            parser.error(f"planning needs {', '.join(missing)}")
    return parsed


def search_policy(parsed: argparse.Namespace) -> Any:
    """The one policy every calibration is searched under.

    Built from the same flags the quickstart builds its own from, and with
    the same defaults, so a line here and a quickstart line are answers to
    the same question. `None` where a flag was not given leaves the
    library's own choice rather than naming one that could fall behind it.
    """

    from fractions import Fraction

    from shadowspill.planner import (
        GenericPlanningOptions,
        InitialPlacement,
        SearchOptions,
    )
    from shadowspill.planner.search.algorithms.pressurefit import PressureFit
    from shadowspill.planner.search.algorithms.pressurefit.options import (
        PressureFitOptions,
    )

    options: dict[str, Any] = {}
    if parsed.initial_placement is not None:
        options["initial_placement"] = InitialPlacement(parsed.initial_placement)
    if parsed.resolution_options is not None:
        options["resolution_options"] = tuple(
            Fraction(share) for share in parsed.resolution_options
        )
    return SearchOptions(
        generic=GenericPlanningOptions(deterministic=parsed.deterministic),
        algorithm=PressureFit(PressureFitOptions(**options)),
    )


def store_paths(parsed: argparse.Namespace) -> tuple[Path, Path]:
    """Where the builds and the plans live, as the arguments place them.

    One rule, the quickstart's: `--artifact-store` roots both, either of the
    two overrides its own tree, and the default is under the output
    directory, so a sweep given nothing keeps everything it made together.
    """

    root = parsed.artifact_store or (Path(parsed.output_dir) / "artifact_store")
    build = parsed.build_store or root / "build"
    plan = parsed.plan_store or (
        root / "plan"
        if parsed.artifact_store
        else Path(parsed.output_dir) / "plan_store"
    )
    return Path(build), Path(plan)


def plan_one_calibration(parsed: argparse.Namespace, raw_data: Path) -> Path:
    """Search every budget under one calibration and save the whole report."""

    # Imported here so `--replot` needs neither a GPU nor a framework.
    import torch

    from shadowspill.memory import device, pinned_host, transfer_route
    from shadowspill.planner import StepDataOrdering
    from shadowspill.planner.program_inputs import TransferBandwidths
    from shadowspill.pytorch import Runtime, plan_step_search
    from tools.qualification.model_state import release_case_model
    from workloads.common.training import optimizer_state_init
    from workloads.full_model import build_case, manifest_for

    fetch, evict = _bandwidth_pairs(parsed.plan_one)[0]
    spill_gib = parsed.plan_spill if parsed.plan_spill is not None else parsed.spill_gib
    label = f"{label_for(fetch, evict)}_spill{spill_gib:g}"
    # `mlops_qwen35` names the implementation first and the family second.
    implementation, _, family = parsed.identity.partition("_")
    manifest = manifest_for(family, implementation)
    manifest = replace(
        manifest,
        sequence_length=parsed.sequence_length,
        spill_budget_bytes=int(parsed.spill_gib * GIB),
        device_physical_capacity_bytes=int(parsed.physical_capacity_gib * GIB),
    )
    runtime = Runtime(
        pools={
            "execution": device(
                physical_capacity=int(parsed.physical_capacity_gib * GIB)
            ),
            "spill": pinned_host(capacity=int(parsed.spill_gib * GIB)),
        },
        routes={
            "fetch": transfer_route(source="spill", destination="execution"),
            "evict": transfer_route(source="execution", destination="spill"),
        },
    )
    case = build_case(manifest, seed=parsed.seed, runtime=runtime)
    vocabulary = int(manifest.model_config.vocab_size)

    def example_microbatches(sequences: int, accumulation: int) -> Any:
        shape = (1, sequences * parsed.sequence_length)
        lengths = (parsed.sequence_length,) * sequences
        return tuple(
            (
                torch.randint(vocabulary, shape),
                torch.randint(vocabulary, shape),
                lengths,
            )
            for _ in range(accumulation)
        )

    latencies = parsed.latency_us
    bandwidths = TransferBandwidths(
        int(fetch * 1_000_000_000),
        int(evict * 1_000_000_000),
        provenance=f"bandwidth_frontier {fetch:g}/{evict:g} GB/s",
        fetch_latency_ns=int(latencies[0] * 1000) if latencies else None,
        evict_latency_ns=int(latencies[1] * 1000) if len(latencies) > 1 else None,
    )
    build_store, plan_store = store_paths(parsed)
    started = time.perf_counter()
    # `build_case` already materialised the model into the runtime's spill
    # pool, so the search is handed the model the runtime owns.
    with case.implementations():
        report = plan_step_search(
            case.model,
            objective=case.objective,
            optimizer=case.optimizer,
            optimizer_state_init=optimizer_state_init,
            hyperparams=("lr",),
            example_microbatches=example_microbatches,
            total_sequences_per_step=parsed.sequences_per_step,
            sequence_length=parsed.sequence_length,
            budgets=[
                (int(budget * GIB), int(spill_gib * GIB))
                for budget in parsed.budget_gib
            ],
            runtime=runtime,
            execution="execution",
            spill="spill",
            transfer_bandwidths=bandwidths,
            min_tokens_per_microbatch=parsed.min_tokens_per_microbatch,
            max_tokens_per_microbatch=parsed.max_tokens_per_microbatch,
            artifact_store=str(build_store),
            build_store=str(build_store),
            plan_store=str(plan_store),
            build_store_mode=parsed.build_store_mode,
            plan_store_mode=parsed.plan_store_mode,
            verbose=True,
            incumbents=parsed.incumbents,
            orderings=(
                None
                if parsed.orderings == "factors"
                else lambda accumulation: (StepDataOrdering.depth_first(accumulation),)
            ),
            search_options=search_policy(parsed),
            export_bypass_key=parsed.export_bypass_key,
        )
    path = report.save(raw_data / f"search_{label}.json")
    print(
        f"planned {legend_for(fetch, evict)} at {spill_gib:g} GiB spill over"
        f" {len(parsed.budget_gib)} budgets in"
        f" {time.perf_counter() - started:.0f} s -> {path}",
        flush=True,
    )
    release_case_model(case, runtime=runtime)
    runtime.close()
    return path


def sweep(parsed: argparse.Namespace, output: Path) -> None:
    """Plan every calibration, each in a subprocess of its own."""

    raw_data = output / "raw_data"
    logs = output / "logs"
    for directory in (raw_data, logs):
        directory.mkdir(parents=True, exist_ok=True)
    for spill_gib in parsed.sim_spill_gib:
        for fetch, evict in parsed.sim_transfer_bandwidths:
            label = f"{label_for(fetch, evict)}_spill{spill_gib:g}"
            named = f"{legend_for(fetch, evict)} at {spill_gib:g} GiB spill"
            report_path = raw_data / f"search_{label}.json"
            if parsed.resume and report_path.exists():
                print(f"{named}: already planned, kept", flush=True)
                continue
            command = [
                sys.executable,
                "-u",
                "-m",
                "benchmarking.bandwidth_frontier",
                *[item for item in sys.argv[1:] if item not in ("--resume",)],
                "--plan-one",
                f"{fetch:g}/{evict:g}",
                "--plan-spill",
                f"{spill_gib:g}",
            ]
            log_path = logs / f"{label}.log"
            print(f"=== {named} -> {log_path}", flush=True)
            with log_path.open("w") as handle:
                code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)
            if code != 0 or not report_path.exists():
                print(f"    FAILED (exit {code}); see {log_path}", flush=True)


def _rows(raw_data: Path) -> list[dict[str, Any]]:
    """Every searched point of every calibration, as one tidy table."""

    from shadowspill.plots.step_search.series import winning_points
    from shadowspill.search import StepSearchReport

    rows: list[dict[str, Any]] = []
    for path in sorted(raw_data.glob("search_*.json")):
        report = StepSearchReport.load(path)
        calibration = report.transfer_bandwidths
        winners = {
            point.execution_budget_bytes: point for point in winning_points(report)
        }
        for point in report.points:
            winner = winners.get(point.execution_budget_bytes)
            rows.append(
                {
                    "label": path.stem.removeprefix("search_"),
                    "fetch_bytes_per_second": (
                        None
                        if calibration is None
                        else calibration.fetch_bytes_per_second
                    ),
                    "evict_bytes_per_second": (
                        None
                        if calibration is None
                        else calibration.evict_bytes_per_second
                    ),
                    "fetch_latency_ns": (
                        None if calibration is None else calibration.fetch_latency_ns
                    ),
                    "evict_latency_ns": (
                        None if calibration is None else calibration.evict_latency_ns
                    ),
                    "execution_budget_gib": point.execution_budget_bytes / GIB,
                    "spill_budget_gib": point.spill_budget_bytes / GIB,
                    "sequences_per_microbatch": point.sequences_per_microbatch,
                    "accumulation_count": point.accumulation_count,
                    "ordering": point.ordering.label,
                    "status": point.status,
                    "makespan_seconds": point.makespan_seconds,
                    "tokens_per_second": (
                        None
                        if point.makespan_seconds is None
                        else report.tokens_per_step / point.makespan_seconds
                    ),
                    "is_winner": bool(
                        winner is not None
                        and winner.sequences_per_microbatch
                        == point.sequences_per_microbatch
                        and winner.accumulation_count == point.accumulation_count
                        and winner.ordering.label == point.ordering.label
                    ),
                    "fetched_gib": (
                        None
                        if point.summary is None
                        else point.summary.transfer_bytes_fetched / GIB
                    ),
                    "evicted_gib": (
                        None
                        if point.summary is None
                        else point.summary.transfer_bytes_evicted / GIB
                    ),
                    "idle_seconds": (
                        None if point.summary is None else point.summary.idle_seconds
                    ),
                    "spill_peak_gib": (
                        None
                        if point.summary is None
                        else point.summary.spill_peak_bytes / GIB
                    ),
                    "unconstrained_step_seconds": (
                        None
                        if point.summary is None
                        else point.summary.unconstrained_step_seconds
                    ),
                    "unconstrained_tokens_per_second": (
                        None
                        if point.summary is None
                        or not point.summary.unconstrained_step_seconds
                        else report.tokens_per_step
                        / point.summary.unconstrained_step_seconds
                    ),
                    "tokens_per_step": report.tokens_per_step,
                    "error": point.error,
                }
            )
    return rows


def write_raw_data(raw_data: Path, parsed: argparse.Namespace) -> Path:
    """The tidy table beside the reports, and what the sweep was asked for."""

    rows = _rows(raw_data)
    table = raw_data / "frontier.csv"
    if rows:
        with table.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (raw_data / "sweep.json").write_text(
        json.dumps(
            {
                "identity": parsed.identity,
                "sequence_length": parsed.sequence_length,
                "sequences_per_step": parsed.sequences_per_step,
                "budgets_gib": list(parsed.budget_gib or ()),
                "spill_gib": parsed.spill_gib,
                "sim_spill_gib": list(parsed.sim_spill_gib or ()),
                "latency_us": list(parsed.latency_us),
                "calibrations": [
                    {"fetch_gb_per_second": fetch, "evict_gb_per_second": evict}
                    for fetch, evict in (parsed.sim_transfer_bandwidths or ())
                ],
                "min_tokens_per_microbatch": parsed.min_tokens_per_microbatch,
                "max_tokens_per_microbatch": parsed.max_tokens_per_microbatch,
                "orderings": parsed.orderings,
                "initial_placement": parsed.initial_placement,
                "resolution_options": list(parsed.resolution_options or ()),
                "deterministic": parsed.deterministic,
                "incumbents": parsed.incumbents,
            },
            indent=1,
        )
        + "\n"
    )
    return table


def _run_throughput(run_directory: Path) -> dict[float, float]:
    """The measured throughput of a quickstart run, by execution budget."""

    source = run_directory / "figures" / "raw_data" / "run_budgets.csv"
    if not source.is_file():
        return {}
    points: dict[float, float] = {}
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            measured = row.get("measured_tokens_per_second")
            budget = row.get("execution_budget_gib")
            if measured and budget:
                points[float(budget)] = float(measured)
    return points


def collect_measured(parsed: argparse.Namespace, raw_data: Path) -> Path | None:
    """Resolve every measured set into the raw data, so a redraw needs no run.

    A set may name a quickstart run or give its numbers directly. Either way
    what lands here is the numbers, with the calibration they belong to and
    where they came from, because a figure has to be redrawable from this
    directory alone and a run elsewhere may be gone by then.
    """

    if not parsed.measured:
        return None
    source = Path(parsed.measured)
    entries: list[dict[str, Any]] = []
    if source.is_dir():
        first = (parsed.sim_transfer_bandwidths or ((None, None),))[0]
        entries.append(
            {
                "fetch_gb_per_second": first[0],
                "evict_gb_per_second": first[1],
                "run": str(source),
            }
        )
    else:
        record = json.loads(source.read_text())
        entries = list(
            record.get("measured", record if isinstance(record, list) else [])
        )
    resolved = []
    for entry in entries:
        points = {
            float(budget): float(value)
            for budget, value in (entry.get("tokens_per_second") or {}).items()
        }
        run = entry.get("run")
        if run:
            points.update(_run_throughput(Path(run)))
        if not points:
            print(f"  measured set with no points, skipped: {entry}")
            continue
        fetch = entry.get("fetch_gb_per_second")
        evict = entry.get("evict_gb_per_second")
        resolved.append(
            {
                "label": entry.get("label")
                or (
                    "measured"
                    if fetch is None
                    else f"measured at {legend_for(float(fetch), float(evict))}"
                ),
                "line": None
                if fetch is None
                else legend_for(float(fetch), float(evict)),
                "fetch_gb_per_second": fetch,
                "evict_gb_per_second": evict,
                "spill_gib": entry.get("spill_gib"),
                "run": run,
                "tokens_per_second": {
                    str(budget): value for budget, value in points.items()
                },
            }
        )
    path = raw_data / "measured.json"
    path.write_text(json.dumps({"measured": resolved}, indent=1) + "\n")
    print(f"  measured sets: {len(resolved)} -> {path}")
    return path


def draw(raw_data: Path, output: Path, title: str | None) -> tuple[Path, ...]:
    """Draw the figures from the raw data alone, one pair per spill budget."""

    from shadowspill.plots import FrontierLine, MeasuredPoints, plot_bandwidth_frontier

    rows = _rows(raw_data)
    if not rows:
        print(f"no search reports under {raw_data}")
        return ()
    sweep_record = json.loads((raw_data / "sweep.json").read_text())
    measured_file = raw_data / "measured.json"
    measured_sets = (
        json.loads(measured_file.read_text()).get("measured", [])
        if measured_file.is_file()
        else []
    )
    ceilings = [
        row["unconstrained_tokens_per_second"]
        for row in rows
        if row["unconstrained_tokens_per_second"]
    ]
    ceiling = max(ceilings) if ceilings else None
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    tokens = sweep_record["sequences_per_step"] * sweep_record["sequence_length"]
    heading = title or (
        f"{sweep_record['identity']}: {sweep_record['sequence_length']:,}-token"
        f" sequences, {tokens:,} tokens per step"
    )
    spills = sorted({row["spill_budget_gib"] for row in rows})
    written: tuple[Path, ...] = ()
    for spill in spills:
        # A spill budget is a different machine, so its figures are their own
        # files; one budget keeps the plain names a reader expects.
        suffix = f"_spill{spill:g}gib" if len(spills) > 1 else ""
        spill_rows = [row for row in rows if row["spill_budget_gib"] == spill]
        budgets = sorted({row["execution_budget_gib"] for row in spill_rows})
        lines = []
        for calibration in sweep_record["calibrations"]:
            fetch = calibration["fetch_gb_per_second"]
            evict = calibration["evict_gb_per_second"]
            label = label_for(fetch, evict)
            best: dict[float, float | None] = {budget: None for budget in budgets}
            for row in spill_rows:
                if (
                    not row["label"].startswith(f"{label}_spill")
                    or not row["is_winner"]
                ):
                    continue
                best[row["execution_budget_gib"]] = row["tokens_per_second"]
            if all(value is None for value in best.values()):
                continue
            lines.append(
                FrontierLine(
                    label=legend_for(fetch, evict),
                    fetch_bytes_per_second=int(fetch * 1_000_000_000),
                    evict_bytes_per_second=int(evict * 1_000_000_000),
                    tokens_per_second=best,
                )
            )
        if not lines:
            continue
        subtitle = (
            f"simulated throughput by transfer bandwidth, {spill:g} GiB spill budget"
        )
        written += (
            plot_bandwidth_frontier(
                figures / f"throughput_by_bandwidth{suffix}.png",
                budgets,
                lines,
                title=heading,
                subtitle=subtitle,
                unconstrained_tokens_per_second=ceiling,
            ),
        )
        # A measured set says which spill budget it ran at when it knows; one
        # that does not is drawn on every figure, as context rather than a
        # claim about that budget.
        series = [
            MeasuredPoints(
                label=entry["label"],
                line=entry.get("line"),
                tokens_per_second={
                    float(budget): float(value)
                    for budget, value in entry["tokens_per_second"].items()
                },
            )
            for entry in measured_sets
            if entry.get("spill_gib") in (None, spill)
        ]
        if series:
            written += (
                plot_bandwidth_frontier(
                    figures / f"throughput_by_bandwidth_measured{suffix}.png",
                    budgets,
                    lines,
                    title=heading,
                    subtitle=(
                        "measured against simulated by transfer bandwidth,"
                        f" {spill:g} GiB spill budget"
                    ),
                    measured=series,
                    unconstrained_tokens_per_second=ceiling,
                    faded_lines=True,
                ),
            )
    return written


def main(argv: list[str] | None = None) -> int:
    parsed = arguments(argv)
    output = Path(parsed.output_dir)
    if parsed.plan_one is not None:
        raw_data = output / "raw_data"
        raw_data.mkdir(parents=True, exist_ok=True)
        plan_one_calibration(parsed, raw_data)
        return 0
    if parsed.replot is not None:
        source = Path(parsed.replot)
        raw_data = source if source.name == "raw_data" else source / "raw_data"
        for path in draw(raw_data, output, parsed.title):
            print(f"  {path}")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    sweep(parsed, output)
    raw_data = output / "raw_data"
    table = write_raw_data(raw_data, parsed)
    print(f"  raw data: {table}")
    collect_measured(parsed, raw_data)
    for path in draw(raw_data, output, parsed.title):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
