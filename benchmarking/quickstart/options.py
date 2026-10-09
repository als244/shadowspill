"""CLI flags and search/profiling policy; no model imports."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import fields
from fractions import Fraction
from pathlib import Path

from shadowspill.planner import GenericPlanningOptions, SearchOptions
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.planner.search.algorithms.pressurefit import PressureFit
from shadowspill.planner.search.algorithms.pressurefit.options import PressureFitOptions
from shadowspill.planner.search.toolkit.resolution import (
    NAMED_RESOLUTION_OPTIONS,
    named_resolution_options,
    validate_resolution_options,
)
from shadowspill.pytorch import ProfilingOptions, StepSearchReport
from shadowspill.schema import artifact_schema
from shadowspill.store import STORE_MODES

IDENTITIES = (
    "mlops_llama3",
    "mlops_qwen35",
    "mlops_qwen3moe",
    "mlops_qwen35moe",
    "mlops_olmoe",
    "pytorch_llama3",
    "pytorch_qwen35",
)
_DTYPE_NAMES = ("bfloat16", "float16", "float32")
_ROUNDINGS = ("nearest", "stochastic")
_REQUEST_SCHEMA = artifact_schema("quickstart_request")
_NOT_REQUEST = frozenset(
    {
        "reproduce",
        "output_dir",
        "force_overwrite",
        "plots",
        "timelines",
        "resolution_plans",
    }
)
_REPRODUCE_MAY_TAKE = frozenset(
    {
        "--reproduce",
        "--output-dir",
        "--plots",
        "--timelines",
        "--no-timelines",
        "--resolution-plans",
        "--no-resolution-plans",
    }
)


def _budget_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def _transfer_bandwidths(value: str) -> TransferBandwidths:
    """Four solo/concurrent rates, optional latencies, or a saved search.json."""

    path = Path(value)
    if path.suffix == ".json":
        try:
            recorded = StepSearchReport.load(path).planned_lanes
        except (OSError, ValueError) as error:
            raise argparse.ArgumentTypeError(f"{value}: {error}") from error
        if recorded is None:
            raise argparse.ArgumentTypeError(f"{value} records no transfer calibration")
        return recorded
    parts = [item.strip() for item in value.split(",")]
    if len(parts) not in (2, 4, 6):
        raise argparse.ArgumentTypeError(
            "expected four rates (fetch solo/concurrent, evict solo/concurrent)"
            " in GB/s,"
            " optionally two latencies in microseconds; two rates set fixed"
            " fetch/evict; or use search.json"
        )
    try:
        rates = tuple(int(float(item) * 1e9) for item in parts[:4])
        if len(parts) == 2:
            rates = (rates[0], rates[0], rates[1], rates[1])
        latencies = tuple(int(float(item) * 1e3) for item in parts[4:])
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{value}: {error}") from error
    return TransferBandwidths(
        *rates,
        provenance=f"quickstart --transfer-bandwidths {value}",
        fetch_latency_ns=latencies[0] if latencies else None,
        evict_latency_ns=latencies[1] if latencies else None,
    )


def _revision() -> str:
    """The revision a run measured, so its outputs name the code they describe.

    A modified tree is marked, because a run from one is not reproducible from
    the hash alone. Falls back to `nogit` outside a checkout rather than
    failing: the outputs are still worth keeping, they just cannot be traced
    to a commit.
    """

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        modified = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"
    return f"{revision}_dirty" if modified else revision


def _started_at() -> str:
    """When a run began, as `MMDD_HHMM`, so reruns of one revision stay apart.

    A revision does not identify a run on its own: the same commit gets
    measured more than once -- on a quiet machine, after a rebuild, against
    another run's store -- and each of those is a measurement worth keeping
    beside the others rather than on top of them.
    """

    return time.strftime("%m%d_%H%M")


def search_policy(arguments: argparse.Namespace) -> SearchOptions:
    """The one search policy this tour plans with.

    Both planning phases have to be told the same thing: the geometry search ranks
    candidates under a policy, and the run then plans the plan that search promised.
    Building the policy twice is how the two drift -- a flag added to one call and not
    the other changes what is measured without changing what is reported -- so it is
    built here and threaded, and every phase is handed this value.
    """

    return SearchOptions(
        workers=getattr(arguments, "search_workers", 0),
        generic=GenericPlanningOptions(deterministic=arguments.deterministic),
        algorithm=PressureFit(
            PressureFitOptions(
                resolution_options=tuple(
                    Fraction(share) for share in arguments.resolution_options
                ),
            )
        ),
    )


def profiling_policy(arguments: argparse.Namespace) -> ProfilingOptions:
    """One effective policy shared by search and execution planning."""

    return ProfilingOptions(
        **{
            option.name: getattr(arguments, f"profile_{option.name}")
            for option in fields(ProfilingOptions)
        }
    )


def _dtype_name(value: str) -> str | None:
    """A floating dtype by its torch name, or ``none`` for the default."""

    lowered = value.strip().lower().removeprefix("torch.")
    if lowered == "none":
        return None
    if lowered not in _DTYPE_NAMES:
        raise argparse.ArgumentTypeError(
            f"a dtype is one of {', '.join(_DTYPE_NAMES)}, or none; not {value!r}"
        )
    return lowered


def _request_record(
    arguments: argparse.Namespace,
    store: Path,
    build_store: Path | None,
    plan_store: Path,
) -> dict[str, object]:
    """The request as given, with the stores resolved, for `--reproduce`."""

    request: dict[str, object] = {}
    for name, value in vars(arguments).items():
        if name in _NOT_REQUEST:
            continue
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, TransferBandwidths):
            value = value.to_dict()
        elif isinstance(value, tuple):
            value = list(value)
        request[name] = value
    request["artifact_store"] = str(store)
    request["build_store"] = None if build_store is None else str(build_store)
    request["plan_store"] = str(plan_store)
    return {
        "schema": _REQUEST_SCHEMA,
        "command": sys.argv[1:],
        "revision": _revision(),
        "started_at": _started_at(),
        "request": request,
    }


def _reproduced_arguments(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> argparse.Namespace:
    """The request a run recorded, pinned to its calibration, refusing a miss."""

    given = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
    foreign = sorted(given - _REPRODUCE_MAY_TAKE)
    if foreign or arguments.model is not None:
        named = [*foreign, *([arguments.model] if arguments.model else [])]
        parser.error(
            "--reproduce takes the whole request from the run; "
            f"{', '.join(named)} would contradict it"
        )
    run = arguments.reproduce
    if not (run / "request.json").exists():
        # torchrun supplies RANK before the CPU group is initialized.
        rank_root = run / f"rank-{int(os.environ.get('RANK', '0')):05d}"
        if rank_root.is_dir():
            run = rank_root
    record_path = run / "request.json"
    report_path = run / "search.json"
    for path in (record_path, report_path):
        if not path.is_file():
            parser.error(f"--reproduce: {path} is missing")
    try:
        record = json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"--reproduce: {record_path}: {error}")
    if not isinstance(record, dict) or record.get("schema") != _REQUEST_SCHEMA:
        parser.error(f"--reproduce: {record_path} is not a quickstart request")
    request = record.get("request")
    if not isinstance(request, dict):
        parser.error(f"--reproduce: {record_path} records no request")
    for name, value in request.items():
        if name in ("artifact_store", "build_store", "plan_store"):
            value = None if value is None else Path(value)
            if value is not None and "rank" in record:
                suffix = f"rank-{record['rank']:05d}"
                if value.name != suffix:
                    parser.error(f"distributed store {value} lacks its rank suffix")
                value = value.parent
        elif name == "resolution_options":
            value = tuple(value)
        setattr(arguments, name, value)
    arguments.transfer_bandwidths = _transfer_bandwidths(str(report_path))
    arguments.plan_store_mode = "require"
    return arguments


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "model",
        nargs="?",
        choices=IDENTITIES,
        help="which workload to run; taken from the run when --reproduce is given",
    )
    parser.add_argument(
        "--factory",
        help=(
            "module:function returning generic experiment arguments; "
            "receives device= and --factory-args"
        ),
    )
    parser.add_argument(
        "--factory-args",
        type=json.loads,
        default={},
        help="JSON keyword arguments for the experiment factory",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="local device, or auto (LOCAL_RANK under a launcher)",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help=(
            "torchrun: initialize Gloo before host admission; "
            "presets use data parallelism"
        ),
    )
    parser.add_argument("--host-headroom-gib", type=float, default=2)
    parser.add_argument(
        "--symmetric-planning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="verify and share distributed CPU search work; fall back on mismatches",
    )
    parser.add_argument("--preparation-timeout", type=float, default=1800)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--sequences-per-step", type=int)
    parser.add_argument(
        "--sequences-per-microbatch",
        type=int,
        help="choose the geometry yourself and skip the search; must divide"
        " the sequences per step",
    )
    parser.add_argument("--min-tokens-per-microbatch", type=int)
    parser.add_argument("--max-tokens-per-microbatch", type=int)
    parser.add_argument(
        "--search-budget-gib",
        type=_budget_list,
        help="comma-separated execution budgets to search and plot across,"
        " for example 10,12,16",
    )
    parser.add_argument(
        "--run-budget-gib",
        type=_budget_list,
        help="comma-separated execution budgets to actually run steps at."
        " Every run budget must appear among the search budgets. With"
        " neither flag the retained budget is searched and run; with only"
        " search budgets, nothing executes",
    )
    parser.add_argument("--spill-gib", type=float)
    parser.add_argument(
        "--external-headroom-gib",
        type=float,
        default=0.5,
        help="device memory reserved outside the execution slab (default: 0.5 GiB)",
    )
    parser.add_argument(
        "--remote-spill",
        metavar="HOST:PORT",
        help=(
            "spill to a memory daemon on another machine instead of to pinned "
            "host memory. The pool is the same size either way -- --spill-gib "
            "still sets it -- so the only thing that differs is where it lives"
        ),
    )
    parser.add_argument(
        "--orderings",
        choices=("factors", "depth-first"),
        default="factors",
        help="which microbatch orderings the search tries per geometry:"
        " every depth x breadth factor pair (the default), or only the"
        " depth-first walk. The loss stays paired and the backward walk"
        " reversed either way; the search does not toggle those",
    )
    parser.add_argument(
        "--search-workers",
        type=int,
        default=0,
        help="CPU planner threads per process; 0 selects automatically",
    )
    parser.add_argument(
        "--resolution-options",
        type=named_resolution_options,
        default=NAMED_RESOLUTION_OPTIONS["quarters"],
        help="which resolutions the search and the runs plan: the shares of"
        " flexible groups to recompute, as 'quarters' (the library"
        " default), 'eighths', 'halves', or a comma-separated list of exact"
        " fractions such as 0,1/2,7/8,1. More shares plan more programs per"
        " point; on the llama3 frontier eighths cost 1.75x the search for a"
        " median gain of nothing",
    )
    parser.add_argument(
        "--transfer-bandwidths",
        type=_transfer_bandwidths,
        default=None,
        help="plan against this calibration instead of the one the runtime"
        " measures at start: FETCH_SOLO,FETCH_CONCURRENT,EVICT_SOLO,EVICT_CONCURRENT"
        " in GB/s, optionally followed by two latencies in microseconds. A pair"
        " sets fixed fetch/evict rates. Or give the path of another"
        " run's search.json to pin to what that run planned against. The run"
        " phase plans against the same lanes as the search either way",
    )
    parser.add_argument("--plots", action="store_true")
    parser.add_argument(
        "--timelines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write every plan's pages under timelines/ as the run closes: the"
        " pools and the lanes over the step, on the simulated clock for every"
        " plan the search made and on the device's too for every budget that"
        " ran; --no-timelines skips them",
    )
    parser.add_argument(
        "--resolution-plans",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep every resolution's best plan in the plan store beside the"
        " answer, certified, so the timelines carry a page per resolution;"
        " several times the plan store, so off by default",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="replace an existing run at the output directory. Its artifact"
        " store is kept, being a content-addressed cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where this run's search report, log, traced steps and figures"
        " are written; defaults to benchmarking/quickstart_reports/"
        "<model>_<revision>_<MMDD_HHMM>/seq<length>/seqsperstep<n>",
    )
    parser.add_argument(
        "--reproduce",
        type=Path,
        default=None,
        metavar="RUN",
        help="repeat the run at RUN (its seq<length>/seqsperstep<n> directory)"
        " exactly: every setting is read from its request.json, the search is"
        " pinned to the calibration its search.json records, and plan-store"
        " mode is require, so a plan the store lacks refuses instead of being"
        " searched again. Only --output-dir, --plots, --timelines and"
        " --resolution-plans may be given with it",
    )
    parser.add_argument(
        "--export-bypass-key",
        default=None,
        help="the caller's name for the code this run builds from; with it, a"
        " build reads each ordering's step program back from the build store"
        " and captures only what is not there. Without it every build captures",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model-dtype",
        choices=_DTYPE_NAMES,
        default="bfloat16",
        help="model weights and activations (default: bfloat16 on every GPU);"
        " use float16 on devices without BF16 support",
    )
    parser.add_argument(
        "--master-dtype",
        type=_dtype_name,
        default=None,
        metavar="DTYPE",
        help="keep a master copy of every weight trained at another dtype at"
        " this one -- float32, say -- and step the masters in the weights'"
        " place; none, the default, steps the weights themselves",
    )
    parser.add_argument(
        "--grad-dtype",
        type=_dtype_name,
        default=None,
        metavar="DTYPE",
        help="the dtype gradients are created and summed at over a step's"
        " microbatches, the weights' own by default. Naming one also asks"
        " the mlops kernels for weight gradients at it and has the optimizer"
        " read gradients at it, so nothing rounds them on the way",
    )
    parser.add_argument(
        "--opt-state-dtype",
        choices=(*_DTYPE_NAMES, "parameter"),
        default=None,
        help="the dtype the optimizer keeps its state at: a dtype, or"
        " parameter for the dtype of what it steps. Its own default when"
        " not given",
    )
    parser.add_argument(
        "--parameter-rounding",
        choices=_ROUNDINGS,
        default=None,
        help="how the optimizer rounds the weights it steps: to nearest, its"
        " default, or stochastically, which keeps small updates in expectation",
    )
    parser.add_argument(
        "--opt-state-rounding",
        choices=_ROUNDINGS,
        default=None,
        help="how the optimizer rounds the state it stores: nearest or stochastic."
        " ShadowSpill defaults to stochastic for BF16 AdamW moments, nearest otherwise",
    )
    parser.add_argument(
        "--round-accumulation-once",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="let a matrix multiply add its product into running gradients"
        " kept narrower than it sums at as it writes them, rounding the sum"
        " once instead of twice; off by default, which is what PyTorch's"
        " own step computes",
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
        " read and write; point it at another run's store to skip work"
        " already paid for there. Overrides --artifact-store for the build"
        " tree",
    )
    parser.add_argument(
        "--plan-store",
        type=Path,
        default=None,
        help="where this run's plans go: every request, result and plan"
        " manifest, kept apart from the build store so a shared store never"
        " hands a run another run's plans. Overrides --artifact-store for the"
        " planning tree. Defaults to <output-dir>/plan_store",
    )
    for tree in ("build", "plan"):
        parser.add_argument(
            f"--{tree}-store-mode",
            choices=STORE_MODES,
            default="contribute",
            help=f"what this run may do about a {tree} artifact the store does"
            " not hold: contribute builds it and writes it back, reuse builds"
            " it and persists nothing, require refuses and names it",
        )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="make the search reproduce exactly at any worker count: a"
        " candidate's placement gate consults only its own placed plans"
        " rather than the shared best-placed record, so every graph-pair"
        " selection reports the plan it actually found. On by default,"
        " because the figures compare selections; --no-deterministic lets"
        " the shared bound skip measuring plans that cannot win, at the"
        " cost of selections that show up or not depending on timing",
    )
    parser.add_argument(
        "--incumbents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="hand each budget the best plan found at a smaller budget of the"
        " same program as the plan to beat, so no program plans worse with"
        " more memory; a point that did not beat it answers with it and says"
        " which budget it came from. --no-incumbents searches every point"
        " alone, for comparing the two",
    )
    profiling = parser.add_argument_group("Task profiling")
    defaults = ProfilingOptions()
    for option in fields(ProfilingOptions):
        default = getattr(defaults, option.name)
        profiling.add_argument(
            "--profile-" + option.name.replace("_", "-"),
            dest="profile_" + option.name,
            type=type(default),
            default=default,
            help=f"{option.name.replace('_', ' ')} (default: {default}); "
            "duration targets use exact-task device time, wall limits use host time",
        )
    return parser


def parse_arguments() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """The flags, with the choices a flag alone can settle already settled."""

    parser = _parser()
    arguments = parser.parse_args()
    if arguments.reproduce is not None:
        arguments = _reproduced_arguments(parser, arguments)
    elif arguments.model is None and arguments.factory is None:
        parser.error(
            "a text preset or --factory is required unless --reproduce names a run"
        )
    if arguments.model is not None and arguments.factory is not None:
        parser.error("choose a text preset or --factory")
    # An unspecified placement is the library's to choose. Resolving it here
    # rather than defaulting the flag keeps one answer to the question: a change
    # to the library default reaches this tour, and the banner reports what the
    # planner will actually do rather than what this file last believed.
    if arguments.host_headroom_gib < 0 or arguments.preparation_timeout <= 0:
        parser.error(
            "host headroom must be nonnegative and preparation timeout positive"
        )
    if arguments.steps < 1:
        parser.error("--steps must be at least 1")
    try:
        profiling_policy(arguments)
        validate_resolution_options(arguments.resolution_options)
    except ValueError as error:
        parser.error(str(error))

    return parser, arguments
