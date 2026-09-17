"""The command line: one case as a reference arm, a planned arm, or both."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from workloads.numerical import DEFAULT_DEVICE_BUDGETS

from .orchestrate import orchestrate
from .planned import planned_worker
from .reference_arm import reference_worker
from .references import DEFAULT_APPROXIMATELY_1B_REFERENCE_DIRECTORY
from .request import CaseRequest, PlannedRequest


def json_argument(value: str, *, description: str) -> Any:
    """Decode one JSON argument, or the file an ``@path`` argument names."""

    source = value
    if value.startswith("@"):
        path = Path(value[1:]).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"{description} file does not exist: {path}")
        source = path.read_text()
    try:
        return json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {description} JSON: {exc}") from exc


def case_option_values(values: list[str]) -> dict[str, Any]:
    """Decode the repeatable ``NAME=JSON`` custom-factory options."""

    result: dict[str, Any] = {}
    for value in values:
        name, separator, encoded = value.partition("=")
        if separator == "" or not name:
            raise ValueError("case options must use NAME=JSON")
        result[name] = json_argument(encoded, description=f"case option {name!r}")
    return result


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    case, checkpoint_step, profiling_metadata = _decode_case(parser, arguments)
    _dispatch(parser, arguments, case, checkpoint_step, profiling_metadata)
    return 0


def _remote_spill(parser: argparse.ArgumentParser, value: str | None) -> Any:
    """The pool named by ``--remote-spill``, or ``None`` for pinned host.

    Parsed here rather than in the matrix so that a case run by hand behaves
    exactly as one the matrix spawned -- which is the whole reason the option
    travels as a string.
    """

    if value is None:
        return None
    host, _, rest = value.partition(":")
    port, _, size = rest.partition(":")
    if not host or not port.isdigit() or not size.isdigit():
        parser.error(f"--remote-spill must read HOST:PORT:BYTES, not {value!r}")
    from shadowspill.network import remote

    return remote(capacity=int(size), host=host, port=int(port))


def _parser() -> argparse.ArgumentParser:
    """Every option the three modes share, and the paths each one takes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "_reference", "_planned"))
    parser.add_argument("family", help="built-in family or custom model name")
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--device-budget", type=int)
    parser.add_argument(
        "--model-implementation",
        choices=("pytorch", "mlops"),
        default="pytorch",
        help="pure PyTorch is the formal numerical authority",
    )
    parser.add_argument("--seed", type=int, default=20_260_811)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--optimizer-ordering",
        choices=("stage_interleaved", "tail"),
        default="stage_interleaved",
        help="place grouped optimizer stages as soon as their gradients are final",
    )
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument(
        "--allow-fully-resident",
        action="store_true",
        help="do not require real FETCH/EVICT activity",
    )
    parser.add_argument(
        "--model-config",
        default="{}",
        metavar="JSON|@FILE",
        help="built-in dataclass field overrides or custom-factory configuration",
    )
    parser.add_argument(
        "--data-ordering",
        help="how the step walks its microbatches, as <depth>x<breadth> with"
        " r for the reversed backward walk and p for the paired loss, for"
        " example 2x4rp; omitted plans depth-first as every step did before",
    )
    parser.add_argument(
        "--data-geometry",
        metavar="JSON|@FILE",
        help="microbatch geometry list; omitted uses the built-in two-shape gate",
    )
    parser.add_argument(
        "--profiling-metadata",
        metavar="JSON|@FILE",
        help=(
            "one JSON-compatible workload descriptor per microbatch; used only "
            "for value-sensitive profile/cache identity"
        ),
    )
    parser.add_argument(
        "--artifact-store",
        type=Path,
        help="roots both stores (run mode defaults below the result dir)",
    )
    parser.add_argument("--build-store", type=Path)
    parser.add_argument("--plan-store", type=Path)
    for tree in ("build", "plan"):
        parser.add_argument(
            f"--{tree}-store-mode",
            choices=("contribute", "reuse", "require", "refresh"),
            default="contribute",
            help=f"what this run may do about a {tree} artifact the store does"
            " not hold: contribute writes it back, reuse persists nothing,"
            " require refuses",
        )
    parser.add_argument(
        "--export-bypass-key",
        help="explicit implementation identity for custom-kernel invalidation",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_APPROXIMATELY_1B_REFERENCE_DIRECTORY,
        help="canonical compiled-reference root used by run mode",
    )
    parser.add_argument(
        "--regenerate-reference",
        action="store_true",
        help="replace the canonical compiled reference in run mode",
    )
    parser.add_argument(
        "--detailed-artifacts",
        action="store_true",
        help=(
            "persist the complete PlanReport, plan records, and per-task "
            "step traces; compact correctness evidence is the default"
        ),
    )
    parser.add_argument(
        "--case-factory",
        metavar="MODULE:FUNCTION",
        help="factory for a model not in the built-in qualification registry",
    )
    parser.add_argument(
        "--case-option",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="repeatable custom-factory option",
    )
    parser.add_argument(
        "--remote-spill",
        metavar="HOST:PORT:BYTES",
        help=(
            "spill to a memory daemon on another machine instead of to pinned "
            "host memory. Everything else about the case is unchanged, which "
            "is what makes the comparison mean something"
        ),
    )
    return parser


def _decode_case(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> tuple[CaseRequest, int, list[Any] | None]:
    """Decode the JSON arguments into the case, refusing what cannot be run."""

    family = str(arguments.family)
    model_implementation = arguments.model_implementation
    try:
        if arguments.steps < 2:
            raise ValueError("steps must be at least two")
        checkpoint_step = arguments.checkpoint_step or max(1, arguments.steps - 2)
        if checkpoint_step < 1 or checkpoint_step >= arguments.steps:
            raise ValueError("checkpoint step must be between one and steps - 1")
        model_config_value = json_argument(
            arguments.model_config, description="model config"
        )
        if not isinstance(model_config_value, dict):
            raise ValueError("model config must decode to an object")
        data_geometry_value = None
        if arguments.data_geometry is not None:
            decoded_geometry = json_argument(
                arguments.data_geometry, description="data geometry"
            )
            if not isinstance(decoded_geometry, list) or not all(
                isinstance(item, dict) for item in decoded_geometry
            ):
                raise ValueError("data geometry must decode to a list of objects")
            data_geometry_value = decoded_geometry
        profiling_metadata_value = None
        if arguments.profiling_metadata is not None:
            decoded_metadata = json_argument(
                arguments.profiling_metadata,
                description="profiling metadata",
            )
            if not isinstance(decoded_metadata, list):
                raise ValueError("profiling metadata must decode to a list")
            profiling_metadata_value = decoded_metadata
        case_options_value = case_option_values(arguments.case_option)
    except ValueError as exc:
        parser.error(str(exc))
    case = CaseRequest(
        family=family,
        model_implementation=model_implementation,
        seed=arguments.seed,
        model_config=model_config_value,
        data_geometry=data_geometry_value,
        case_factory=arguments.case_factory,
        case_options=case_options_value,
        optimizer_ordering=arguments.optimizer_ordering,
        data_ordering=arguments.data_ordering,
        steps=arguments.steps,
    )
    if family not in DEFAULT_DEVICE_BUDGETS and arguments.case_factory is None:
        parser.error("unknown model name requires --case-factory MODULE:FUNCTION")
    if (
        arguments.family not in DEFAULT_DEVICE_BUDGETS
        and arguments.case_factory is None
    ):
        parser.error("unknown model name requires --case-factory MODULE:FUNCTION")
    return case, checkpoint_step, profiling_metadata_value


def _dispatch(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
    case: CaseRequest,
    checkpoint_step: int,
    profiling_metadata_value: list[Any] | None,
) -> None:
    """Run the mode the command line named, with the paths it takes."""

    family = case.family
    model_implementation = case.model_implementation
    if arguments.mode == "run":
        if len(arguments.paths) != 1:
            parser.error("run requires one result directory")
        if arguments.device_budget is None and family not in DEFAULT_DEVICE_BUDGETS:
            parser.error("custom model run requires --device-budget")
        orchestrate(
            family,
            model_implementation,
            Path(arguments.paths[0]),
            arguments.device_budget or DEFAULT_DEVICE_BUDGETS.get(family, 0),
            seed=arguments.seed,
            model_config_argument=arguments.model_config,
            data_geometry_argument=arguments.data_geometry,
            case_factory=arguments.case_factory,
            case_option_arguments=arguments.case_option,
            optimizer_ordering=arguments.optimizer_ordering,
            data_ordering=arguments.data_ordering,
            steps=arguments.steps,
            checkpoint_step=checkpoint_step,
            require_pressure=not arguments.allow_fully_resident,
            artifact_store=arguments.artifact_store,
            build_store=arguments.build_store,
            plan_store=arguments.plan_store,
            profiling_metadata_argument=arguments.profiling_metadata,
            build_store_mode=arguments.build_store_mode,
            plan_store_mode=arguments.plan_store_mode,
            export_bypass_key=arguments.export_bypass_key,
            reference_directory=arguments.reference_dir,
            regenerate_reference=arguments.regenerate_reference,
            detailed_artifacts=arguments.detailed_artifacts,
        )
    elif arguments.mode == "_reference":
        if len(arguments.paths) != 1:
            parser.error("_reference requires one output path")
        reference_worker(case, Path(arguments.paths[0]))
    else:
        if len(arguments.paths) != 3:
            parser.error("_planned requires reference, result, and device budget")
        planned_worker(
            PlannedRequest(
                case=case,
                reference_path=Path(arguments.paths[0]),
                result_path=Path(arguments.paths[1]),
                device_budget=int(arguments.paths[2]),
                checkpoint_step=checkpoint_step,
                require_pressure=not arguments.allow_fully_resident,
                artifact_store=arguments.artifact_store,
                build_store=arguments.build_store,
                plan_store=arguments.plan_store,
                profiling_metadata=profiling_metadata_value,
                build_store_mode=arguments.build_store_mode,
                plan_store_mode=arguments.plan_store_mode,
                export_bypass_key=arguments.export_bypass_key,
                detailed_artifacts=arguments.detailed_artifacts,
                spill_pool=_remote_spill(parser, arguments.remote_spill),
            )
        )
