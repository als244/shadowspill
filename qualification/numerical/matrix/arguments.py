"""Matrix selection and numerical-case command-line options."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Final

from qualification.device_defaults import add_memory_budget_arguments
from qualification.numerical.cases import DEFAULT_DEVICE_BUDGETS
from qualification.numerical.references import DEFAULT_REFERENCE_DIRECTORY
from qualification.numerical.spill import add_spill_arguments
from qualification.precision import add_dtype_arguments

FAMILIES: Final = tuple(DEFAULT_DEVICE_BUDGETS)
IMPLEMENTATIONS: Final = ("pytorch", "mlops")
DEFAULT_IMPLEMENTATIONS: Final = {
    "llama3": IMPLEMENTATIONS,
    "qwen35": IMPLEMENTATIONS,
    "olmoe": ("mlops",),
}


def parse_bytes(value: str) -> int:
    normalized = value.strip().lower().replace("_", "")
    factors = (("gib", 1 << 30), ("mib", 1 << 20), ("kib", 1 << 10))
    for suffix, factor in factors:
        if normalized.endswith(suffix):
            number = normalized[: -len(suffix)]
            if not number.isdigit():
                raise argparse.ArgumentTypeError(f"invalid byte count {value!r}")
            result = int(number) * factor
            if result <= 0:
                raise argparse.ArgumentTypeError("byte count must be positive")
            return result
    if not normalized.isdigit() or int(normalized) <= 0:
        raise argparse.ArgumentTypeError(f"invalid positive byte count {value!r}")
    return int(normalized)


def parse_budget_overrides(
    values: list[str], *, valid_models: set[str] | None = None
) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        family, separator, budget = value.partition("=")
        if (
            separator == ""
            or not family
            or (valid_models is not None and family not in valid_models)
        ):
            raise argparse.ArgumentTypeError(
                "budget must be MODEL=BYTES for one of "
                + ", ".join(sorted(valid_models or ()))
            )
        result[family] = parse_bytes(budget)
    return result


def create_parser() -> argparse.ArgumentParser:
    """Which cells to run, what to run them with, and where to put them."""

    parser = argparse.ArgumentParser(
        description=(
            "Run fresh-process compiled-reference/planned parity, checkpoint "
            "replay, transfer, and physical-budget gates while reporting "
            "graph-pair selections diagnostically."
        )
    )
    parser.add_argument(
        "--models",
        "--families",
        dest="models",
        nargs="+",
        default=FAMILIES,
        help="built-in family names or names consumed by --case-factory",
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=IMPLEMENTATIONS,
        help=(
            "explicit provider cross-product; omitted runs the supported "
            "five-cell matrix (pure-PyTorch OLMoE is deferred)"
        ),
    )
    parser.add_argument(
        "--budget",
        action="append",
        default=[],
        metavar="FAMILY=BYTES",
        help="override one family budget; suffixes KiB, MiB, and GiB are accepted",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qualification/results/numerical_matrix"),
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIRECTORY,
        help=(
            "canonical compiled-reference root; one identity-checked reference "
            "is retained under each model/provider directory"
        ),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--empty-caches",
        action="store_true",
        help=(
            "start every subprocess with empty Inductor and Triton caches "
            "in a temporary directory removed "
            "afterwards, so nothing is reused from an earlier run or another "
            "case"
        ),
    )
    add_dtype_arguments(parser)
    add_memory_budget_arguments(parser)
    add_spill_arguments(parser)
    parser.add_argument("--seed", type=int, default=20_260_811)
    parser.add_argument(
        "--optimizer-ordering",
        choices=("stage_interleaved", "tail"),
        default="stage_interleaved",
        help=(
            "group optimizer updates by stage and place them at their gradient frontier"
        ),
    )
    parser.add_argument(
        "--data-ordering",
        help="how every selected case walks its microbatches, as"
        " <depth>x<breadth> with r for the reversed backward walk and p for"
        " the paired loss, for example 2x4rp; the product must be the case's"
        " microbatch count. Omitted plans depth-first, which is what every"
        " stored reference was compared against so far",
    )
    parser.add_argument(
        "--model-config",
        default="{}",
        metavar="JSON|@FILE",
        help="model configuration passed to every selected case",
    )
    parser.add_argument(
        "--data-geometry",
        metavar="JSON|@FILE",
        help="microbatch geometry passed to every selected case",
    )
    parser.add_argument(
        "--case-factory",
        metavar="MODULE:FUNCTION",
        help="qualification-case factory for custom model names",
    )
    parser.add_argument(
        "--case-option",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="repeatable custom-factory argument",
    )
    parser.add_argument(
        "--regenerate-reference",
        action="store_true",
        help="replace canonical references instead of reusing compatible files",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue remaining cases after a failed case",
    )
    parser.add_argument(
        "--detailed-artifacts",
        action="store_true",
        help=(
            "retain saved Plans and per-task traces; compact correctness "
            "evidence is the default"
        ),
    )
    return parser


def budget_overrides_for_args(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> dict[str, int]:
    """Every selected model must have a name and a budget before anything runs."""

    try:
        invalid_names = [
            name
            for name in arguments.models
            if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None
        ]
        if invalid_names:
            raise RuntimeError(
                "model names must contain only letters, digits, '.', '_', or '-': "
                + ", ".join(invalid_names)
            )
        custom_names = [name for name in arguments.models if name not in FAMILIES]
        if custom_names and arguments.case_factory is None:
            raise RuntimeError(
                "custom model names require --case-factory MODULE:FUNCTION"
            )
        overrides = parse_budget_overrides(
            arguments.budget, valid_models=set(arguments.models)
        )
        missing_budgets = [name for name in custom_names if name not in overrides]
        if missing_budgets:
            raise RuntimeError(
                "custom model budgets must be explicit with --budget: "
                + ", ".join(missing_budgets)
            )
    except (argparse.ArgumentTypeError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    return overrides
