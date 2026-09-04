"""Report where one qualification cell's step stops being reproducible.

    python -m tools.qualification.nondeterminism llama3
    python -m tools.qualification.nondeterminism olmoe --model-implementation mlops

Takes the same geometry knobs as the numerical matrix, so a cell that fails a
bitwise replay there can be probed here with the same shape and data.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, cast

import torch

from workloads.numerical import build_case
from workloads.providers import ModelImplementation

from .probe import ProbeResult, probe_step

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _report(result: ProbeResult, *, limit: int) -> None:
    first, second = result.objective_values
    print(
        f"  objective reproducible: {result.objective_reproducible}"
        f"   ({first:.10g} vs {second:.10g})"
    )
    print(
        f"  modules observed: {result.modules_observed}"
        f"   gradients observed: {result.gradients_observed}"
    )
    for label, found in (
        ("forward outputs", result.forward_divergences),
        ("incoming gradients", result.backward_divergences),
        ("parameter gradients", result.gradient_divergences),
    ):
        if not found:
            print(f"  {label}: all reproducible")
            continue
        print(f"  {label}: {len(found)} differ, in execution order")
        for item in found[:limit]:
            print(
                f"      {item.name}  max|diff|={item.maximum_absolute:.3e}"
                f"  numel={item.numel:,}"
            )
        if len(found) > limit:
            print(f"      ... and {len(found) - limit} more")
    if result.reproducible:
        print("\n  VERDICT: the step is bitwise reproducible.")
        return
    forward, backward = result.first_forward, result.first_backward
    if forward is not None:
        print(
            f"\n  VERDICT: reproducibility is lost in the forward, first at"
            f" {forward.name}."
        )
    elif backward is not None:
        print(
            f"\n  VERDICT: the forward is reproducible; the backward is not,"
            f" first at {backward.name}."
        )
    else:
        print(
            "\n  VERDICT: only parameter gradients differ, so the"
            " nondeterminism is in a gradient accumulation rather than in"
            " any module's own output."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", help="built-in family or --case-factory name")
    parser.add_argument(
        "--model-implementation", default="mlops", choices=("mlops", "pytorch")
    )
    parser.add_argument("--seed", type=int, default=20_260_811)
    parser.add_argument("--model-config", default="{}")
    parser.add_argument("--data-geometry")
    parser.add_argument("--case-factory")
    parser.add_argument("--case-option", action="append", default=[])
    parser.add_argument(
        "--microbatch",
        type=int,
        default=0,
        help="which microbatch of the case to probe",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="entries printed per stage"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="ask for ordered kernels before probing: mlops's "
        "deterministic_kernels for the operations that take the request, and "
        "torch.use_deterministic_algorithms for the ones that read only that "
        "global. A divergence that survives this comes from somewhere neither "
        "reaches",
    )
    parser.add_argument(
        "--no-modules",
        action="store_true",
        help="compare only the objective and parameter gradients; hooks on "
        "every module cost memory on a large model",
    )
    arguments = parser.parse_args()

    case_options: dict[str, Any] = {}
    for item in arguments.case_option:
        key, _, value = item.partition("=")
        case_options[key] = value
    case = build_case(
        arguments.family,
        model_implementation=cast(ModelImplementation, arguments.model_implementation),
        seed=arguments.seed,
        model_config=json.loads(arguments.model_config),
        data_geometry=(
            json.loads(arguments.data_geometry) if arguments.data_geometry else None
        ),
        case_factory=arguments.case_factory,
        case_options=case_options,
    )
    model = case.model.to(_DEVICE)
    microbatch = [
        item.to(_DEVICE) if isinstance(item, torch.Tensor) else item
        for item in case.microbatches[arguments.microbatch]
    ]

    def objective() -> torch.Tensor:
        result = case.objective(model, *microbatch)
        return cast(torch.Tensor, getattr(result, "loss", result))

    if arguments.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    print(
        f"{arguments.family}/{arguments.model_implementation} on {_DEVICE}"
        f"   deterministic_algorithms="
        f"{torch.are_deterministic_algorithms_enabled()}"
    )
    with case.implementations(deterministic=arguments.deterministic):
        result = probe_step(model, objective, capture_modules=not arguments.no_modules)
    _report(result, limit=arguments.limit)
    return 0 if result.reproducible else 1


if __name__ == "__main__":
    raise SystemExit(main())
