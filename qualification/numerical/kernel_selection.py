"""One measured external-kernel selection shared by both numerical arms.

Only warm-up may tune. Reference execution and build captures pin the recorded
choices; profile/execute replay the choices already packaged by build. This
controls external autotuners, not every Inductor fusion or vendor BLAS heuristic.
Provider-specific configuration remains outside the ShadowSpill engine.
"""

import hashlib
import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

SCHEMA = "qualification.kernel_selection.v1"


def selection_path(reference: Path) -> Path:
    return reference.with_name("kernel-selection.json")


def selection_digest(selection: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            selection, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _target() -> dict[str, Any]:
    import torch

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "name": properties.name,
        "capability": [properties.major, properties.minor],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
    }


def _provider(request: Any) -> Any:
    import torch

    if request.model_implementation != "mlops" or torch.version.hip:
        return None
    from mlops.providers.sonicmoe.moe import CAPABILITIES, available

    if torch.cuda.get_device_capability()[0] not in CAPABILITIES or not available():
        return None
    from mlops import quack_runtime

    quack_runtime.initialize()
    return quack_runtime


@contextmanager
def capture(request: Any) -> Iterator[dict[str, Any]]:
    from shadowspill.pytorch.tuning import capture_choices

    provider = _provider(request)
    with ExitStack() as stack:
        triton = stack.enter_context(capture_choices())
        external = stack.enter_context(provider.capture_choices()) if provider else None
        yield {
            "schema": SCHEMA,
            "case_identity": request.identity(),
            "target": _target(),
            "triton": triton,
            "mlops_sonic": external,
        }


def load(path: Path, request: Any) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            "numerical kernel selection is missing; "
            "regenerate the reference with its warm-up"
        )
    value = json.loads(path.read_text())
    if (
        value.get("schema") != SCHEMA
        or value.get("case_identity") != request.identity()
    ):
        raise RuntimeError(
            "numerical kernel selection identity differs; use --regenerate-reference"
        )
    return value


@contextmanager
def frozen(selection: dict[str, Any], request: Any) -> Iterator[None]:
    from shadowspill.pytorch.tuning import pin_choices

    if selection["target"] != _target():
        raise RuntimeError(
            "numerical kernel selection target differs; use --regenerate-reference"
        )
    provider = _provider(request)
    if (provider is None) != (selection["mlops_sonic"] is None):
        raise RuntimeError(
            "numerical kernel selection provider differs; use --regenerate-reference"
        )
    with ExitStack() as stack:
        stack.enter_context(pin_choices(selection["triton"]))
        if provider:
            stack.enter_context(provider.pin_choices(selection["mlops_sonic"]))
        yield


def save(path: Path, selection: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(selection, indent=2) + "\n")
    temporary.replace(path)
