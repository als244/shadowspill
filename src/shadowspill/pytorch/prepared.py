"""Portable captured computation, separate from model state and runtime ownership.

A prepared step is an explicit program artifact, not a guess that two Python
callables mean the same thing. Loading one deliberately selects that computation.
Only load artifacts from a trusted source, as with PyTorch's graph-pair cache.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import pickle
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode, unset_fake_temporarily

from shadowspill.errors import CaptureError
from shadowspill.planner import StepDataOrdering
from shadowspill.schema import artifact_schema
from shadowspill.step import StepProgram
from shadowspill.store import atomic_json

from .capture.artifacts import ObjectiveSchema
from .capture.fake import fake_device_inputs
from .capture.objective import TrainingObjectiveCapture, bind_training_objective
from .guards import capture_input_signature
from .store import _serializable_export

if TYPE_CHECKING:
    from .planning.artifacts import TrainingCaptureArtifacts


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _model_contract(model: nn.Module) -> str:
    state: dict[str, torch.Tensor] = dict(
        model.named_parameters(remove_duplicate=False)
    )
    state.update(model.named_buffers(remove_duplicate=False))
    return capture_input_signature(state).digest


def _operator_modules(program: torch.export.ExportedProgram) -> set[str]:
    """Remember providers that export executed, but fresh admission will not.

    torch.export records custom operators by dispatcher name, not by their
    defining Python module. Lazy providers must be imported before load can
    resolve those names. This adapter targets the supported PyTorch version.
    """
    from torch._library.custom_ops import _maybe_get_opdef
    from torch.fx.node import map_aggregate

    modules: set[str] = set()

    def visit(value: Any) -> Any:
        if isinstance(value, torch._ops.OpOverload):
            definition = _maybe_get_opdef(value)
            if definition is not None:
                for function in (
                    definition._init_fn,
                    definition._abstract_fn,
                    definition._setup_context_fn,
                    definition._backward_fn,
                ):
                    if function is not None:
                        modules.add(function.__module__)
        return value

    for module in program.graph_module.modules():
        if isinstance(module, torch.fx.GraphModule):
            for node in module.graph.nodes:
                visit(node.target)
                map_aggregate((node.args, node.kwargs), visit)
    return modules


@dataclass(frozen=True, slots=True)
class PreparedStep:
    """Exported objective templates and the programs searched across budgets.

    No weights, optimizer state, runtime, pool, or executable callable is owned.
    ``programs`` contain the measured planning problems, one per ordering.
    Their profiling environment and maximum supported budgets remain explicit.
    Admission binds fresh state and reconstructs occurrence-specific bindings
    from these exports and the content-addressed graph-pair/profile stores.
    """

    programs: tuple[StepProgram, ...]
    exports: tuple[bytes, ...]
    schemas: tuple[ObjectiveSchema, ...]
    positions: tuple[int, ...]
    signature_digests: tuple[str, ...]
    model_contract: str
    options: Mapping[str, Any]
    operator_modules: tuple[str, ...] = ()

    @classmethod
    def from_capture(
        cls,
        model: nn.Module,
        captured: TrainingCaptureArtifacts,
        *,
        options: Mapping[str, Any],
    ) -> PreparedStep:
        templates: dict[int, int] = {}
        exports: list[bytes] = []
        schemas: list[ObjectiveSchema] = []
        positions: list[int] = []
        operator_modules: set[str] = set()
        for capture in captured.captures:
            program = capture.exported.exported_program
            key = id(program)
            if key not in templates:
                templates[key] = len(exports)
                operator_modules.update(_operator_modules(program))
                stream = io.BytesIO()
                with unset_fake_temporarily():
                    torch.export.save(_serializable_export(program), stream)
                exports.append(stream.getvalue())
                schemas.append(capture.objective_schema)
            positions.append(templates[key])
        return cls(
            (),
            tuple(exports),
            tuple(schemas),
            tuple(positions),
            tuple(item.digest for item in captured.signatures),
            _model_contract(model),
            dict(options),
            tuple(sorted(operator_modules)),
        )

    def validate(
        self,
        model: nn.Module,
        signature_digests: Sequence[str],
        options: Mapping[str, Any],
    ) -> None:
        if _model_contract(model) != self.model_contract:
            raise CaptureError("prepared step model state geometry or aliasing differs")
        if tuple(signature_digests) != self.signature_digests:
            raise CaptureError("prepared step microbatch signatures differ")
        if dict(options) != dict(self.options):
            raise CaptureError("prepared step capture options differ")

    def bind(
        self,
        model: nn.Module,
        inputs: tuple[tuple[object, ...], ...],
        *,
        fake_mode: FakeTensorMode,
        device_ordinal: int,
    ) -> tuple[TrainingObjectiveCapture, ...]:
        # Export deserialization creates its own metadata fake mode. Do not
        # install the binding mode until the programs have been reconstructed.
        from torch._export.serde.serialize import _reconstruct_fake_tensor

        with (
            unset_fake_temporarily(),
            torch.serialization.safe_globals([_reconstruct_fake_tensor]),
        ):
            for module in self.operator_modules:
                importlib.import_module(module)
            programs = tuple(
                torch.export.load(io.BytesIO(data)) for data in self.exports
            )
        captures = []
        with fake_mode:
            for position, microbatch in zip(self.positions, inputs, strict=True):
                values = fake_device_inputs(
                    microbatch, fake_mode, device_index=device_ordinal
                )
                captures.append(
                    bind_training_objective(
                        programs[position],
                        model,
                        self.schemas[position],
                        values,
                    )
                )
        return tuple(captures)

    def save(self, directory: str | os.PathLike[str]) -> None:
        """Publish a complete artifact, with file hashes and a manifest written last.

        Metadata uses Python serialization to retain arbitrary static metric
        leaves and their pytree types. This is a trusted local build artifact.
        """
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        payloads = {
            f"export-{index}.pt2": data for index, data in enumerate(self.exports)
        }
        payloads["metadata.pkl"] = pickle.dumps(
            {
                "schemas": self.schemas,
                "positions": self.positions,
                "signature_digests": self.signature_digests,
                "model_contract": self.model_contract,
                "options": dict(self.options),
                "operator_modules": self.operator_modules,
            }
        )
        for index, program in enumerate(self.programs):
            payloads[f"program-{index}.json"] = program.to_json().encode()
        files = {}
        for name, data in payloads.items():
            # Content-addressed names keep an earlier manifest valid while a
            # writer publishes a replacement, including after interruption.
            digest = _digest(data)
            filename = f"{digest}-{name}"
            destination = root / filename
            if not destination.exists():
                with tempfile.NamedTemporaryFile(dir=root, delete=False) as temporary:
                    temporary.write(data)
                os.replace(temporary.name, destination)
            files[name] = {"file": filename, "sha256": digest}
        atomic_json(
            root / "manifest.json",
            {
                "schema": artifact_schema("pytorch.prepared_step"),
                "torch": str(torch.__version__),
                "files": files,
                "exports": len(self.exports),
                "programs": len(self.programs),
                "orderings": [p.data_ordering.to_dict() for p in self.programs],
            },
        )

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        ordering: StepDataOrdering | None = None,
    ) -> PreparedStep:
        """Read a trusted artifact, optionally selecting one ordering for admission."""
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema") != artifact_schema("pytorch.prepared_step"):
            raise ValueError("unsupported prepared step schema")
        if manifest.get("torch") != str(torch.__version__):
            raise ValueError("prepared step was built with a different PyTorch version")
        indices = tuple(range(manifest["programs"]))
        if ordering is not None and "orderings" in manifest:
            if len(manifest["orderings"]) != len(indices):
                raise ValueError("invalid prepared step ordering index")
            indices = tuple(
                i for i in indices if manifest["orderings"][i] == ordering.to_dict()
            )
            if not indices:
                raise ValueError(
                    "prepared step does not contain the requested ordering"
                )
        required = {
            "metadata.pkl",
            *(f"export-{i}.pt2" for i in range(manifest["exports"])),
            *(f"program-{i}.json" for i in indices),
        }
        payloads = {}
        for name in sorted(required):
            entry = manifest["files"][name]
            filename = entry["file"]
            if Path(filename).name != filename:
                raise ValueError("invalid prepared step artifact path")
            data = (root / filename).read_bytes()
            if _digest(data) != entry["sha256"]:
                raise ValueError(f"prepared step artifact checksum differs: {name}")
            payloads[name] = data
        metadata = pickle.loads(payloads["metadata.pkl"])
        result = cls(
            programs=tuple(
                StepProgram.from_json(payloads[f"program-{i}.json"].decode())
                for i in indices
            ),
            exports=tuple(
                payloads[f"export-{i}.pt2"] for i in range(manifest["exports"])
            ),
            **metadata,
        )
        if not result.programs or any(
            p.signature_digests != result.signature_digests for p in result.programs
        ):
            raise ValueError("prepared step programs do not match the captured inputs")
        if ordering is not None:
            from dataclasses import replace

            selected = tuple(p for p in result.programs if p.data_ordering == ordering)
            if not selected:
                raise ValueError(
                    "prepared step does not contain the requested ordering"
                )
            if "orderings" in manifest and len(selected) != len(result.programs):
                raise ValueError(
                    "prepared step ordering index differs from its programs"
                )
            result = replace(result, programs=selected)
        if (
            len(result.positions) != len(result.signature_digests)
            or any(not 0 <= i < len(result.exports) for i in result.positions)
            or len(result.schemas) != len(result.exports)
        ):
            raise ValueError("invalid prepared step export bindings")
        return result


__all__ = ["PreparedStep"]
