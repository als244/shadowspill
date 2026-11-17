"""CPU verification of all-value allocation guarantees for prepared captures.

Compiler geometry plus source-owned opaque-call guarantees establish invariance.
Measured profiles still supply actual bytes/alias traces, and the existing runtime
admission engine still binds those traces to the complete operation family.
"""

from collections.abc import Mapping
from hashlib import sha256
from typing import Any

from shadowspill.artifacts._values import digest, plain
from shadowspill.artifacts.program import Program
from shadowspill.serialization.sharing import writing_metadata

from .templates import CaptureTemplate

_SCOPE = (
    "fixed tensor geometry/layout and captured non-tensor arguments; all tensor values"
)

# These implementations dispatch on captured shapes, dtypes and scalar constants,
# never tensor data. Do not extend this to all ATen/custom operators. The engine
# source pin and Program toolchain pin qualify these exact rules.
_BUILTIN = {
    **dict.fromkeys(
        (
            "extern_kernels.mm",
            # aten.mm.dtype_out uses the same dense addmm implementation with
            # beta=0, alpha=1 and a captured output dtype. Its output/copies and
            # BLAS workspace depend on geometry/layout, never tensor values.
            "extern_kernels.mm_dtype",
            "extern_kernels.addmm",
            "extern_kernels.bmm",
            "extern_kernels.baddbmm",
        ),
        "shadowspill/aten-dense-matmul-allocation@1",
    ),
    "torch.ops.shadowspill.accumulate_matmul_.default": (
        "shadowspill/accumulate-matmul-allocation@1"
    ),
    # Task-boundary gradient storage uses an input-shaped clone or a captured
    # padded element count plus an input-shaped view. Neither reads values to
    # choose allocation extents; the implementation is part of the engine pin.
    "torch.ops.shadowspill.materialize_gradient.default": (
        "shadowspill/gradient-storage-allocation@1"
    ),
    "torch.ops.shadowspill.pad_storage.default": (
        "shadowspill/gradient-storage-allocation@1"
    ),
    # Inductor RandomSeeds writes a captured count of int64 seeds into its
    # preallocated output. ATen's out/random kernel uses only that fixed extent,
    # dtype and scalar range; previous output values and RNG state cannot change
    # storage geometry. This does not certify other random/provider operators.
    "aten.randint.low_out": "shadowspill/aten-randint-out-allocation@1",
}


def allocation_coverage(template: CaptureTemplate, program: Program) -> dict[str, Any]:
    """Derive an inspectable proof without framework imports or code execution."""
    contracts = dict(_BUILTIN)
    declarations = template.signature()["allocation_invariants"]
    for owner, declared in declarations.items():
        for name, contract in declared.items():
            if name in contracts and contracts[name] != contract:
                raise ValueError(
                    f"conflicting allocation contract for {name} ({owner})"
                )
            contracts[name] = contract

    collective = set(program.bindings.get("collective_tasks", {}))
    called = {task.profile_id for task in program.tasks if task.requires_entrypoint}
    control = {
        task.profile_id for task in program.tasks if not task.requires_entrypoint
    } - called
    # Saved additive-gradient forms may not occur in the fresh task graph.
    # Every compiled definition needs coverage, including those derivatives.
    compute = (
        {item.profile_key for item in program.task_definitions} - collective - control
    )
    if compute != program.executables.keys():
        raise ValueError("allocation coverage requires every exact compiled task")
    memory = program.bindings.get("compiler_memory", {})
    if memory.keys() != compute:
        raise ValueError("allocation coverage requires every compiler memory record")
    tasks = {}
    for key in sorted(compute):
        behavior = memory[key].get("allocation_behavior")
        if (
            not isinstance(behavior, Mapping)
            or set(behavior) != {"version", "scope", "opaque_calls", "unresolved"}
            or type(behavior["version"]) is not int
            or behavior["version"] != 1
            or behavior["scope"] != _SCOPE
        ):
            raise ValueError(f"{key}: missing supported allocation behavior")
        opaque, unknown = behavior["opaque_calls"], behavior["unresolved"]
        if (
            not isinstance(opaque, (tuple, list))
            or any(not isinstance(name, str) or not name for name in opaque)
            or not isinstance(unknown, (tuple, list))
        ):
            raise ValueError(f"{key}: malformed allocation behavior")
        if unknown:
            raise ValueError(f"{key}: unresolved allocation behavior: {list(unknown)}")
        missing = set(opaque) - contracts.keys()
        if missing:
            raise ValueError(
                f"{key}: missing all-value allocation contract: {sorted(missing)}"
            )
        tasks[key] = {
            "executable": sha256(program.executables[key]).hexdigest(),
            "compiler_memory": digest(memory[key]),
            "contracts": {name: contracts[name] for name in sorted(set(opaque))},
        }
    preparations = {}
    for provider in program.bindings.get("operator_preparation", {}).get(
        "providers", ()
    ):
        name = provider["factory"]
        if name not in contracts:
            raise ValueError(f"missing bound-task allocation contract: {name}")
        preparations[name] = contracts[name]
    with writing_metadata():
        captured = program.to_dict()
        captured["bindings"].pop("allocation_coverage", None)
        captured_digest = digest(captured)
    return {
        "version": 1,
        "scope": _SCOPE,
        "template": template.id,
        "capture": captured_digest,
        "tasks": tasks,
        "preparations": preparations,
        # Collective byte/protocol evidence is separately checked by existing
        # runtime admission and cooperating-worker issue-order validation.
        "collectives": digest(program.bindings.get("collective_tasks", {})),
    }


def validate_allocation_coverage(template: CaptureTemplate, program: Program) -> None:
    if plain(program.bindings.get("allocation_coverage")) != allocation_coverage(
        template, program
    ):
        raise ValueError("saved all-value allocation coverage differs from the capture")
