"""Share equivalent frozen kernels without sharing rank-owned capture metadata."""

from dataclasses import replace
from typing import Any

from shadowspill.artifacts._values import digest, plain
from shadowspill.artifacts.program import Program, TaskDefinition
from shadowspill.training.preparation.planning.sharing import computation_identity


def reuse_compiled_tasks(program: Program, canonical: Program) -> Program:
    """Keep the old shared-binary behavior for equivalent per-rank captures.

    The compiler's task identity binds the graph, physical ABI, dependencies,
    toolchain and resources. Finalization adds selected kernels and provider
    preparation. Equivalent full captures use one complete frozen implementation,
    including provider tuning, with rank-local staging metadata. Otherwise only
    tasks with identical provider/ABI contracts may reuse an Inductor winner.
    Profiling still measures every physical worker.
    """
    if (
        {"id", "binding"} <= program.bindings.get("preparation_template", {}).keys()
        and {"id", "binding"}
        <= canonical.bindings.get("preparation_template", {}).keys()
        and computation_identity(program) == computation_identity(canonical)
    ):
        if (
            program.executables == canonical.executables
            and program.operator_binaries == canonical.operator_binaries
            and program.task_definitions == canonical.task_definitions
        ):
            return program
        # The caller regenerates the allocation certificate for this exact
        # binary/template pair before publication. Ownership is not executable.
        return replace(
            canonical,
            bindings={
                **canonical.bindings,
                **{
                    key: program.bindings[key]
                    for key in ("preparation_template", "training_operation")
                    if key in program.bindings
                },
            },
        )
    if program.setup != canonical.setup or program.bindings.get(
        "compiled_device_index"
    ) != canonical.bindings.get("compiled_device_index"):
        return program
    definitions = {row.profile_key: row for row in canonical.task_definitions}
    selected = {}
    for row in program.task_definitions:
        key = row.profile_key
        other = definitions.get(key)
        if (
            other is None
            or key not in program.executables
            or key not in canonical.executables
            or row.workspace_bytes != other.workspace_bytes
        ):
            continue
        local, shared = _identity(program, row), _identity(canonical, other)
        if local is None or shared is None or local != shared:
            continue
        if any(
            program.bindings.get(field, {}).get(key)
            != canonical.bindings.get(field, {}).get(key)
            for field in ("compiler_memory", "task_inputs", "operator_dependencies")
        ):
            continue
        if program.executables[key] != canonical.executables[key]:
            selected[key] = other
    if not selected:
        return program
    bindings = dict(program.bindings)
    for name in ("inductor_kernels", "finalized_tasks"):
        bindings[name] = {
            **bindings.get(name, {}),
            **{key: canonical.bindings[name][key] for key in selected},
        }
    return replace(
        program,
        executables={
            **program.executables,
            **{key: canonical.executables[key] for key in selected},
        },
        task_definitions=tuple(
            replace(
                row, compatibility_digest=selected[row.profile_key].compatibility_digest
            )
            if row.profile_key in selected
            else row
            for row in program.task_definitions
        ),
        bindings=bindings,
    )


def _identity(program: Program, definition: TaskDefinition) -> dict[str, Any] | None:
    key = definition.profile_key
    identity = plain(program.bindings.get("finalized_tasks", {}).get(key))
    if identity is None:
        return None
    if (
        not {"task", "triton_choices", "inductor_kernels", "output_devices"}.issubset(
            identity
        )
        or digest(identity) != definition.compatibility_digest
        or identity["inductor_kernels"]
        != plain(program.bindings.get("inductor_kernels", {}).get(key))
    ):
        raise ValueError(f"task {key}: invalid finalized implementation identity")
    return {**identity, "inductor_kernels": sorted(identity["inductor_kernels"])}
