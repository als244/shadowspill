"""Process-local compiled callable ownership for structural profiling."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.compilation import compiler as compiler_api
from shadowspill.pytorch.compilation.compiler import CompiledTask, CompiledTaskSet
from shadowspill.pytorch.compilation.inductor import ExecutableTaskManifest
from shadowspill.pytorch.contracts import CompilationError
from shadowspill.pytorch.optimizer import OpaqueOptimizerArtifact

from .inputs import RepresentativeInputSummary, materialize_representative_inputs
from .runner import ProfilableArtifact


@dataclass(frozen=True, slots=True)
class ProfileExecutable:
    """A compiled task paired with one profiling occurrence's values."""

    compiled: CompiledTask
    artifact: GraphArtifact
    example_arguments: tuple[object, ...] = ()
    representative_inputs: tuple[RepresentativeInputSummary, ...] = ()

    @property
    def function(self) -> Callable[..., object]:
        return self.compiled.function

    @property
    def manifest(self) -> ExecutableTaskManifest:
        return self.compiled.manifest

    @property
    def execution_provider(self) -> str:
        return self.compiled.execution_provider

    @property
    def graph_node_count(self) -> int:
        return self.compiled.graph_node_count

    def __call__(self) -> object:
        with torch.no_grad():
            return self.function(*self.example_arguments)


class ProfileExecutableStore:
    """Own compiled code while releasing occurrence-local CUDA values eagerly."""

    def __init__(self, *, device_ordinal: int) -> None:
        self._device_ordinal = device_ordinal
        self._items: dict[str, ProfileExecutable] = {}
        self._warmed: set[str] = set()
        self._compilation_wall_time_ns = 0
        self._phase_totals: dict[str, int] = {}
        self._phases_by_abi: dict[str, tuple[tuple[str, int], ...]] = {}

    @property
    def compilation_wall_time_ns(self) -> int:
        return self._compilation_wall_time_ns

    @property
    def compilation_phase_timings_ns(self) -> tuple[tuple[str, int], ...]:
        return tuple(self._phase_totals.items())

    @property
    def compilation_phase_timings_by_abi(
        self,
    ) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
        return tuple(sorted(self._phases_by_abi.items()))

    def prepare_manifests(
        self,
        artifacts: Sequence[ProfilableArtifact],
        *,
        progress: Callable[[int, int, str, str], None] | None = None,
    ) -> dict[str, ExecutableTaskManifest]:
        """Compile missing structural tasks and return their storage manifests."""

        unique = _unique_graph_artifacts(artifacts)
        manifests: dict[str, ExecutableTaskManifest] = {}
        for index, digest in enumerate(sorted(unique), start=1):
            artifact = unique[digest]
            executable = self._items.get(digest)
            state = "available"
            if executable is None:
                state = "compiling"
                executable = self._compile(artifact)
                self._items[digest] = executable
            if progress is not None:
                progress(index, len(unique), state, digest)
            manifests[digest] = executable.manifest
            if digest not in self._warmed and executable.example_arguments:
                self._items[digest] = _without_arguments(executable)
        return manifests

    def get(self, artifact: GraphArtifact) -> ProfileExecutable:
        """Return compiled code rebound to this occurrence without creating values."""

        digest = artifact.compatibility_digest
        executable = self._items.get(digest)
        if executable is None:
            executable = self._compile(artifact)
        elif executable.artifact is not artifact:
            executable = replace(executable, artifact=artifact)
        self._items[digest] = executable
        return executable

    def with_arguments(self, executable: ProfileExecutable) -> ProfileExecutable:
        """Materialize deterministic values for one structural occurrence."""

        executable = self._with_arguments(executable)
        self._items[executable.artifact.compatibility_digest] = executable
        return executable

    def release_occurrence_values(self, executable: ProfileExecutable) -> None:
        self._items[executable.artifact.compatibility_digest] = _without_arguments(
            executable
        )

    def mark_warmed(self, digest: str) -> None:
        self._warmed.add(digest)

    def remove(self, digest: str) -> None:
        self._items.pop(digest, None)
        self._warmed.discard(digest)

    def take_selected(
        self,
        artifacts: Sequence[ProfilableArtifact],
        *,
        warmup: Callable[[ProfileExecutable, str], None],
        progress: Callable[[int, int, str, str], None] | None = None,
    ) -> CompiledTaskSet:
        """Transfer selected callables, warming cache-only entrypoints as needed."""

        selected = _selected_graph_artifacts(artifacts)
        functions: dict[str, Callable[..., object]] = {}
        manifests: dict[str, ExecutableTaskManifest] = {}
        for index, digest in enumerate(sorted(selected), start=1):
            artifact = selected[digest]
            executable = self._items.pop(digest, None)
            if progress is not None:
                progress(
                    index,
                    len(selected),
                    "warmed" if executable is not None else "compiling",
                    digest,
                )
            if executable is None:
                executable = self._compile(artifact)
            elif executable.artifact is not artifact:
                executable = replace(executable, artifact=artifact)
            if digest not in self._warmed:
                if not executable.example_arguments:
                    executable = self._with_arguments(executable)
                warmup(executable, digest)
            self._warmed.discard(digest)
            functions[digest] = executable.function
            manifests[digest] = executable.manifest
        return CompiledTaskSet(functions, manifests)

    def discard(self) -> None:
        self._items.clear()
        self._warmed.clear()

    def _compile(self, artifact: GraphArtifact) -> ProfileExecutable:
        started = time.perf_counter_ns()
        try:
            try:
                compiled = compiler_api.compile_artifact(
                    artifact,
                    device_ordinal=self._device_ordinal,
                )
            except CompilationError as error:
                if error.structural_abi is not None:
                    raise
                raise _compilation_error(artifact, error) from error
            except BaseException as error:
                raise _compilation_error(artifact, error) from error
            phases = compiled.compilation_phase_timings_ns
            for name, duration in phases:
                self._phase_totals[name] = self._phase_totals.get(name, 0) + duration
            self._phases_by_abi[artifact.compatibility_digest] = phases
            return ProfileExecutable(
                compiled,
                artifact,
                example_arguments=compiled.example_arguments,
            )
        finally:
            self._compilation_wall_time_ns += time.perf_counter_ns() - started

    def _with_arguments(self, executable: ProfileExecutable) -> ProfileExecutable:
        representatives = materialize_representative_inputs(
            executable.artifact,
            device_ordinal=self._device_ordinal,
        )
        arguments = tuple(
            value.detach() if isinstance(value, torch.Tensor) else value
            for value in representatives.arguments
        )
        return replace(
            executable,
            example_arguments=arguments,
            representative_inputs=representatives.summaries,
        )


def _unique_graph_artifacts(
    artifacts: Sequence[ProfilableArtifact],
) -> dict[str, GraphArtifact]:
    return {
        artifact.compatibility_digest: artifact
        for artifact in artifacts
        if isinstance(artifact, GraphArtifact)
    }


def _selected_graph_artifacts(
    artifacts: Sequence[ProfilableArtifact],
) -> dict[str, GraphArtifact]:
    selected: dict[str, GraphArtifact] = {}
    for artifact in artifacts:
        if isinstance(artifact, OpaqueOptimizerArtifact):
            continue
        if not isinstance(artifact, GraphArtifact):
            raise TypeError(
                f"unsupported executable artifact {type(artifact).__name__}"
            )
        selected.setdefault(artifact.compatibility_digest, artifact)
    return selected


def _without_arguments(executable: ProfileExecutable) -> ProfileExecutable:
    return replace(
        executable,
        compiled=replace(executable.compiled, example_arguments=()),
        example_arguments=(),
        representative_inputs=(),
    )


def _compilation_error(
    artifact: GraphArtifact,
    cause: BaseException,
) -> CompilationError:
    operators = tuple(artifact.operator_targets)
    operator_text = ", ".join(operators) or "none"
    return CompilationError(
        "ShadowSpill failed to compile structural ABI "
        f"{artifact.compatibility_digest} "
        f"(kind={artifact.kind}, operators=[{operator_text}]): {cause}",
        structural_abi=artifact.compatibility_digest,
        task_kind=artifact.kind,
        operators=operators,
    )


__all__ = ["ProfileExecutable", "ProfileExecutableStore"]
