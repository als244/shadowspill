from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.compilation.compiler import (
    CompiledTask,
    compile_artifact,
    materialize_example_arguments,
)
from shadowspill.pytorch.compilation.inductor import (
    compile_explicit_inductor_task,
    compile_inductor_task,
)
from shadowspill.pytorch.compilation.inductor import compiler as inductor_compiler
from shadowspill.pytorch.optimizer import capture_optimizer
from shadowspill.pytorch.profiling.inputs import materialize_representative_inputs
from shadowspill.task.manifest import ExecutableRootAllocation, ExecutableTaskManifest


class _Add(nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right


class _MultiplyByOne(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.ones_like(value)


def _artifact(kind: str = "inference") -> GraphArtifact:
    inputs = (torch.randn(8, 8), torch.randn(8, 8))
    return GraphArtifact.capture(
        kind=kind,  # type: ignore[arg-type]
        graph_module=torch.fx.symbolic_trace(_Add()),
        example_inputs=inputs,
    )


def _manifest(artifact: GraphArtifact) -> ExecutableTaskManifest:
    return ExecutableTaskManifest(
        semantic_contract_digest=artifact.storage_contract.compatibility_digest,
        storage_contract=artifact.storage_contract,
        contract_capture_ns=0,
        compatibility_digest="0" * 64,
        root_allocations=tuple(
            ExecutableRootAllocation(
                root.root_id,
                0 if root.kind.value == "input" else root.minimum_span_bytes,
            )
            for root in artifact.storage_contract.roots
        ),
    )


def _compiled_task(
    artifact: GraphArtifact,
    function: Any,
    arguments: tuple[object, ...] = (),
) -> CompiledTask:
    return CompiledTask(artifact, function, arguments, _manifest(artifact))


def test_compiled_task_disables_dispatcher_autograd() -> None:
    observed: list[bool] = []
    artifact = _artifact("forward")
    executable = _compiled_task(
        artifact,
        lambda: observed.append(torch.is_grad_enabled()),
    )

    with torch.enable_grad():
        executable()

    assert observed == [False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_materialization_preserves_storage_alias_and_compiles() -> None:
    source = torch.arange(32, dtype=torch.float32)
    first = source[2:18].view(4, 4)
    second = source[4:20].view(4, 4)
    arguments = materialize_example_arguments(
        (first, second, {"mode": 2}), device_ordinal=0
    )
    left, right, metadata = arguments
    assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
    assert left.untyped_storage()._cdata == right.untyped_storage()._cdata
    assert left.storage_offset() == 2
    assert right.storage_offset() == 4
    assert metadata == {"mode": 2}

    executable = compile_artifact(_artifact(), device_ordinal=0)
    assert tuple(
        item.requested_bytes for item in executable.manifest.root_allocations
    ) == (256,)
    output = executable()
    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(
        output,
        executable.example_arguments[0] + executable.example_arguments[1],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_inductor_cache_restores_the_exact_executable_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    artifact = _artifact()

    first = compile_artifact(artifact, device_ordinal=0)
    first_output = first()
    monkeypatch.setattr(
        inductor_compiler,
        "_graph_lowering_contract",
        lambda *args, **kwargs: pytest.fail(
            "warm AOT/Inductor cache unexpectedly rebuilt GraphLowering"
        ),
    )
    second = compile_artifact(artifact, device_ordinal=0)
    second_output = second()

    torch.testing.assert_close(first_output, second_output)
    assert first.manifest.compatibility_digest == second.manifest.compatibility_digest
    assert first.manifest.storage_contract == second.manifest.storage_contract
    assert tuple((tmp_path / "shadowspill" / "task_manifests").rglob("*.json"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_inductor_manifest_captures_post_grad_output_alias() -> None:
    value = torch.randn(32)
    artifact = GraphArtifact.capture(
        kind="inference",
        graph_module=torch.fx.symbolic_trace(_MultiplyByOne()),
        example_inputs=(value,),
    )
    assert artifact.storage_contract.roots[0].kind.value == "fresh"

    executable = compile_artifact(artifact, device_ordinal=0)
    compiled_contract = executable.manifest.storage_contract
    assert compiled_contract.roots[0].kind.value == "input"
    assert compiled_contract.roots[0].source_input == 0
    output = executable()
    assert isinstance(output, torch.Tensor)
    assert output.data_ptr() == executable.example_arguments[0].data_ptr()  # type: ignore[union-attr]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("module", [_Add(), _MultiplyByOne()])
def test_explicit_inductor_path_matches_outer_aot_for_inference(
    module: nn.Module,
) -> None:
    inputs = (
        (torch.randn(8, 8), torch.randn(8, 8))
        if isinstance(module, _Add)
        else (torch.randn(8, 8),)
    )
    artifact = GraphArtifact.capture(
        kind="inference",
        graph_module=torch.fx.symbolic_trace(module),
        example_inputs=inputs,
    )
    arguments = tuple(
        value.detach() if isinstance(value, torch.Tensor) else value
        for value in materialize_example_arguments(inputs, device_ordinal=0)
    )

    outer = compile_inductor_task(
        artifact.graph_module,
        arguments,
        semantic_contract=artifact.storage_contract,
    )
    explicit = compile_explicit_inductor_task(
        artifact.graph_module,
        arguments,
        semantic_contract=artifact.storage_contract,
    )

    torch.testing.assert_close(
        explicit.function(*arguments),
        outer.function(*arguments),
    )
    assert explicit.manifest.storage_contract == outer.manifest.storage_contract
    assert explicit.manifest.root_allocations == outer.manifest.root_allocations
    phase_names = {name for name, _duration in explicit.phase_timings_ns}
    assert "torch_decomposition_normalization" in phase_names
    assert "torch_inductor_core" in phase_names
    assert all(duration > 0 for _name, duration in explicit.phase_timings_ns)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_optimizer_compilation_uses_no_grad_mutation_contract() -> None:
    model = nn.Sequential(nn.Linear(6, 10), nn.Linear(10, 3))
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    # The update is captured over state that exists, as planning installs it.
    optimizer.step()
    captured = capture_optimizer(dict(model.named_parameters()), optimizer)
    assert captured.update is not None

    executable = compile_artifact(captured.update, device_ordinal=0)
    representatives = materialize_representative_inputs(
        captured.update, device_ordinal=0
    )
    with torch.no_grad():
        outputs = executable.function(*representatives.arguments)
    assert isinstance(outputs, tuple | list)
    assert len(outputs) == len(captured.mutation_names)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_explicit_optimizer_preserves_outer_aot_mutation_contract() -> None:
    model = nn.Sequential(nn.Linear(6, 10), nn.Linear(10, 3))
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    artifact = capture_optimizer(dict(model.named_parameters()), optimizer).update
    assert artifact is not None

    outer_arguments = tuple(
        value.detach() if isinstance(value, torch.Tensor) else value
        for value in materialize_example_arguments(
            artifact.example_arguments, device_ordinal=0
        )
    )
    explicit_arguments = tuple(
        value.detach() if isinstance(value, torch.Tensor) else value
        for value in materialize_example_arguments(
            artifact.example_arguments, device_ordinal=0
        )
    )
    outer = compile_inductor_task(
        artifact.graph_module,
        outer_arguments,
        semantic_contract=artifact.storage_contract,
    )
    explicit = compile_explicit_inductor_task(
        artifact.graph_module,
        explicit_arguments,
        semantic_contract=artifact.storage_contract,
    )

    outer_outputs = outer.function(*outer_arguments)
    explicit_outputs = explicit.function(*explicit_arguments)
    torch.testing.assert_close(explicit_outputs, outer_outputs)
    torch.testing.assert_close(explicit_arguments, outer_arguments)
    assert explicit.manifest.storage_contract == outer.manifest.storage_contract
    assert explicit.manifest.root_allocations == outer.manifest.root_allocations
