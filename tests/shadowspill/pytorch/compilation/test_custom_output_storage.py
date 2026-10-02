"""Opaque operator allocations can exceed the span of their returned view."""

from __future__ import annotations

import torch

from shadowspill.pytorch.capture.aot import capture_forward, inference_artifact
from shadowspill.pytorch.compilation.compiler import compile_artifact


@torch.library.custom_op("shadowspill_storage_test::padded", mutates_args=())
def _padded(value: torch.Tensor) -> list[torch.Tensor]:
    return [value.new_zeros(4, dtype=torch.int32)[:2]]


@_padded.register_fake
def _padded_fake(value):
    return [value.new_empty(4, dtype=torch.int32)[:2]]


def test_compiler_preserves_opaque_output_backing_extent():
    class Model(torch.nn.Module):
        def forward(self, value):
            return _padded(value)[0]

    value = torch.ones(1)
    artifact = inference_artifact(capture_forward(Model(), (value,)))
    compiled = compile_artifact(
        artifact, device_ordinal=0, representative_arguments=(value,)
    )
    (output,) = compiled()
    (allocation,) = compiled.manifest.root_allocations
    (view,) = compiled.manifest.storage_contract.output_views
    assert view.span_bytes == 8
    assert allocation.requested_bytes == output.untyped_storage().nbytes() == 16
    torch.testing.assert_close(output, torch.zeros(2, dtype=torch.int32))
