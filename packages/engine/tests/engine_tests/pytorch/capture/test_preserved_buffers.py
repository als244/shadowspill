"""Real functional export keeps output values without committing buffer writes."""

import pytest
import torch
from torch import nn
from torch.export.graph_signature import OutputKind

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.buffers import preserve_buffers, preserved_buffer_names
from shadowspill.pytorch.capture.export import capture_forward, inference_artifact


class Counted(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4.0))
        self.register_buffer("counter", torch.arange(4.0))
        self.register_buffer("alias", self.counter)
        self.register_buffer("unused", torch.zeros(1))

    def forward(self, x):
        self.alias.add_(2)
        x.add_(3)
        return x * self.weight + self.counter, self.alias[1:3], None


def test_functional_buffer_updates_still_affect_outputs_but_not_registered_state():
    model = Counted()
    names = preserved_buffer_names(model, ("counter", "unused"))
    assert set(names) == {"counter", "alias", "unused"}
    with torch.no_grad():
        capture = preserve_buffers(capture_forward(model, (torch.ones(4),)), names)
    # Preserve unrelated input effects so operation validation can reject them.
    assert [m.kind for m in capture.mutations] == [OutputKind.USER_INPUT_MUTATION]
    assert all(m.target != "counter" for m in capture.mutations)
    captured = capture.exported_program.module()
    for _ in range(2):
        x = torch.ones(4)
        values, view, literal = captured(x)
        torch.testing.assert_close(
            values, 4 * torch.arange(4.0) + torch.arange(4.0) + 2
        )
        torch.testing.assert_close(view, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(x, torch.full((4,), 4.0))
        torch.testing.assert_close(model.counter, torch.arange(4.0))
        assert literal is None
    artifact = inference_artifact(capture)
    assert len(artifact.storage_contract.mutations) == 1


@pytest.mark.parametrize("names", [("missing",), ("weight",), "counter"])
def test_preservation_requires_registered_buffer_names(names):
    with pytest.raises(CaptureError, match="registered model buffers"):
        preserved_buffer_names(Counted(), names)


def test_preservation_cannot_hide_writes_to_parameter_backed_buffers():
    model = Counted()
    model.register_buffer("parameter_view", model.weight.detach()[1:])
    with pytest.raises(CaptureError, match="alias model parameters"):
        preserved_buffer_names(model, ("parameter_view",))


def test_unselected_buffer_effects_are_not_silently_removed():
    model = Counted()
    with torch.no_grad():
        capture = capture_forward(model, (torch.ones(4),))
    kept = preserve_buffers(capture, preserved_buffer_names(model, ("unused",)))
    assert kept is capture
    assert {m.kind for m in kept.mutations} == {
        OutputKind.BUFFER_MUTATION,
        OutputKind.USER_INPUT_MUTATION,
    }
