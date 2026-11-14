"""Functionalized state updates must not lend mutable storage to caller results."""

import torch
from torch import nn

from shadowspill.build.outputs import own_public_outputs
from shadowspill.pytorch.capture.export import capture_forward


class MutatedOutputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("counter", torch.zeros(4))

    def forward(self, x):
        self.counter.add_(x)
        x.mul_(2)
        return self.counter, self.counter[::2], x, x[::2]


def test_owned_views_of_updated_state_and_inputs_keep_values_after_mutation():
    capture = capture_forward(MutatedOutputs(), (torch.ones(4),))
    assert len(capture.mutations) == 2
    own_public_outputs(capture)
    outputs = capture.exported_program.graph_module(*capture.flat_inputs)
    returned = tuple(outputs[index] for index in capture.user_output_indices)
    assert (
        returned[0].untyped_storage().data_ptr()
        == returned[1].untyped_storage().data_ptr()
    )
    assert (
        returned[2].untyped_storage().data_ptr()
        == returned[3].untyped_storage().data_ptr()
    )
    snapshots = [value.clone() for value in returned]
    for mutation in capture.mutations:
        value = outputs[mutation.output_index]
        assert all(
            value.untyped_storage().data_ptr() != result.untyped_storage().data_ptr()
            for result in returned
        )
        value.zero_()
    torch.testing.assert_close(returned, tuple(snapshots), rtol=0, atol=0)
