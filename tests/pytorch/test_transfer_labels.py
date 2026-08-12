from __future__ import annotations

from shadowspill.pytorch._transfer_labels import TransferLabelIndex
from tests.ir._examples import representative_program, representative_schedule


def test_transfer_labels_describe_object_relationships_and_execution_tasks() -> None:
    program = representative_program()
    schedule = representative_schedule()
    labels = TransferLabelIndex(
        program,
        {
            "forward_save": "execution_000000.forward.encoder",
            "backward_marker": "execution_000001.control.backward_marker",
            "forward_recompute": "execution_000002.recompute.encoder",
            "consume": "execution_000003.backward.encoder",
        },
    ).labels_for(schedule.actions[:2])

    assert labels[0] == (
        "shadowspill.runtime.transfer.evict.activation_storage."
        "role_activation.bytes_128.from_output."
        "execution_000000.forward.encoder.trigger."
        "execution_000000.forward.encoder"
    )
    assert labels[1] == (
        "shadowspill.runtime.transfer.fetch.activation_storage."
        "role_activation.bytes_128.for_input."
        "execution_000003.backward.encoder.trigger."
        "execution_000001.control.backward_marker"
    )
