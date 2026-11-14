"""Ready frames retain the exact saved view ABI, including empty rank slices."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from engine_tests.fixtures.definition_operations import definitions, records
from engine_tests.training.test_prepared_requests import family, upload
from engine_tests.training.test_setup_resolution import arguments, installed
from shadowspill.parallelism import (
    GroupRef,
    InputFieldPlacement,
    InputPlacement,
    OutputPlacement,
    Parallelism,
)
from shadowspill.pytorch.guards import capture_input_signature
from shadowspill.training.inputs.adapters import InputAdapter
from shadowspill.training.inputs.materialize import materialize_buffer
from shadowspill.training.inputs.routing import route_buffer
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.templates import capture_inventory
from shadowspill.training.session.prepared import PreparedRequests
from shadowspill.training.session.worker_inputs import ReadyInputLoader
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.inputs import TensorField
from shadowspill_client.protocol.payload import snapshot_batch


@pytest.mark.parametrize("workers", [2, 3])
def test_worker_reconstructs_strides_storage_extents_bf16_and_empty_views(
    tmp_path, workers
):
    calls = []
    model, _, _ = definitions()

    def adapter(config, layout):
        def fill(context, fields):
            calls.append((context.routed.owner.location, context.layout.logical_rank))
            fields["inputs", "metadata"][:] = np.uint16(0x3F80)

        return InputAdapter(
            metadata=[
                TensorField(("inputs", "metadata"), "bfloat16", (2,)),
                TensorField(("inputs", "empty"), "float32", (2, 0, 3)),
            ],
            fill=fill,
            padding_safe=True,
        )

    placement = InputPlacement(
        GroupRef("together", (tuple(range(workers)),)),
        fields=[InputFieldPlacement(("inputs", "pixels"), "slice", 3)],
        outputs={"prediction": OutputPlacement("leader")},
    )
    setup = resolve_training(
        installed(model=replace(model.definition, input_adapter=adapter)),
        **arguments(),
        parallelism=Parallelism(ranks_per_replica=workers, inputs=placement),
        operations={"forward": ["predictions"]},
    )
    inv = capture_inventory(prepare_layouts(setup))
    prepared = PreparedRequests(family(inv), cache_root=tmp_path)
    key = setup.config.operations[0]
    bound = inv.layouts.staging[key]
    command, _ = upload(
        records(targets=False), operation="forward", outputs="predictions"
    )
    with bound.reserve(BufferBudget(bound.staging_bytes)) as reservation:
        arena = reservation.budget
        with snapshot_batch(records(targets=False), arena) as snapshot:
            data = prepared(command, snapshot, arena)
            try:
                assert len(calls) == len(data.spool.frames)
                expected = {}
                for buffer in range(len(data.assignment.schedule.buffers)):
                    with materialize_buffer(
                        data.assignment, setup.context.schema, buffer, arena
                    ) as physical:
                        for rank in range(workers):
                            binding = data.candidate.source.layout.bindings[
                                buffer, rank
                            ]
                            with binding.inputs.adapt(
                                route_buffer(physical, placement, rank), arena
                            ) as adapted:
                                args = binding.arguments(adapted)
                                expected[buffer, rank] = (
                                    capture_input_signature((args, {})),
                                    args[0]["pixels"].tolist(),
                                )
                                del args
                completed = len(calls)
                before = arena.used
                for rank in range(workers):
                    frames = [
                        (frame, data.candidate.source.layout.bindings[buffer, worker])
                        for (buffer, worker), frame in sorted(data.spool.frames.items())
                        if worker == rank
                    ]
                    with ReadyInputLoader(
                        frames, data.controls.loss_params, arena
                    ) as loader:
                        with pytest.raises(ValueError, match="once in order"):
                            loader(1)
                        for buffer in range(len(data.assignment.schedule.buffers)):
                            args, kwargs = loader(buffer)
                            signature, pixels = expected[buffer, rank]
                            assert capture_input_signature((args, kwargs)) == signature
                            assert args[0]["pixels"].tolist() == pixels
                            torch.testing.assert_close(
                                args[0]["metadata"], torch.ones(2, dtype=torch.bfloat16)
                            )
                            assert arena.used[0] <= bound.staging_bytes
                            del args, kwargs
                    assert arena.used == before
                    with pytest.raises(RuntimeError, match="closed"):
                        loader(0)
                assert len(calls) == completed  # No adapter ran during worker reads.
            finally:
                data.close()
        assert arena.used == (0, 0)
    assert not list(tmp_path.iterdir())
