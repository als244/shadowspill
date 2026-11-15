"""Saved-family selection and actual owned uploads; no native/GPU claims here."""

import io
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from engine_tests.fixtures.definition_operations import definitions, records
from engine_tests.fixtures.external_definition import SOURCE
from engine_tests.server.test_request_records import request, store
from engine_tests.training.test_preparation_templates import inventory
from engine_tests.training.test_setup_resolution import arguments, installed
from shadowspill.training.definitions import (
    ModelDefinition,
    ObjectiveDefinition,
    PreparationDefinition,
    Registry,
    RepresentativePattern,
    RepresentativeSample,
)
from shadowspill.training.inputs.adapters import InputAdapter
from shadowspill.training.inputs.packing import PackingError
from shadowspill.training.inputs.shapes import BufferShape
from shadowspill.training.inputs.staging import (
    InputEnvelope,
    InputStaging,
    StagingLimitError,
    combine_input_staging,
)
from shadowspill.training.preparation.family_records import (
    PlannedCandidate,
    PreparedPlans,
    WorkerPlans,
)
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.scalars import control_values
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.templates import capture_inventory
from shadowspill.training.session import prepared as prepared_module
from shadowspill.training.session.dispatch import RequestDispatcher
from shadowspill.training.session.inputs import RequestInputs
from shadowspill.training.session.prepared import PreparedRequests
from shadowspill.training.session.state import OperationState
from shadowspill_client import (
    BatchConfig,
    BuildCandidates,
    Datum,
    Execution,
    Memory,
    Model,
    Objective,
    Operation,
    PlanningOptions,
    Preparation,
    Weights,
)
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.controls import ControlSchema, ScalarControl
from shadowspill_client.protocol.definitions import OutputSet
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField
from shadowspill_client.protocol.payload import MEDIA_TYPE, snapshot_batch
from shadowspill_client.protocol.payload_sizes import PayloadSize
from shadowspill_client.protocol.requests import ArtifactRef, OperationRequest
from shadowspill_client.protocol.results import OperationResult


def family(inv=None):
    """Selection isolation: prepared CPU layouts with sentinel saved Plan IDs."""
    inv = inventory() if inv is None else inv
    plans, families = {}, {}
    for key, rows in inv.families.items():
        selected = []
        for index, source in enumerate(rows):
            name = f"{key.operation}-{key.outputs}-{index}"
            workers = {}
            for rank, uses in source.workers.items():
                worker = WorkerPlans(
                    name,
                    None if uses.masked is None else name + ".mask",
                    None if uses.optimizer is None else name + ".optim",
                )
                workers[rank] = worker
                for member in worker.members():
                    plans.setdefault(rank, {})[member] = SimpleNamespace(
                        timeline=SimpleNamespace(makespan_ns=100 + index),
                        summary={"duration_seconds": (100 + index) * 1e-9},
                        program=SimpleNamespace(
                            bindings={
                                "gradient_scale_control": "scale",
                                "registrations": [],
                            },
                            objects=(),
                            alias_groups=(),
                            tasks=(),
                        ),
                    )
            shape = source.layout.candidate.schedule.buffers[0]
            duration = sum(np.prod(dims) for dims in shape.fields.values())
            selected.append(
                PlannedCandidate(name, source, None, workers, float(duration))
            )
        families[key] = tuple(selected)
    return PreparedPlans(inv, families, plans, inv.layouts.coverage, {}, (), 0)


def upload(datums=None, *, operation="step", name="a", sequence=1, **controls):
    datums = records() if datums is None else datums
    with snapshot_batch(datums, BufferBudget(8 << 20)) as snapshot:
        wire = b"".join(snapshot.chunks())
        ref = ArtifactRef(
            name + ".input", snapshot.sha256, snapshot.size_bytes, MEDIA_TYPE
        )
    command = OperationRequest(
        request(name, sequence).identity, operation, ref, len(datums), **controls
    )
    return command, wire


def queue(journal, tmp_path, inv=None):
    prepared = PreparedRequests(family(inv), cache_root=tmp_path)
    state = OperationState()
    dispatcher = RequestDispatcher(journal, state, max_pending=4)
    inputs = RequestInputs(
        journal, dispatcher.order, prepared.setup.config.execution, prepared
    )
    return prepared, state, dispatcher, inputs


def finish(dispatcher, inputs, command):
    data = inputs.take(command.identity.request_id)

    class Work:
        def close(self):
            pass

    def execute(transition):
        assert data.worker_plans(transition).keys() == data.candidate.workers.keys()
        return Work()

    def publish(receipt, work):
        return OperationResult(
            receipt.identity,
            receipt.operation,
            receipt.history_id,
            receipt.outcome,
            receipt.completed_updates,
            receipt.accumulation,
            optim=data.controls.optim,
            loss_params=data.controls.loss_params,
            grad_scale=data.controls.grad_scale,
        )

    result = dispatcher.run(
        command.identity.request_id, execute, publish, participates=data.participates
    )
    inputs.finish(command.identity.request_id)
    return data, result


def test_real_upload_selects_only_prepared_shapes_and_defaults_are_nonsticky(tmp_path):
    with store(tmp_path) as journal:
        prepared, state, dispatcher, inputs = queue(journal, tmp_path)
        for i, overrides in enumerate(
            (
                {
                    "optim": {"lr": 0.03},
                    "loss_params": {"temperature": 2.0},
                    "grad_scale": -0.5,
                },
                {},
            ),
            1,
        ):
            command, wire = upload(name=f"call-{i}", sequence=i, **overrides)
            prepared.register(inputs, command)
            assert (
                inputs.receive(command.identity.request_id, io.BytesIO(wire)).state
                == "accepted"
            )
            data = inputs.take(command.identity.request_id)
            assert data.candidate.source.layout.candidate.count == 2
            assert data.assignment.schedule.buffers[0].max_datums == 2
            assert sorted(i for row in data.assignment.records for i in row) == [
                0,
                1,
                2,
            ]
            assert data.participates  # The third Datum has zero numerical weight.
            assert data.controls.grad_scale == (-0.5 if overrides else 1.0)
            assert data.controls.loss_params["temperature"] == (
                2.0 if overrides else 1.0
            )
            assert data.controls.optim["lr"] == pytest.approx(
                0.03 if overrides else 0.01
            )
            paths = [frame.path for frame in data.spool.frames.values()]
            for frame in data.spool.frames.values():
                with frame.read(BufferBudget(frame.layout.size.staging)) as values:
                    assert frame.binding.frame_arrays(values)[
                        "inputs", "pixels"
                    ].shape == (2, 1, 2, 2)

            # Execution borrows the same prepared owner once; finish helper must
            # not take it a second time.
            class Work:
                def close(self):
                    pass

            def execute(transition, data=data):
                assert set(data.worker_plans(transition).values()) == {
                    data.candidate.workers[0].main
                }
                return Work()

            def publish(receipt, work, data=data):
                return OperationResult(
                    receipt.identity,
                    receipt.operation,
                    receipt.history_id,
                    receipt.outcome,
                    receipt.completed_updates,
                    receipt.accumulation,
                    optim=data.controls.optim,
                    loss_params=data.controls.loss_params,
                    grad_scale=data.controls.grad_scale,
                )

            outcome = dispatcher.run(
                command.identity.request_id,
                execute,
                publish,
                participates=data.participates,
            )
            assert outcome.state == "completed"
            inputs.finish(command.identity.request_id)
            assert data.closed and all(not p.exists() for p in paths)
            assert inputs.staging.used == (0, 0)
        assert state.boundary.completed_updates == 2


@pytest.mark.parametrize("output", [None, "predictions"])
def test_forward_default_selects_the_prepared_output_before_decoding(tmp_path, output):
    with store(tmp_path) as journal:
        prepared, state, dispatcher, inputs = queue(journal, tmp_path)
        command, wire = upload(
            records(targets=False), operation="forward", outputs=output
        )
        assert prepared.register(inputs, command).state == "uploading"
        assert inputs.receive("a", io.BytesIO(wire)).state == "accepted"
        data, result = finish(dispatcher, inputs, command)
        assert data.candidate.operation.outputs == "predictions"
        assert not data.controls.loss_params
        assert result.state == "completed" and state.boundary.completed_updates == 0
        assert data.closed and inputs.staging.used == (0, 0)


@pytest.mark.parametrize(
    "outputs", [("metrics", "predictions"), ("predictions", "metrics")]
)
def test_output_defaults_follow_saved_order_within_each_mode(outputs):
    from shadowspill.training.session.controls import operation_key
    from shadowspill_client.protocol.codec import decode, encode

    registry = installed()
    setup = resolve_training(
        registry,
        **arguments(),
        operations={
            "step": ["metrics"],
            "forward": [Operation("predictions", "training"), *outputs],
        },
    )
    # Reopening the immutable configuration must preserve the same defaults.
    setup = resolve_training(registry, config=decode(encode(setup.config)))
    command = OperationRequest(request().identity, "forward")
    key = operation_key(setup, command)
    assert key.outputs == outputs[0] and key.mode == "evaluation"
    assert (
        operation_key(setup, replace(command, mode="training")).outputs == "predictions"
    )
    assert (
        operation_key(setup, replace(command, outputs=outputs[1])).outputs == outputs[1]
    )
    for bad in (
        replace(command, outputs="missing"),
        replace(command, mode="training", outputs="metrics"),
        OperationRequest(request().identity, "forward_backward"),
    ):
        with pytest.raises(ValueError, match="not prepared"):
            operation_key(setup, bad)


def test_empty_masked_and_optimizer_forms_follow_the_single_state_machine(tmp_path):
    with store(tmp_path) as journal:
        prepared, state, dispatcher, inputs = queue(journal, tmp_path)
        command, wire = upload(records(masked=True), operation="forward_backward")
        prepared.register(inputs, command)
        inputs.receive("a", io.BytesIO(wire))
        data, result = finish(dispatcher, inputs, command)
        assert not data.participates and result.state == "skipped"
        assert state.boundary.accumulation.open
        optim = OperationRequest(request("optim", 2).identity, "optim_step")
        prepared.register(inputs, optim)
        assert inputs.receive("optim").state == "accepted"
        _, result = finish(dispatcher, inputs, optim)
        assert result.state == "skipped" and not state.boundary.accumulation.open
        empty = OperationRequest(request("empty", 3).identity, "step")
        prepared.register(inputs, empty)
        assert inputs.receive("empty").state == "accepted"
        _, result = finish(dispatcher, inputs, empty)
        assert result.state == "skipped" and state.boundary.completed_updates == 0
        assert inputs.staging.used == (0, 0)


@pytest.mark.parametrize("failure", ["coefficient", "dtype", "adapter", "unprepared"])
def test_complete_preexecution_failure_preserves_prior_contribution_and_rejects_suffix(
    tmp_path, failure
):
    model, _, _ = definitions()
    calls = []

    def adapter(config, layout):
        def fill(context, fields):
            calls.append(context.routed.owner.location)
            if failure == "adapter" and len(calls) == 2:
                raise ValueError("bad second microbatch")

        return InputAdapter(fill=fill, padding_safe=True)

    inv = capture_inventory(
        prepare_layouts(
            resolve_training(
                installed(model=replace(model.definition, input_adapter=adapter)),
                **arguments(),
                operations={"forward_backward": ["metrics"]},
            )
        )
    )
    with store(tmp_path) as journal:
        prepared, state, _, inputs = queue(journal, tmp_path, inv)
        state.complete(state.begin("forward_backward", "prior", participates=True))
        before = state.boundary
        options = (
            {"loss_params": {"temperature": 1e100}} if failure == "coefficient" else {}
        )
        data = records()
        if failure == "dtype":
            data[2] = replace(
                data[2], loss_inputs={**data[2].loss_inputs, "class_id": 1.5}
            )
        command, wire = upload(
            data,
            operation="step" if failure == "unprepared" else "forward_backward",
            **options,
        )
        prepared.register(inputs, command)
        queued = OperationRequest(
            request("optim", 2).identity, "optim_step", dependencies=["a"]
        )
        prepared.register(inputs, queued)
        inputs.receive("optim")
        failed = inputs.receive("a", io.BytesIO(wire))
        assert failed.state == "failed" and failed.acceptance is None
        assert failed.failure.execution_effect == "not_started"
        assert state.boundary == before and journal.get("optim")[1].state == "failed"
        assert inputs.staging.used == (0, 0)
        assert not list(tmp_path.glob("inputs-*"))
        assert len(calls) == (2 if failure == "adapter" else 0)


def variable_inventory(batches, *, groups=None):
    registry = Registry()
    schema = InputSchema(
        [
            TensorField((name,), "float32", (name + "_length",), "pack", 0)
            for name in ("x", "y")
        ],
        [
            Capacity(name, "elements", "sum", (name,), 0, scheduled=True)
            for name in ("x", "y")
        ]
        + [Capacity("records", "records", "count")],
    )
    model = ModelDefinition(
        "external/two_axes@1",
        dict,
        lambda _: None,
        schema,
        batching="independent" if groups is None else "grouped",
        input_groups=groups,
        input_adapter=lambda config, layout: InputAdapter(padding_safe=True),
        preparation="external/two_axes_shapes@1",
    )
    objective = ObjectiveDefinition(
        "external/no_loss@1",
        lambda *args: {},
        InputSchema([]),
        outputs={"metrics": OutputSet(model_inputs=[("x",), ("y",)])},
        padding_safe=True,
    )

    def buffers(context, batch, paths, size):
        limits = context.batches[batch].limits
        return [
            BufferShape(
                int(limits["records"]),
                {path: (int(limits[path[-1]]) // size,) for path in paths},
            )
        ]

    preset = PreparationDefinition(
        model.preparation,
        lambda _: Preparation(
            candidates=BuildCandidates(microbatch_sizes=[1, 2]),
            planning=PlanningOptions(microbatch_counts=[1, 2, 3]),
        ),
        buffers,
        representative=lambda context, batch, physical: RepresentativePattern(
            "one",
            [
                RepresentativeSample(
                    {path: 0.0 for path in physical.fields},
                    {"x_length": 1, "y_length": 1},
                )
            ],
        ),
    )
    registry.add_model(model, source=SOURCE)
    registry.add_objective(objective, source=SOURCE)
    registry.add_preparation(preset, source=SOURCE)
    return capture_inventory(
        prepare_layouts(
            resolve_training(
                registry,
                model=Model(model.id, initialize=Weights.random(seed=1)),
                objective=Objective(objective.id),
                batch=batches,
                memory=Memory(1, 1),
                operations={"forward": ["metrics"]},
            )
        )
    )


def test_capacities_from_different_options_cannot_be_combined(tmp_path):
    inv = variable_inventory(
        [
            BatchConfig(limits={"x": x, "y": y, "records": 2})
            for x, y in ((8, 2), (2, 8))
        ]
    )
    with store(tmp_path) as journal:
        prepared, state, _, inputs = queue(journal, tmp_path, inv)
        command, wire = upload(
            [Datum({"x": np.zeros(3), "y": np.zeros(3)})], operation="forward"
        )
        prepared.register(inputs, command)
        failed = inputs.receive("a", io.BytesIO(wire))
        assert (
            failed.state == "failed"
            and "no requested batch option" in failed.failure.message
        )
        assert state.boundary.valid and inputs.staging.used == (0, 0)


def test_padding_cap_and_installed_groups_use_only_requested_witness(
    tmp_path, monkeypatch
):
    grouped = []

    def groups(config, batch):
        grouped.append(len(batch.fields))
        return [[0, 1, 2]]

    inv = variable_inventory(
        [BatchConfig(limits={"x": 128, "y": 128, "records": 3})], groups=groups
    )
    for rows in inv.families.values():
        assert all(row.layout.candidate.count < 3 for row in rows)
    with store(tmp_path) as journal:
        prepared, _, _, inputs = queue(journal, tmp_path, inv)
        command, wire = upload(
            [Datum({"x": np.zeros(40), "y": np.zeros(1)}) for _ in range(3)],
            operation="forward",
        )

        def exhausted(*args, **kwargs):
            raise PackingError("bounded repair stopped")

        monkeypatch.setattr(prepared_module, "pack_batch", exhausted)
        prepared.register(inputs, command)
        assert inputs.receive("a", io.BytesIO(wire)).state == "accepted"
        data = inputs.take("a")
        assert data.assignment.strategy == "coverage"
        assert data.assignment.schedule.buffers[0].fields["inputs", "x"] == (128,)
        assert data.assignment.records == ((0, 1, 2),)
        assert data.assignment.groups == (((0, 1, 2),),) and grouped == [3]
        inputs.finish("a")
        assert data.closed and inputs.staging.used == (0, 0)


def test_three_40k_datums_need_the_admitted_128k_buffer(tmp_path):
    inv = variable_inventory(
        [BatchConfig(limits={"x": 128 * 1024, "y": 4, "records": 3})]
    )
    with store(tmp_path) as journal:
        prepared, _, _, inputs = queue(journal, tmp_path, inv)
        command, wire = upload(
            [Datum({"x": np.zeros(40 * 1024), "y": np.zeros(1)}) for _ in range(3)],
            operation="forward",
        )
        prepared.register(inputs, command)
        assert inputs.receive("a", io.BytesIO(wire)).state == "accepted"
        data = inputs.take("a")
        assert data.assignment.schedule.buffers[0].fields["inputs", "x"] == (
            128 * 1024,
        )
        assert len(data.assignment.schedule.buffers) == 1
        assert sorted(i for row in data.assignment.records for i in row) == [0, 1, 2]
        inputs.finish("a")


def test_grouping_hook_is_explicit_and_must_cover_every_datum_once(tmp_path):
    model, _, _ = definitions()
    with pytest.raises(TypeError, match="input_groups hook"):
        replace(model.definition, batching="grouped")
    with pytest.raises(ValueError, match="requires grouped"):
        replace(model.definition, input_groups=lambda config, batch: [])
    inv = variable_inventory(
        [BatchConfig(limits={"x": 8, "y": 8, "records": 3})],
        groups=lambda config, batch: [[0, 0, 2]],
    )
    with store(tmp_path) as journal:
        prepared, state, _, inputs = queue(journal, tmp_path, inv)
        command, wire = upload(
            [Datum({"x": np.zeros(1), "y": np.zeros(1)}) for _ in range(3)],
            operation="forward",
        )
        prepared.register(inputs, command)
        failed = inputs.receive("a", io.BytesIO(wire))
        assert failed.state == "failed" and "exactly once" in failed.failure.message
        assert state.boundary.valid and inputs.staging.used == (0, 0)


def test_combined_decode_bound_is_admitted_before_any_body_or_worker():
    a = InputStaging(InputEnvelope(1, PayloadSize(1, 1000, 0, 1)), 1, (1,), 1)
    b = InputStaging(InputEnvelope(2, PayloadSize(8000, 1, 0, 8000)), 1, (1,), 2)
    limit = max(a.staging_bytes, b.staging_bytes)
    with pytest.raises(StagingLimitError, match="complete input family"):
        combine_input_staging([a, b], Execution(max_staging_bytes=limit))
    combined = combine_input_staging([a, b], Execution())
    assert combined.envelope.snapshot.staging == 24000
    assert combined.envelope.max_datums == 2 and combined.spool_bytes == 2


def test_cpu_control_representations_and_import_boundary():
    schema = ControlSchema(
        {
            "f": ScalarControl(1.0),
            "i": ScalarControl(0, "int"),
            "b": ScalarControl(False, "bool"),
        }
    )
    values = control_values(schema, {"f": 0.1, "i": (1 << 63) - 1, "b": True})
    assert values["f"] == float(np.float32(0.1)) and values["b"] is True
    for changes in ({"f": 1e100}, {"i": 1 << 63}):
        with pytest.raises(ValueError, match="captured scalar dtype"):
            control_values(schema, {**values, **changes})
    bounded = ControlSchema(
        {"positive": ScalarControl(1.0, minimum=0, minimum_inclusive=False)}
    )
    with pytest.raises(ValueError, match="captured scalar dtype"):
        control_values(bounded, {"positive": 1e-100})
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for name in ('torch', 'triton', 'mlops', 'shadowspill_workloads'):
    sys.modules[name] = None
from engine_tests.training.test_prepared_requests import family
from shadowspill.training.session.prepared import PreparedRequests
from pathlib import Path
assert PreparedRequests(family(), cache_root=Path('/unused')).setup.config.operations
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
