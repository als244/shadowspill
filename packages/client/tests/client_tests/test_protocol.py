"""Request identity and truthful failure effects survive canonical wire records."""

import gc
import json
import weakref
from dataclasses import replace

import pytest

from shadowspill_client import Datum, Model
from shadowspill_client.protocol.codec import (
    MAX_METADATA_NODES,
    ProtocolError,
    RecordCodec,
    decode,
    digest,
    encode,
)
from shadowspill_client.protocol.requests import (
    ArtifactRef,
    OperationRequest,
    RequestIdentity,
)
from shadowspill_client.protocol.results import (
    Acceptance,
    Accumulation,
    ArtifactIssue,
    Failure,
    OperationResult,
    RequestStatus,
)

IDENTITY = RequestIdentity("request_2", "run_1", 2, "writer_1", 1, "server_1")
PAYLOAD = ArtifactRef("payload_1", "a" * 64, 64)


def result():
    return OperationResult(
        IDENTITY,
        "step",
        "history_1",
        "completed",
        3,
        metrics={"loss_sum": 0.5},
        outputs={"logprobs": PAYLOAD},
        artifact_issues=[
            ArtifactIssue("trace_failed", "Optional trace conversion failed")
        ],
        optim={"lr": 0.01},
        grad_scale=0.5,
        loss=0.75,
    )


@pytest.mark.parametrize(
    "value",
    [
        OperationRequest(IDENTITY, "step", PAYLOAD, 2, optim={"lr": 0.01}),
        OperationRequest(IDENTITY, "forward_backward", PAYLOAD, 2),
        OperationRequest(IDENTITY, "forward", PAYLOAD, 2, mode="training"),
        OperationRequest(IDENTITY, "optim_step", grad_scale=-1),
        OperationRequest(IDENTITY, "discard"),
        OperationRequest(IDENTITY, "close"),
        OperationResult(IDENTITY, "discard", "history_1", "completed", 3),
        OperationResult(IDENTITY, "close", "history_1", "completed", 3),
        OperationResult(
            IDENTITY,
            "close",
            "history_1",
            "completed",
            3,
            accumulation=Accumulation(1, True, ("old",), 1),
            state_valid=False,
        ),
        Acceptance(IDENTITY, 3),
        result(),
        RequestStatus(IDENTITY, "completed", Acceptance(IDENTITY, 3), result()),
        RequestStatus(
            IDENTITY,
            "failed",
            failure=Failure(
                "bad_batch",
                "Invalid input",
                "not_started",
                True,
                accumulation=Accumulation(1, True, ["request_1"], 1),
            ),
        ),
        RequestStatus(
            IDENTITY,
            "failed",
            Acceptance(IDENTITY, 3),
            failure=Failure(
                "worker_lost", "Worker exited after dispatch", "unknown", False
            ),
        ),
        RequestStatus(
            IDENTITY,
            "failed",
            Acceptance(IDENTITY, 3),
            failure=Failure(
                "outputs_failed",
                "Update completed but output publication failed",
                "completed",
                True,
                3,
            ),
        ),
    ],
)
def test_versioned_wire_roundtrip(value):
    restored = decode(encode(value))
    assert type(restored) is type(value)
    assert restored == value


def test_request_hash_covers_identity_controls_and_payload_integrity():
    left = OperationRequest(
        IDENTITY, "step", PAYLOAD, 2, optim={"lr": 1.0, "weight_decay": 0.01}
    )
    right = replace(left, optim={"weight_decay": 0.01, "lr": 1.0})
    assert digest(left) == digest(right)
    assert digest(left) != digest(
        replace(left, payload=replace(PAYLOAD, sha256="b" * 64))
    )
    assert digest(left) != digest(
        replace(left, identity=replace(IDENTITY, request_id="request_3"))
    )
    assert digest(left) != digest(replace(left, grad_scale=0))
    assert left.grad_scale == 1
    assert OperationRequest(IDENTITY, "forward").mode == "evaluation"


@pytest.mark.parametrize(
    "options",
    [
        {"payload": PAYLOAD},
        {"datum_count": 1},
        {"optim": {"lr": 1}},
        {"grad_scale": 1},
        {"loss_params": {"temperature": 1}},
        {"outputs": "predictions"},
        {"mode": "training"},
        {"runtime_trace": True},
    ],
)
@pytest.mark.parametrize("operation", ["discard", "close"])
def test_state_commands_reject_compute_inputs_and_controls(options, operation):
    with pytest.raises(ValueError):
        OperationRequest(IDENTITY, operation, **options)


@pytest.mark.parametrize(
    "options",
    [
        {"metrics": {"value": 1}},
        {"outputs": {"value": PAYLOAD}},
        {"loss": 0},
        {"outcome": "skipped"},
        {"grad_scale": 1},
        {"optim": {"lr": 1}},
        {"loss_params": {"temperature": 1}},
        {"accumulation": Accumulation(1, True, ("old",), 1)},
    ],
)
@pytest.mark.parametrize("operation", ["discard", "close"])
def test_state_command_result_records_only_resolved_state(options, operation):
    result = OperationResult(IDENTITY, operation, "history_1", "completed", 3)
    with pytest.raises(ValueError):
        replace(result, **options)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: OperationRequest(IDENTITY, "step", datum_count=1),
        lambda: OperationRequest(IDENTITY, "forward", optim={"lr": 0.1}),
        lambda: OperationRequest(IDENTITY, "forward_backward", grad_scale=1),
        lambda: OperationRequest(
            IDENTITY, "optim_step", loss_params={"temperature": 1}
        ),
        lambda: OperationRequest(IDENTITY, "optim_step", payload=PAYLOAD),
        lambda: OperationRequest(IDENTITY, "step", mode="evaluation"),
        lambda: OperationRequest(IDENTITY, "step", grad_scale=float("inf")),
        lambda: OperationRequest(IDENTITY, "step", dependencies=[IDENTITY.request_id]),
        lambda: replace(IDENTITY, writer_epoch=0),
        lambda: replace(PAYLOAD, id="../../server/file"),
        lambda: Accumulation(1, True, ["request_1"], 2),
        lambda: Failure("worker_lost", "Unknown effect", "unknown", True),
        lambda: Failure("output_failed", "Known completed effect", "completed", True),
        lambda: RequestStatus(IDENTITY, "accepted"),
        lambda: RequestStatus(IDENTITY, "uploading", Acceptance(IDENTITY, 3)),
        lambda: RequestStatus(IDENTITY, "completed", Acceptance(IDENTITY, 3)),
        lambda: RequestStatus(
            IDENTITY, "running", Acceptance(replace(IDENTITY, request_id="other"), 3)
        ),
        lambda: replace(result(), grad_scale=None),
        lambda: replace(result(), loss=float("nan")),
        lambda: replace(result(), operation="optim_step", loss=1),
    ],
)
def test_invalid_controls_identities_and_ambiguous_outcomes_are_rejected(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_objective_loss_and_custom_metric_named_loss_remain_distinct():
    value = replace(result(), loss=0.75, metrics={"loss": 0.25})
    restored = decode(encode(value))
    assert restored.loss == 0.75 and restored.metrics["loss"] == 0.25


def test_decoder_rejects_unknown_versions_types_fields_and_duplicate_keys():
    for patch, match in (
        ({"version": 2}, "unsupported protocol version"),
        ({"version": True}, "unsupported protocol version"),
        ({"kind": "os.system"}, "unknown protocol record"),
        ({"value": {"definition": "x", "unknown": 1}}, "unknown fields"),
    ):
        encoded = json.loads(encode(Model("x")))
        encoded.update(patch)
        with pytest.raises(ProtocolError, match=match):
            decode(json.dumps(encoded).encode())
    with pytest.raises(ProtocolError, match="duplicate JSON field"):
        decode(b'{"version":1,"version":1,"kind":"config.model","value":{}}')
    with pytest.raises(ProtocolError, match="nonfinite"):
        decode(
            b'{"version":1,"kind":"config.model","value":{"definition":"x","config":{"x":NaN}}}'
        )
    with pytest.raises(ProtocolError, match="unregistered"):
        encode(Datum(inputs={"pixels": [[1, 2]]}))


def test_metadata_is_bounded_and_does_not_expand_binary_payloads():
    request = OperationRequest(
        IDENTITY, "step", replace(PAYLOAD, size_bytes=2**40), 1000000
    )
    assert len(encode(request)) < 1000
    with pytest.raises(ProtocolError, match="exceeds 10 bytes"):
        decode(encode(request), max_bytes=10)
    with pytest.raises(ProtocolError, match="exceeds 10 bytes"):
        encode(request, max_bytes=10)
    nested = (
        b'{"version":1,"kind":"config.model","value":'
        + b"[" * 70
        + b"0"
        + b"]" * 70
        + b"}"
    )
    with pytest.raises(ProtocolError, match="nesting"):
        decode(nested)


@pytest.mark.parametrize("item", [b"{}", b"[]", b"0", b'"x"'])
def test_metadata_node_bound_precedes_json_object_allocation(item, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized graph reached JSON allocation")

    monkeypatch.setattr(json, "loads", forbidden)
    wire = b"[" + b",".join([item] * MAX_METADATA_NODES) + b"]"
    assert len(wire) < 1 << 20
    with pytest.raises(ProtocolError, match="nodes"):
        decode(wire)


def test_structure_limits_preserve_escaped_strings_and_canonical_roundtrip():
    value = Model("custom", {"text": '\\"[{,:}]\\\\' * 1000})
    assert decode(encode(value)) == value
    with pytest.raises(ProtocolError, match="nodes"):
        encode(Model("custom", {"items": [[] for _ in range(MAX_METADATA_NODES)]}))


def test_bad_record_does_not_retain_decoder_objects_in_exception_context(monkeypatch):
    class Payload:
        pass

    retained = []

    def invalid(cls, data):
        temporary = Payload()
        retained.append(weakref.ref(temporary))
        raise ValueError("unusable metadata")

    wire = encode(Model("custom"))
    monkeypatch.setattr(Model, "from_dict", classmethod(invalid))
    with pytest.raises(ProtocolError) as error:
        decode(wire)
    gc.collect()
    assert error.value.__context__ is None and error.value.__cause__ is None
    assert retained[0]() is None
    trace = error.value.__traceback__
    while trace:
        if trace.tb_frame.f_code.co_name == "decode":
            assert trace.tb_frame.f_locals["data"] == b""
            if "envelope" in trace.tb_frame.f_locals:
                assert trace.tb_frame.f_locals["envelope"] is None
        trace = trace.tb_next


def test_private_codec_owns_its_allowlist_and_uses_the_same_strict_envelope():
    kinds = {"private.request": OperationRequest}
    codec = RecordCodec(kinds)
    kinds.clear()
    request = OperationRequest(IDENTITY, "forward")
    encoded = codec.encode(request)
    assert codec.decode(encoded) == request
    assert encoded == encode(request).replace(b"compute.request", b"private.request")
    with pytest.raises(ProtocolError, match="unknown protocol record"):
        decode(encoded)
    with pytest.raises(ProtocolError, match="unknown protocol record"):
        codec.decode(encode(request))
    with pytest.raises(ProtocolError, match="unregistered"):
        codec.encode(Model("x"))
    with pytest.raises(ProtocolError, match="exceeds 10"):
        codec.decode(encoded, max_bytes=10)
    with pytest.raises(ProtocolError, match="duplicate JSON"):
        codec.decode(encoded.replace(b'"version":1', b'"version":1,"version":1'))


@pytest.mark.parametrize(
    "kinds",
    [{}, {"": Model}, {"bad": object}, {"a": Model, "b": Model}],
)
def test_codec_registries_must_be_nonempty_unambiguous_record_classes(kinds):
    with pytest.raises((TypeError, ValueError)):
        RecordCodec(kinds)
