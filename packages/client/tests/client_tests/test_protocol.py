"""Request identity and truthful failure effects survive canonical wire records."""

import json
from dataclasses import replace

import pytest

from shadowspill_client import Datum, Model
from shadowspill_client.protocol.codec import ProtocolError, decode, digest, encode
from shadowspill_client.protocol.requests import (
    ArtifactRef,
    ComputeRequest,
    RequestIdentity,
)
from shadowspill_client.protocol.results import (
    Acceptance,
    Accumulation,
    ArtifactIssue,
    ComputeResult,
    Failure,
    RequestStatus,
)

IDENTITY = RequestIdentity("request_2", "run_1", 2, "writer_1", 1, "server_1")
PAYLOAD = ArtifactRef("payload_1", "a" * 64, 64)


def result():
    return ComputeResult(
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
    )


@pytest.mark.parametrize(
    "value",
    [
        ComputeRequest(IDENTITY, "step", PAYLOAD, 2, optim={"lr": 0.01}),
        ComputeRequest(IDENTITY, "forward_backward", PAYLOAD, 2),
        ComputeRequest(IDENTITY, "forward", PAYLOAD, 2, mode="training"),
        ComputeRequest(IDENTITY, "optim_step", grad_scale=-1),
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
    left = ComputeRequest(
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
    assert ComputeRequest(IDENTITY, "forward").mode == "evaluation"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ComputeRequest(IDENTITY, "step", datum_count=1),
        lambda: ComputeRequest(IDENTITY, "forward", optim={"lr": 0.1}),
        lambda: ComputeRequest(IDENTITY, "forward_backward", grad_scale=1),
        lambda: ComputeRequest(IDENTITY, "optim_step", loss_params={"temperature": 1}),
        lambda: ComputeRequest(IDENTITY, "optim_step", payload=PAYLOAD),
        lambda: ComputeRequest(IDENTITY, "step", mode="evaluation"),
        lambda: ComputeRequest(IDENTITY, "step", grad_scale=float("inf")),
        lambda: ComputeRequest(IDENTITY, "step", dependencies=[IDENTITY.request_id]),
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
    ],
)
def test_invalid_controls_identities_and_ambiguous_outcomes_are_rejected(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


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
    request = ComputeRequest(
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
