"""Creation factories can recover complete choices after the HTTP body is gone."""

import io
from dataclasses import replace

import pytest

from client_tests.test_training_config import resolved
from engine_tests.server.test_artifacts import store
from engine_tests.server.test_jobs import configured, until
from engine_tests.server.test_storage import IDENTITY
from shadowspill.server.creation_records import CreationRecord, load_creation
from shadowspill.server.records.artifact_index import storage_key
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES
from shadowspill_client.protocol.creation import TrainingCreation
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.runs import RunInfo


def info(record):
    return RunInfo(
        "owner",
        "session",
        "run",
        "server",
        record.reference.sha256,
        record.configuration,
    )


def test_choices_reopen_under_the_catalog_identity_without_a_handler_closure(tmp_path):
    value = TrainingCreation(
        resolved(), plans=ArtifactRef("plans", "a" * 64, 100), wait_for_resources=True
    )
    record = CreationRecord.encode(value)
    with store(tmp_path) as artifacts:
        record.publish(artifacts)
        record.publish(artifacts)
        assert len(artifacts.page()) == 2
    with store(tmp_path) as artifacts:
        assert load_creation(artifacts, info(record)) == value
        with pytest.raises(PermissionError, match="another owner"):
            load_creation(artifacts, replace(info(record), owner_id="someone-else"))
        with pytest.raises(ValueError, match="configuration"):
            load_creation(
                artifacts,
                replace(
                    info(record),
                    configuration=replace(record.configuration, sha256="c" * 64),
                ),
            )


def test_choices_cannot_be_loaded_from_configuration_alone(tmp_path):
    record = CreationRecord.encode(TrainingCreation(resolved()))
    with store(tmp_path) as artifacts:
        artifacts.put(record.configuration, lambda _: io.BytesIO(record.config))
        with pytest.raises(ValueError, match="metadata is missing"):
            load_creation(artifacts, info(record))


@pytest.mark.parametrize("kind", ["request", "config"])
def test_post_publication_corruption_is_rejected_before_creation(tmp_path, kind):
    record = CreationRecord.encode(TrainingCreation(resolved()))
    with store(tmp_path) as artifacts:
        record.publish(artifacts)
        ref = record.reference if kind == "request" else record.configuration
        path = artifacts.directory / "data" / storage_key(ref.id)
        with open(path, "r+b") as stream:
            stream.write(b"x")
        with pytest.raises(ValueError, match="checksum"):
            load_creation(artifacts, info(record))


def test_metadata_has_a_fixed_byte_bound_even_when_constructed_directly():
    with pytest.raises(ValueError, match="byte bound"):
        CreationRecord(b"x" * (MAX_METADATA_BYTES + 1), b"config")


def test_admitted_job_loads_exact_published_choices_in_its_factory(
    tmp_path, monkeypatch
):
    value = TrainingCreation(resolved(), wait_for_resources=True)
    record = CreationRecord.encode(value)
    with configured(tmp_path) as ctx:
        seen, original = [], ctx.jobs._factory

        def factory(info):
            with ctx.service.artifacts(info.owner_id) as artifacts:
                seen.append(load_creation(artifacts, info))
            return original(info)

        monkeypatch.setattr(ctx.jobs, "_factory", factory)

        def publish():
            with ctx.service.artifacts("owner") as artifacts:
                record.publish(artifacts)

        info = ctx.jobs.create(
            IDENTITY,
            request_sha256=record.reference.sha256,
            configuration=record.configuration,
            publish=publish,
        )
        until(lambda: bool(seen))
        assert seen == [value] and info.configuration == record.configuration
        assert (
            ctx.service.catalog.get(IDENTITY).request_sha256 == record.reference.sha256
        )
