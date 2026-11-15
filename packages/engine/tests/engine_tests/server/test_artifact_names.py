"""Names retain immutable content without copying or changing another artifact kind."""

import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from engine_tests.server.test_artifacts import reference, store
from shadowspill.orchestration.launch.publication import OutputSpec
from shadowspill.server.records.artifact_index import ArtifactAlias


def publish(artifacts):
    return artifacts.put(reference(), lambda _: io.BytesIO(b"result"))


@pytest.mark.parametrize("kind", ["plans", "checkpoint"])
def test_unencodable_name_cannot_poison_durable_storage(tmp_path, kind):
    with store(tmp_path) as artifacts:
        ref = publish(artifacts)
        with pytest.raises(UnicodeEncodeError):
            artifacts.name(ref, ArtifactAlias("s", "r", "\ud800", ref.id, kind))
        artifacts.name(ref, ArtifactAlias("s", "r", "valid", ref.id, kind))
    with store(tmp_path) as recovered:
        assert recovered.resolve_alias("s", "r", "valid", kind=kind) == ref


def test_names_are_scoped_idempotent_and_durable_without_payload_copy(tmp_path):
    with store(tmp_path) as artifacts:
        ref = publish(artifacts)
        initial = artifacts.used
        alias = ArtifactAlias("session", "run", "latest / résumé", ref.id, kind="plans")
        assert artifacts.name(ref, alias) == ref
        count = len(tuple(artifacts._journal.records()))
        assert artifacts.name(ref, alias) == ref
        assert len(tuple(artifacts._journal.records())) == count
        assert artifacts.used == initial and artifacts._disk.used == (0, 0)
        assert (
            artifacts.resolve_alias("session", "run", alias.name, kind="plans") == ref
        )
        assert artifacts.resolve_alias("session", "run", alias.name) is None
        assert (
            artifacts.resolve_alias("session", "other", alias.name, kind="plans")
            is None
        )
        assert artifacts.resolve_alias("other", "run", alias.name, kind="plans") is None
        # Existing checkpoint records omit the kind and retain their meaning.
        checkpoint = ArtifactAlias.from_dict(
            {
                "session_id": "session",
                "run_id": "run",
                "name": alias.name,
                "artifact_id": ref.id,
            }
        )
        assert checkpoint.kind == "checkpoint"
        artifacts.name(ref, checkpoint)
    with store(tmp_path) as recovered:
        assert (
            recovered.resolve_alias("session", "run", alias.name, kind="plans") == ref
        )
        assert recovered.resolve_alias("session", "run", alias.name) == ref
        assert recovered.used == initial and recovered._index.alias_count() == 2


@pytest.mark.parametrize("fault", ["absent", "reference", "target", "missing_file"])
def test_name_requires_exact_owned_committed_content(tmp_path, fault):
    with store(tmp_path) as artifacts:
        ref = reference()
        if fault != "absent":
            publish(artifacts)
        alias = ArtifactAlias("s", "r", "name", ref.id, kind="plans")
        if fault == "reference":
            ref = replace(ref, sha256="f" * 64)
        elif fault == "target":
            alias = replace(alias, artifact_id="unknown")
        elif fault == "missing_file":
            artifacts._path(ref).unlink()
        with pytest.raises((ValueError, FileNotFoundError)):
            artifacts.name(ref, alias)
        assert artifacts._index.alias_count() == 0


@pytest.mark.parametrize("fault", ["acknowledgement", "index"])
def test_uncertain_name_recovers_from_the_existing_journal(
    tmp_path, monkeypatch, fault
):
    artifacts = store(tmp_path)
    ref = publish(artifacts)
    alias = ArtifactAlias("s", "r", "name", ref.id, kind="plans")
    original = artifacts._journal.append

    def failed(*args):
        if fault == "acknowledgement":
            original(*args)
        raise OSError("name acknowledgement lost")

    monkeypatch.setattr(
        artifacts._journal if fault == "acknowledgement" else artifacts._index,
        "append" if fault == "acknowledgement" else "set_alias",
        failed,
    )
    with pytest.raises(OSError):
        artifacts.name(ref, alias)
    with pytest.raises(RuntimeError, match="reconcile"):
        artifacts.name(ref, alias)
    artifacts.close()
    with store(tmp_path) as recovered:
        assert recovered.resolve_alias("s", "r", "name", kind="plans") == ref
        assert recovered.name(ref, alias) == ref
        with recovered.open(ref) as stream:
            assert stream.read() == b"result"


def test_concurrent_new_names_respect_capacity_and_exact_retries(tmp_path):
    with store(tmp_path, max_artifacts=1) as artifacts, ThreadPoolExecutor(2) as pool:
        ref = publish(artifacts)

        def name(value):
            alias = ArtifactAlias("s", "r", value, ref.id, kind="plans")
            try:
                return artifacts.name(ref, alias), alias
            except ValueError as error:
                assert "capacity" in str(error)
                return None

        results = list(pool.map(name, ("one", "two")))
        successful = [item for item in results if item is not None]
        assert len(successful) == artifacts._index.alias_count() == 1
        assert artifacts.name(*successful[0]) == ref
    with store(tmp_path, max_artifacts=1) as recovered:
        assert recovered._index.alias_count() == 1


def test_alias_capacity_race_rejects_before_linking_a_staged_generation(tmp_path):
    with store(tmp_path, max_artifacts=3) as artifacts:
        ref = publish(artifacts)
        target = replace(ref, id="staged")
        alias = ArtifactAlias("s", "r", "checkpoint", target.id)

        def write(_, writer):
            writer.write(memoryview(b"result"))
            return target

        with artifacts.stage_group(
            (OutputSpec(target.id, target.size_bytes, target.media_type),),
            write,
            alias=alias,
        ) as staged:
            for name in ("one", "two", "three"):
                artifacts.name(ref, ArtifactAlias("s", "r", name, ref.id, kind="plans"))
            with pytest.raises(ValueError, match="capacity"):
                staged.commit()
        assert artifacts.get(target.id) is None and not artifacts._path(target).exists()
        assert artifacts.used == (ref.size_bytes, 1)
        assert artifacts._index.alias_count() == 3
