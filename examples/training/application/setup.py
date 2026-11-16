"""Create or resume a ready client separately from submitting training calls."""

import json
from contextlib import contextmanager
from pathlib import Path

from shadowspill_client.protocol.requests import ArtifactRef

from .records import record


def read_resume(path, settings_type):
    with Path(path).open("rb") as stream:
        raw = stream.read((64 << 10) + 1)
    if len(raw) > 64 << 10:
        raise ValueError("resume record exceeds 64 KiB")
    value = json.loads(raw)
    settings = settings_type(**value["settings"])
    if value["application"] != settings.application:
        raise ValueError("resume record belongs to another application")
    return value, settings


def start(service, settings, directory, *, data, resume=None, preview=None):
    """Validate progress before acquiring workers; retain IDs before waiting.

    A caller may supply a metadata-only preview after resolving its objective or
    validating its input vocabulary. Otherwise the service resolves the request.
    Both paths use the same server-owned preparation and initialization engine.
    """
    cursor = 0
    if resume is None:
        if preview is None:
            preview = service.describe_training(**settings.request()).result()
        print(preview.summary(), flush=True)
        (directory / "setup.json").write_text(
            json.dumps(preview.config.to_dict(), indent=2) + "\n"
        )
        creation = service.create_training_client(config=preview.config)
    else:
        if preview is not None:
            raise ValueError("resume uses the checkpoint's saved Plans, not a preview")
        checkpoint = service.artifact(ArtifactRef.from_dict(resume["checkpoint"]))
        info = checkpoint.manifest().result().info
        metadata = info.to_dict()["metadata"]
        if (
            metadata.get("application") != settings.application
            or metadata.get("settings") != record(settings)
            or metadata.get("data") != data
            or metadata.get("next_update") != info.completed_updates
            or info.plans != ArtifactRef.from_dict(resume["plans"])
        ):
            raise ValueError("checkpoint settings, inputs or progress differ")
        cursor = info.completed_updates
        if not 0 <= cursor <= settings.steps:
            raise ValueError("checkpoint progress lies outside requested training")
        creation = service.create_training_client(
            plans=info.plans, checkpoint=checkpoint
        )
    (directory / "run.json").write_text(
        json.dumps(
            {
                "application": settings.application,
                "settings": record(settings),
                "session_id": creation.session_id,
                "run_id": creation.run_id,
            },
            indent=2,
        )
        + "\n"
    )
    training = creation.result()
    (directory / "plans.json").write_text(
        json.dumps(training.plans.reference.to_dict(), indent=2) + "\n"
    )
    return training, cursor


@contextmanager
def close_after(training):
    """Release workers without hiding an application failure or retrying a call."""
    try:
        yield training
    except BaseException as error:
        try:
            training.close().result()
        except BaseException as cleanup:
            error.add_note(f"Training close also failed: {cleanup}")
        raise
    else:
        training.close().result()
