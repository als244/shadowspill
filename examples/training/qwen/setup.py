"""Resolve model/search settings and acquire workers before the training loop."""

import json
from dataclasses import asdict, replace
from pathlib import Path

from shadowspill_client.protocol.requests import ArtifactRef

from .settings import Settings, objective


def record(settings):
    return json.loads(json.dumps(asdict(settings)))


def read_resume(path):
    with Path(path).open("rb") as stream:
        raw = stream.read((64 << 10) + 1)
    if len(raw) > 64 << 10:
        raise ValueError("resume record exceeds 64 KiB")
    value = json.loads(raw)
    settings = Settings(**value["settings"])
    if value["application"] != settings.application:
        raise ValueError("resume record belongs to another application")
    return value, settings


def resolve(service, settings):
    request = settings.request()
    preview = service.describe_training(**request).result()
    if preview.config.model.definition != f"shadowspill/{settings.family}@1":
        raise ValueError(f"this application requires a {settings.family} model")
    # Keep the checkpoint's balancing coefficient unless explicitly overridden.
    # This metadata-only preview neither reserves workers nor loads model tensors.
    weight = settings.balance_weight
    if weight is None:
        weight = preview.config.model.config["architecture"]["router_aux_loss_coef"]
    changed = weight != request["objective"].defaults["balance.weight"]
    request["objective"] = objective(weight)
    if settings.resolution_options is not None or settings.tail_save_counts is not None:
        search = preview.config.preparation.planning.to_dict()["search"]
        if search["algorithm"]["name"] != "pressurefit":
            raise ValueError("resolution/tail choices require PressureFit")
        for name in ("resolution_options", "tail_save_counts"):
            values = getattr(settings, name)
            if values is not None:
                search["algorithm"]["options"][name] = list(values)
        request["prepare"] = replace(
            request["prepare"],
            planning=replace(request["prepare"].planning, search=search),
        )
        changed = True
    if changed:
        # Reuse the resolved content pin rather than rereading checkpoint payloads.
        request["model"] = preview.config.model
        preview = service.describe_training(**request).result()
    return preview


def data_identity(source, validation):
    return {
        "train": source.identity,
        "validation": None if validation is None else validation.identity,
    }


def start(service, settings, source, validation, directory, *, resume=None):
    cursor = 0
    if resume is None:
        preview = resolve(service, settings)
        if (
            max(source.vocab_size, 0 if validation is None else validation.vocab_size)
            > (preview.config.model.config["architecture"]["vocab_size"])
        ):
            raise ValueError("corpus vocabulary exceeds model vocabulary")
        print(preview.summary(), flush=True)
        (directory / "setup.json").write_text(
            json.dumps(preview.config.to_dict(), indent=2) + "\n"
        )
        creation = service.create_training_client(config=preview.config)
    else:
        checkpoint = service.artifact(ArtifactRef.from_dict(resume["checkpoint"]))
        info = checkpoint.manifest().result().info
        metadata = info.to_dict()["metadata"]
        if (
            metadata.get("application") != settings.application
            or metadata["settings"] != record(settings)
            or metadata["data"] != data_identity(source, validation)
            or metadata["next_update"] != info.completed_updates
            or info.plans != ArtifactRef.from_dict(resume["plans"])
        ):
            raise ValueError("checkpoint settings, corpus or progress differ")
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
