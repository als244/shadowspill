"""CPU-only checkpoint architecture identification for supplied recipes."""

from shadowspill_workloads.models.families import checkpoint_families

CHECKPOINT_FAMILIES = {
    kind: family.name for kind, family in checkpoint_families().items()
}


def checkpoint_family(metadata):
    text = metadata.get("text_config", metadata)
    kind = text.get("model_type", metadata.get("model_type"))
    try:
        return CHECKPOINT_FAMILIES[kind]
    except KeyError as error:
        raise ValueError(f"unsupported checkpoint model_type: {kind!r}") from error
