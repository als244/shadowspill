"""Refine portable text setup using metadata, before acquiring any workers."""

from dataclasses import replace


def resolve(
    service,
    request,
    *,
    definition=None,
    defaults_from_model=None,
    resolution_options=None,
    tail_save_counts=None,
):
    request = dict(request)
    preview = service.describe_training(**request).result()
    if definition is not None and preview.config.model.definition != definition:
        raise ValueError(f"this application requires model definition {definition}")
    defaults = dict(request["objective"].defaults)
    for name, path in (defaults_from_model or {}).items():
        value = preview.config.model.config
        for key in path:
            value = value[key]
        defaults[name] = value
    changed = defaults != request["objective"].defaults
    if changed:
        request["objective"] = replace(request["objective"], defaults=defaults)
    if resolution_options is not None or tail_save_counts is not None:
        search = preview.config.preparation.planning.to_dict()["search"]
        if search["algorithm"]["name"] != "pressurefit":
            raise ValueError("resolution/tail choices require PressureFit")
        for name, values in (
            ("resolution_options", resolution_options),
            ("tail_save_counts", tail_save_counts),
        ):
            if values is not None:
                search["algorithm"]["options"][name] = list(values)
        request["prepare"] = replace(
            request["prepare"],
            planning=replace(request["prepare"].planning, search=search),
        )
        changed = True
    if changed:
        # Retain the content pin rather than rereading checkpoint payloads.
        request["model"] = preview.config.model
        preview = service.describe_training(**request).result()
    return preview
