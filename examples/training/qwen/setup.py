"""Resolve text model and search choices using metadata-only service previews."""

from dataclasses import replace

from .settings import objective


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
