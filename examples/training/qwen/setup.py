"""Bind the checkpoint's balancing coefficient to the selected objective."""

from examples.training.text import preparation


def resolve(service, settings):
    return preparation.resolve(
        service,
        settings.request(),
        model=f"shadowspill/{settings.family}@1",
        defaults_from_model={"balance.weight": ("architecture", "router_aux_loss_coef")}
        if settings.balance_weight is None
        else None,
        resolution_options=settings.resolution_options,
        tail_save_counts=settings.tail_save_counts,
    )
