"""Supplied definitions use the public registry, like external integrations.

Loading the registry imports no tensor framework and constructs no models.
Each model family owns its registration hook; objectives and policies are shared.
"""

from importlib import import_module


def register(registry):
    from shadowspill_workloads.models.families import families

    from .moe_objective import register as register_moe_objective
    from .pretrained import register as register_pretrained
    from .text_policies import register as register_text_policies
    from .token import register as register_token

    register_token(registry)
    register_moe_objective(registry)
    for family in families():
        if family.training is not None:
            module, name = family.training.split(":")
            getattr(import_module(module), name)(registry)
    register_pretrained(registry)
    register_text_policies(registry)
