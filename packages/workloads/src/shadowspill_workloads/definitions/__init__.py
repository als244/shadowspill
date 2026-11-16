"""Supplied definitions use the public registry, like external integrations.

Regression, packed Llama and token objectives are registered during the refactor.
Other catalog text/media definitions migrate as their preparation paths land.
Importing this module does not construct a model or import a tensor framework.
"""


def register(registry):
    from .llama import register as register_llama
    from .moe_objective import register as register_moe_objective
    from .regression import register as register_regression
    from .token import register as register_token

    register_regression(registry)
    register_token(registry)
    register_moe_objective(registry)
    register_llama(registry)
