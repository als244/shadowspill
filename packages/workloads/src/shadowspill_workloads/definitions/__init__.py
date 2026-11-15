"""Supplied definitions use the public registry, like external integrations.

Regression and token objective definitions are registered during the refactor.
Catalog text/media model definitions migrate as their preparation paths land.
Importing this module does not construct a model or import a tensor framework.
"""


def register(registry):
    from .regression import register as register_regression
    from .token import register as register_token

    register_regression(registry)
    register_token(registry)
