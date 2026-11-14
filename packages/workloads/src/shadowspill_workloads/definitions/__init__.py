"""Supplied definitions use the public registry, like external integrations.

Only the small regression integration is registered here during the refactor.
Catalog text/media definitions migrate as their input and preparation paths land.
Importing this module does not construct a model or import a tensor framework.
"""


def register(registry):
    from .regression import register as register_regression

    register_regression(registry)
