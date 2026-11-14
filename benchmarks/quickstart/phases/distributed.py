"""Determine the worker topology from a quickstart factory's candidates."""

from shadowspill.resources.description import resource_groups

from .candidates import build_options


def preparation_size(setup):
    """DP compute is local; model-owned communicating resources require peers."""
    # The launcher chooses one worker group for every candidate in this build.
    for candidate in setup.get("candidates", {}).values():
        build_options(setup, candidate)
    options = setup.get("build_options", {})
    if not resource_groups(options.get("resources", {})):
        return 1
    return options["parallelism"].world_size
