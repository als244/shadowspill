"""Select trainable parameters once, preserving tied tensor identities."""

from collections.abc import Sequence
from fnmatch import fnmatchcase
from typing import Any

from shadowspill.parallelism.selectors import validate_selectors


def select_trainable(model: Any, patterns: Sequence[str]) -> None:
    """Apply tensor-name globs after model/adapter construction, before capture.

    A selected alias selects its shared parameter. Check every pattern before
    changing any flags; an empty selection explicitly freezes all parameters.
    """
    parameters = dict(model.named_parameters(remove_duplicate=False))
    validate_selectors(tuple(parameters), dict.fromkeys(patterns))
    selected = {
        id(parameter)
        for name, parameter in parameters.items()
        if any(fnmatchcase(name, pattern) for pattern in patterns)
    }
    for parameter in parameters.values():
        parameter.requires_grad_(id(parameter) in selected)
