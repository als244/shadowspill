"""Source-owned guarantees for opaque code, never inferred from timing samples."""

from collections.abc import Mapping
from types import MappingProxyType


def allocation_invariants(values: Mapping[str, str]) -> Mapping[str, str]:
    """Validate exact call/factory names and versioned, provider-owned contracts.

    Each declaration guarantees identical allocator event geometry, output
    aliases and retained storage for all tensor values and reachable provider
    state at the captured tensor geometry, non-tensor arguments and resources.
    This includes tensor-valued controls. Allocation must use the observed
    allocator; hidden or asynchronous allocations are not covered. A preparation
    factory's declaration also covers bound-task activation/wrapping.

    A declaration is an installed author's semantic guarantee, like padding
    safety, not an empirical conclusion. Value-dependent allocation requires a
    separate maximum-bound mechanism; it cannot use this declaration.
    """
    if not isinstance(values, Mapping):
        raise TypeError("allocation invariants must be a mapping")
    result = {}
    for name, contract in values.items():
        if (
            not isinstance(name, str)
            or not name
            or name.count(":") > 1
            or not all(
                part.isidentifier() for part in name.replace(":", ".").split(".")
            )
        ):
            raise ValueError("allocation invariants require exact call/factory names")
        if not isinstance(contract, str):
            raise ValueError("allocation invariant must name a versioned contract")
        owner, separator, version = contract.rpartition("@")
        if (
            not owner
            or not separator
            or not version
            or "@" in owner
            or any(character.isspace() for character in contract)
        ):
            raise ValueError("allocation invariant must name a versioned contract")
        result[name] = contract
    return MappingProxyType(result)
