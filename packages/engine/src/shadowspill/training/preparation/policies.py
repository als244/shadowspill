"""Resolve portable policy references, then construct only at the build boundary."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any, cast

from shadowspill.training.definitions.policies import PolicyKind
from shadowspill.training.definitions.registry import Registry
from shadowspill_client import BuildOptions, Policy
from shadowspill_client.protocol.definitions import DefinitionSource
from shadowspill_client.protocol.preparation import ResolvedPreparation


def resolve_build_policies(
    registry: Registry,
    preparation: ResolvedPreparation,
    *,
    sources: Mapping[str, DefinitionSource] | None = None,
) -> tuple[ResolvedPreparation, Mapping[str, DefinitionSource]]:
    """Pin every installed policy, preserving explicit None and case ordering."""
    used: dict[str, DefinitionSource] = {}
    cases = []
    for case in preparation.cases:
        values: dict[str, Any] = {}
        for name, value in case.build.to_dict().items():
            if value is None:
                values[name] = None
                continue
            policy = Policy.from_dict(value)
            if sources is not None and policy.definition not in sources:
                raise ValueError(
                    f"saved preparation is missing policy source: {policy.definition}"
                )
            resolved = registry.resolve_policy(
                policy.definition,
                kind=cast(PolicyKind, name),
                config=policy.config,
                source=None if sources is None else sources[policy.definition],
            )
            if sources is not None and resolved.config != policy.config:
                raise ValueError("policy defaults changed from the saved configuration")
            used[policy.definition] = resolved.source
            values[name] = Policy(policy.definition, resolved.config).to_dict()
        cases.append(replace(case, build=BuildOptions.from_dict(values)))
    if sources is not None and sources.keys() != used.keys():
        raise ValueError("saved preparation contains unreferenced policy sources")
    return replace(preparation, cases=cases), MappingProxyType(used)


def materialize_build_options(
    registry: Registry,
    options: BuildOptions,
    sources: Mapping[str, DefinitionSource],
) -> dict[str, Any]:
    """Use ordinary engine policy objects; no Python is decoded from requests."""
    if set(options.values) != {"partition", "chunking", "graph_pair_policy"}:
        raise ValueError("materialize only completely resolved build options")
    result: dict[str, Any] = {}
    for name, value in options.to_dict().items():
        if value is None:
            if name == "partition":
                raise ValueError("partition requires an installed policy")
            result[name] = None
            continue
        policy = Policy.from_dict(value)
        if policy.definition not in sources:
            raise ValueError(f"missing policy source: {policy.definition}")
        resolved = registry.resolve_policy(
            policy.definition,
            kind=cast(PolicyKind, name),
            config=policy.config,
            source=sources[policy.definition],
        )
        if resolved.config != policy.config:
            raise ValueError("policy configuration is not fully resolved")
        result[name] = resolved.create()
    return result
