"""Resolve preset overrides and expand complete build cases on CPU."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from itertools import product
from math import prod
from typing import Any, Literal

from shadowspill.planning.memory_search.algorithm import SearchOptions
from shadowspill.training.definitions.preparation import (
    PreparationContext,
    ResolvedPreparationDefinition,
)
from shadowspill_client import (
    BuildCase,
    BuildOptions,
    PlanningOptions,
    Policy,
    Preparation,
    ProfilingOptions,
)
from shadowspill_client._records import integer, plain
from shadowspill_client.protocol.preparation import ResolvedPreparation

type Origin = Literal["explicit", "preset", "engine_default"]


def _search(values: Mapping[str, Any] | None) -> dict[str, Any]:
    if values is None:
        return SearchOptions().to_dict()
    # Core decoding checks required keys; additionally reject ignored extras at
    # every record level before using it as an exact resolved configuration.
    restored = SearchOptions.from_dict(values, "preparation.planning.search")
    result = restored.to_dict()
    if plain(values) != result:
        raise ValueError(
            "search settings differ from the complete core SearchOptions record"
        )
    return result


def resolve_preparation(
    definition: ResolvedPreparationDefinition,
    context: PreparationContext,
    requested: Preparation | None = None,
    *,
    max_build_cases: int = 256,
) -> ResolvedPreparation:
    integer(max_build_cases, "max_build_cases", minimum=1)
    supplied = requested or Preparation()
    preset = definition.definition.defaults(replace(context, requested=supplied))
    if not isinstance(preset, Preparation):
        raise TypeError("installed preparation defaults must return Preparation")
    if supplied.preset is not None and supplied.preset != definition.definition.id:
        raise ValueError("requested preset differs from its resolved definition")
    if preset.preset is not None and preset.preset != definition.definition.id:
        raise ValueError("installed preset defaults name another definition")
    origins: dict[str, Origin] = {}
    defaults: dict[str, Any] = {
        "partition": Policy("shadowspill/partition@1", {"mode": "auto"}).to_dict(),
        "chunking": None,
        "graph_pair_policy": None,
    }
    fixed = dict(defaults)
    for name in fixed:
        origins[f"build.{name}"] = "engine_default"
    for source, settings in (("preset", preset.build), ("explicit", supplied.build)):
        fixed.update(settings.to_dict())
        for name in settings.values:
            origins[f"build.{name}"] = source  # type: ignore[assignment]

    requested_cases = supplied.candidates.cases
    preset_cases = preset.candidates.cases
    axes = ("microbatch_sizes", "chunking", "graph_pair_policies")
    if (
        preset_cases is not None
        and requested_cases is None
        and any(getattr(supplied.candidates, axis) is not None for axis in axes)
    ):
        raise ValueError("a coupled-case preset requires complete case overrides")
    complete = requested_cases if requested_cases is not None else preset_cases
    cases: list[BuildCase] = []
    if complete is not None:
        if len(complete) > max_build_cases:
            raise ValueError(
                f"build-case expansion exceeds configured limit {max_build_cases}"
            )
        origin: Origin = "explicit" if requested_cases is not None else "preset"
        origins["candidates.cases"] = origin
        for index, case in enumerate(complete):
            origins[f"cases.{index}.microbatch_size"] = origin
            # A fixed explicit choice overrides a preset case, while an explicit
            # complete case overrides a preset fixed choice. Client records have
            # already rejected overlapping choices within the explicit layer.
            merged = {**fixed, **case.build.to_dict()}
            if requested_cases is None:
                merged.update(supplied.build.to_dict())
            cases.append(replace(case, build=BuildOptions.from_dict(merged)))
            for name in defaults:
                origins[f"cases.{index}.{name}"] = (
                    "explicit"
                    if name in supplied.build.values
                    else origin
                    if name in case.build.values
                    else origins[f"build.{name}"]
                )
    else:
        choices: list[tuple[Any, ...]] = []
        for axis, option in (
            ("microbatch_sizes", None),
            ("chunking", "chunking"),
            ("graph_pair_policies", "graph_pair_policy"),
        ):
            explicit, inherited = (
                getattr(supplied.candidates, axis),
                getattr(preset.candidates, axis),
            )
            if explicit is not None:
                values, origin = tuple(explicit), "explicit"
            elif option is not None and option in supplied.build.values:
                values, origin = (fixed[option],), "explicit"
            elif inherited is not None:
                values, origin = tuple(inherited), "preset"
            elif option is not None:
                values, origin = (fixed[option],), origins[f"build.{option}"]
            else:
                raise ValueError("preset or request must specify microbatch sizes")
            origins[f"candidates.{axis}"] = origin
            choices.append(values)
        if prod(map(len, choices)) > max_build_cases:
            raise ValueError(
                f"build-case expansion exceeds configured limit {max_build_cases}"
            )
        for size, chunking, graph_pair in product(*choices):
            index = len(cases)
            origins[f"cases.{index}.microbatch_size"] = origins[
                "candidates.microbatch_sizes"
            ]
            origins[f"cases.{index}.partition"] = origins["build.partition"]
            origins[f"cases.{index}.chunking"] = origins["candidates.chunking"]
            origins[f"cases.{index}.graph_pair_policy"] = origins[
                "candidates.graph_pair_policies"
            ]
            cases.append(
                BuildCase(
                    size,
                    BuildOptions.from_dict(
                        {
                            **fixed,
                            "chunking": chunking,
                            "graph_pair_policy": graph_pair,
                        }
                    ),
                )
            )
    if any(case.build.values["partition"] is None for case in cases):
        raise ValueError(
            "partition requires an installed policy; None is not a partition policy"
        )
    if any(case in cases[:index] for index, case in enumerate(cases)):
        raise ValueError("resolved build cases collide after fixed overrides")

    planning: dict[str, Any] = {}
    for name, fallback in (
        ("microbatch_counts", (1,)),
        ("search", None),
        ("workers", 0),
    ):
        explicit, inherited = (
            getattr(supplied.planning, name),
            getattr(preset.planning, name),
        )
        value: Any
        value, origin = (
            (explicit, "explicit")
            if explicit is not None
            else (inherited, "preset")
            if inherited is not None
            else (fallback, "engine_default")
        )
        planning[name] = _search(value) if name == "search" else value
        origins[f"planning.{name}"] = origin
    profiling, origin = (
        (supplied.profiling, "explicit")
        if supplied.profiling is not None
        else (preset.profiling, "preset")
        if preset.profiling is not None
        else (ProfilingOptions(), "engine_default")
    )
    origins["profiling"] = origin
    headroom, origin = (
        (supplied.external_headroom_bytes, "explicit")
        if supplied.external_headroom_bytes is not None
        else (preset.external_headroom_bytes, "preset")
        if preset.external_headroom_bytes is not None
        else (512 << 20, "engine_default")
    )
    origins["external_headroom_bytes"] = origin
    # Reuse identities are caller input, never a server-local cache path hidden
    # in a default. A preset may only describe repeatable phase choices.
    if preset.reuse:
        raise ValueError("installed presets cannot supply artifact reuse identities")
    return ResolvedPreparation(
        definition.definition.id,
        definition.source,
        cases,
        profiling,
        PlanningOptions(**planning),
        origins,
        supplied.reuse,
        headroom,
    )
