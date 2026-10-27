"""Expand matrices and identify cases by their resolved settings."""

from __future__ import annotations

import hashlib
import itertools
import json
import re

from baselines.common.results import varied
from baselines.report import DERIVED_GEOMETRY

CODES = {
    "tokens_per_microbatch": "mb",
    "recompute": "rc",
    "activation_offload": "act",
    "offload_policy": "op",
    "pin_memory": "pin",
    "forward_prefetch": "fpf",
    "backward_prefetch": "bpf",
    "reshard_after_forward": "raf",
    "reshard_after_backward": "rab",
    "gradient_sync": "gs",
    "compile_blocks": "cb",
    "float8": "f8",
    "optimizer": "opt",
    "offload_gradients": "og",
    "zero_stage": "z",
    "offload_param": "ofp",
    "offload_optimizer": "ofo",
    "deepcompile": "dc",
    "execution_gib": "ex",
    "spill_gib": "sp",
    "resolution_options": "res",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def expand(specification, world_size, settings_type):
    """``[(name, settings)]`` for a matrix: validated, refused if invalid,
    and deduplicated by what runs."""

    unknown = set(specification) - {"schema_version", "defaults", "grid", "cases"}
    if unknown or specification.get("schema_version") != 1:
        raise ValueError("expected schema_version=1 and defaults/grid or cases")
    defaults = specification.get("defaults", {})
    settings_type.from_dict(defaults)
    if "grid" in specification and "cases" in specification:
        raise ValueError("choose a Cartesian grid or an explicit cases list")
    if "cases" in specification:
        overrides = specification["cases"]
    else:
        grids = specification.get("grid", {})
        grids = grids if isinstance(grids, list) else [grids]
        if not grids or any(
            not isinstance(grid, dict)
            or any(
                not isinstance(values, list) or not values for values in grid.values()
            )
            for grid in grids
        ):
            raise ValueError("each grid axis must be a nonempty list")
        overrides = [
            dict(zip(grid, values, strict=True))
            for grid in grids
            for values in itertools.product(*grid.values())
        ]
    if not isinstance(overrides, list) or not overrides:
        raise ValueError("at least one case is required")
    unique, seen = [], set()
    for override in overrides:
        settings = settings_type.from_dict({**defaults, **override})
        settings.validate(world_size)
        identity = digest(settings.as_dict())
        if identity in seen:
            continue
        seen.add(identity)
        unique.append((identity, settings))
    values = {
        identity: flat(settings.describe(world_size)) for identity, settings in unique
    }
    order = list(CODES)
    knobs = sorted(
        (key for key in varied(values) if key not in DERIVED_GEOMETRY),
        key=lambda key: (order.index(key) if key in order else len(order), key),
    )
    names = [case_name(values[identity], knobs, identity[:8]) for identity, _ in unique]
    if len(set(names)) != len(names):
        raise ValueError("two cases share a name; their digests begin alike")
    return list(zip(names, (settings for _, settings in unique), strict=True))


def flat(description):
    """A case's geometry and configuration as one mapping."""

    return {**description["geometry"], **description["configuration"]}


def case_name(values, knobs, identity):
    """A case's folder name: each setting its matrix varies, as its code and
    value, then the digest of everything it runs."""

    parts = []
    for key in knobs:
        code, value = CODES.get(key, key), values.get(key)
        if isinstance(value, bool):
            parts.append(f"{code}-{'on' if value else 'off'}")
        elif isinstance(value, int):
            parts.append(f"{code}{value}")
        elif isinstance(value, float):
            parts.append(f"{code}{value:g}")
        else:
            text = value if isinstance(value, str) else json.dumps(value)
            parts.append(f"{code}-{re.sub(r'[^A-Za-z0-9.]+', '-', text).strip('-')}")
    return "_".join(parts or ["case"]) + "__" + identity


def keep_names(cases, output):
    """The cases with the names ``output`` already gives their digests: a
    case keeps its folder when a grown matrix varies other settings."""

    held = {
        path.name.rsplit("__", 1)[1]: path.name
        for path in (output / "cases").glob("*__*")
    }
    return [
        (held.get(name.rsplit("__", 1)[1], name), settings) for name, settings in cases
    ]


def comparable(held, manifest):
    """What differs between a manifest an output holds and a resume's that
    makes their rows incomparable: the engine, the world size, the
    interpreter, its version or its packages. The matrix may change, and so
    may the code, ShadowSpill's and mlops's sources included."""

    def environment(value):
        return {
            key: item for key, item in (value or {}).items() if key != "mlops_source"
        }

    differences = [
        key
        for key in ("engine", "world_size", "python")
        if held.get(key) != manifest[key]
    ]
    if environment(held.get("environment")) != environment(manifest["environment"]):
        differences.append("environment")
    return differences
