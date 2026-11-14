"""The versions the NVIDIA environment prefers, from NVIDIA's own lock files
at the pinned revisions.

    python nvidia_versions.py prefer BRIDGE_LOCK AUTOMODEL_LOCK
        [--keep NAME ...] [--automodel NAME ...]
    python nvidia_versions.py departures PREFERRED RESOLVED

``prefer`` prints one ``name==version`` line per package, and a summary on
standard error, for the resolver to prefer: it departs from one only where a
requirement forces it. A package a lock pins from a package index gets that
version: Bridge's lock first, AutoModel's for a package Bridge's does not pin
or AutoModel's own requirements exclude, and AutoModel's for every
``--automodel`` package (one the resolver departed from with Bridge's). A
package a lock pins more than once is taken in the version whose resolution
markers hold for this interpreter, and left open when none or several do.
What the running interpreter's environment already holds under a requirement
a lock's version would break keeps that requirement (PyTorch's own pins:
triton and its CUDA libraries), as does every ``--keep`` package. A package
either lock takes from git or a checkout (Transformer Engine from Bridge's
pinned revision) is pinned by that source, not here, even where the other
lock takes it from an index.

``departures`` prints, one per line, ``name preferred resolved`` for each
preferred package the resolution put at another version.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
import tomllib

from packaging.markers import Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def holds(marker: str | None, extras: tuple[str, ...] = ("",)) -> bool:
    """Whether a marker holds here for any of ``extras``."""

    if not marker:
        return True
    parsed = Marker(marker)
    return any(parsed.evaluate({"extra": extra}) for extra in extras)


def locked_versions(path: str) -> tuple[dict[str, str], list[str], set[str]]:
    """Each index-pinned package's version, the names left open, and the
    names the lock takes from git or a checkout."""

    with open(path, "rb") as handle:
        lock = tomllib.load(handle)
    entries: dict[str, list[dict]] = {}
    sourced = set()
    for package in lock.get("package", []):
        if "registry" in package.get("source", {}):
            entries.setdefault(normalized(package["name"]), []).append(package)
        else:
            sourced.add(normalized(package["name"]))
    versions, open_names = {}, []
    for name, candidates in entries.items():
        if len(candidates) > 1:
            candidates = [
                package
                for package in candidates
                if any(
                    holds(marker) for marker in package.get("resolution-markers", [])
                )
            ]
        if len(candidates) == 1:
            versions[name] = candidates[0]["version"]
        else:
            open_names.append(name)
    return versions, sorted(open_names), sourced


def member_requirements(path: str, member: str) -> dict[str, list[SpecifierSet]]:
    """What a workspace member of a lock requires, by package."""

    with open(path, "rb") as handle:
        lock = tomllib.load(handle)
    requirements: dict[str, list[SpecifierSet]] = {}
    for package in lock.get("package", []):
        if normalized(package["name"]) != member:
            continue
        for entry in package.get("metadata", {}).get("requires-dist", []):
            if "specifier" in entry and holds(
                entry.get("marker"), ("", entry.get("extra", ""))
            ):
                requirements.setdefault(normalized(entry["name"]), []).append(
                    SpecifierSet(entry["specifier"])
                )
    return requirements


def applies(requirement: Requirement, extras: set[str]) -> bool:
    marker = requirement.marker
    return marker is None or any(marker.evaluate({"extra": extra}) for extra in extras)


def installed_requirements() -> tuple[dict[str, str], dict[str, list[Requirement]]]:
    """Each installed package's version, and the requirements it holds: its
    own, and those of the extras some installed package asks of it."""

    distributions = {
        normalized(distribution.metadata["Name"]): distribution
        for distribution in importlib.metadata.distributions()
    }
    parsed = {}
    for name, distribution in distributions.items():
        parsed[name] = []
        for text in distribution.requires or ():
            try:
                parsed[name].append(Requirement(text))
            except InvalidRequirement:
                continue
    extras = {name: {""} for name in distributions}
    changed = True
    while changed:
        changed = False
        for name, requirements in parsed.items():
            for requirement in requirements:
                target = normalized(requirement.name)
                wanted = set(requirement.extras)
                if (
                    target in extras
                    and applies(requirement, extras[name])
                    and not wanted <= extras[target]
                ):
                    extras[target] |= wanted
                    changed = True
    held = {
        name: [
            requirement
            for requirement in parsed[name]
            if applies(requirement, extras[name])
        ]
        for name in distributions
    }
    return {
        name: distribution.version for name, distribution in distributions.items()
    }, held


def allows(specifiers: list[SpecifierSet], version: str) -> bool:
    # A pre-release a lock pins is a version like any other here.
    return all(
        specifier.contains(Version(version), prereleases=True)
        for specifier in specifiers
        if specifier
    )


def preferences(
    bridge_lock: str, automodel_lock: str, keep: set[str], automodel_first: set[str]
):
    bridge, bridge_open, bridge_sourced = locked_versions(bridge_lock)
    automodel, automodel_open, automodel_sourced = locked_versions(automodel_lock)
    # A package either project takes from its own source is pinned there.
    keep = keep | bridge_sourced | automodel_sourced
    automodel_requires = member_requirements(automodel_lock, "nemo-automodel")
    proposed = {}
    for name in sorted(set(bridge) | set(automodel)):
        if name in keep:
            continue
        if (
            name in bridge
            and name not in automodel_first
            and allows(automodel_requires.get(name, []), bridge[name])
        ):
            proposed[name] = (bridge[name], "Bridge")
        elif name in automodel:
            proposed[name] = (automodel[name], "AutoModel")
    # A package that keeps its installed version keeps its requirements too;
    # a lock's version that breaks one stays as installed, which can make
    # that package's requirements binding in turn.
    installed, requirements = installed_requirements()
    held: dict[str, str] = {}
    while True:
        staying = [
            name
            for name, version in installed.items()
            if name in keep
            or name in held
            or name not in proposed
            or Version(proposed[name][0]) == Version(version)
        ]
        breaking: dict[str, list[str]] = {}
        for owner in staying:
            for requirement in requirements[owner]:
                target = normalized(requirement.name)
                if (
                    target in proposed
                    and target not in held
                    and not allows([requirement.specifier], proposed[target][0])
                ):
                    breaking.setdefault(target, []).append(
                        f"{owner} {requirement.specifier}"
                    )
        if not breaking:
            break
        for target, reasons in breaking.items():
            held[target] = (
                f"{proposed[target][0]} (kept by {', '.join(sorted(set(reasons)))})"
            )
    chosen = {
        name: version for name, (version, _) in proposed.items() if name not in held
    }
    sources = {"Bridge": 0, "AutoModel": 0}
    for name, (_, source) in proposed.items():
        if name not in held:
            sources[source] += 1
    left_open = sorted((set(bridge_open) | set(automodel_open)) - set(proposed) - keep)
    return chosen, held, sources, left_open


def pins(path: str) -> dict[str, str]:
    """The ``name==version`` lines of a requirements file, by package."""

    found = {}
    with open(path) as handle:
        for line in handle:
            match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;#]+)", line)
            if match:
                found[normalized(match.group(1))] = match.group(2)
    return found


def departures(preferred: str, resolved: str) -> list[tuple[str, str, str]]:
    wanted, got = pins(preferred), pins(resolved)
    return [
        (name, version, got[name])
        for name, version in sorted(wanted.items())
        if name in got and Version(got[name]) != Version(version)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prefer = commands.add_parser("prefer")
    prefer.add_argument("bridge_lock")
    prefer.add_argument("automodel_lock")
    prefer.add_argument("--keep", action="append", default=[], metavar="NAME")
    prefer.add_argument("--automodel", action="append", default=[], metavar="NAME")
    compare = commands.add_parser("departures")
    compare.add_argument("preferred")
    compare.add_argument("resolved")
    args = parser.parse_args(argv)
    if args.command == "departures":
        for name, wanted, got in departures(args.preferred, args.resolved):
            print(name, wanted, got)
        return
    chosen, held, sources, left_open = preferences(
        args.bridge_lock,
        args.automodel_lock,
        {normalized(name) for name in args.keep},
        {normalized(name) for name in args.automodel},
    )
    for name, version in chosen.items():
        print(f"{name}=={version}")
    print(
        f"{len(chosen)} versions preferred from NVIDIA's locks ({sources['Bridge']} "
        f"Bridge's, {sources['AutoModel']} AutoModel's); {len(held)} kept by the "
        f"environment's own requirements; {len(left_open)} locked more than once "
        "and left open",
        file=sys.stderr,
    )
    for name, reason in held.items():
        print(f"  kept: {name} {reason}", file=sys.stderr)
    if left_open:
        print(f"  open: {', '.join(left_open)}", file=sys.stderr)


if __name__ == "__main__":
    main()
