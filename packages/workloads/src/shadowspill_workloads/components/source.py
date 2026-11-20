"""Installed component provenance, including optional providers without imports."""

from functools import cache
from hashlib import file_digest, sha256
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

from shadowspill_client.protocol.components import ComponentSource


def _files(root):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in (".py", ".so", ".pyd", ".dll", ".dylib")
    )


def _add(digest, name, path):
    digest.update(name.encode() + b"\0")
    with path.open("rb") as stream:
        digest.update(file_digest(stream, "sha256").digest())


@cache
def _provider_digest(name):
    # A running controller/worker treats its installed providers as immutable,
    # like the engine's setup_sources(). Large extension binaries are read once.
    digest = sha256(name.encode() + b"\0")
    spec = find_spec(name)
    if spec is None:
        digest.update(b"absent")
    elif spec.submodule_search_locations:
        for index, directory in enumerate(spec.submodule_search_locations):
            root = Path(directory)
            for path in _files(root):
                _add(digest, f"{index}/{path.relative_to(root).as_posix()}", path)
    elif spec.origin is not None:
        _add(digest, name, Path(spec.origin))
    else:
        raise ValueError(f"cannot pin provider source: {name}")
    return digest.digest()


def component_source(paths, *, providers=()):
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    files = {Path(__file__).resolve()}
    for relative in paths:
        path = root / relative
        files.update(_files(path) if path.is_dir() else [path])
    for path in sorted(files):
        _add(digest, path.relative_to(root).as_posix(), path)
    for name in sorted(providers):
        digest.update(_provider_digest(name))
    return ComponentSource(
        "shadowspill-workloads", version("shadowspill-workloads"), digest.hexdigest()
    )
