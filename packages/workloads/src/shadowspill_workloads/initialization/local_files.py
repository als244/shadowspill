"""Content-pinned local checkpoints, with bounded reads and no source writes.

First resolution streams checksums without constructing tensors. Resolved setup
retains that manifest, so rechecks read only bounded metadata. Admitted workers
verify every payload before opening the existing checkpoint reader.
"""

import hashlib
import json
import os
import stat
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from . import headers, weight_files
from .hub import read_config


def accepts(reference):
    return isinstance(reference, str) and Path(reference).is_absolute()


def revision(manifest):
    if any(len(file.checksum) != 64 for file in manifest):
        raise ValueError("local checkpoint content pins require SHA256 checksums")
    value = [asdict(file) for file in sorted(manifest, key=lambda file: file.name)]
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "local-sha256:" + hashlib.sha256(payload).hexdigest()


def _facts(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _file(directory, name):
    facts = (directory / name).stat()
    if not stat.S_ISREG(facts.st_mode):
        raise ValueError(f"local checkpoint requires a regular file: {name}")
    return weight_files.WeightFile(name, facts.st_size, "0" * 64)


def _checksum(directory, file):
    path = directory / file.name
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != file.size:
            raise ValueError(f"local checkpoint changed while pinning: {file.name}")
        remaining = file.size
        while remaining:
            data = stream.read(min(weight_files.CHUNK_BYTES, remaining))
            if not data:
                raise ValueError(f"truncated local checkpoint: {file.name}")
            digest.update(data)
            remaining -= len(data)
        if stream.read(1) or _facts(os.fstat(stream.fileno())) != _facts(before):
            raise ValueError(f"local checkpoint changed while pinning: {file.name}")
        if _facts(path.stat()) != _facts(before):
            raise ValueError(
                f"local checkpoint was replaced while pinning: {file.name}"
            )
    return replace(file, checksum=digest.hexdigest())


def describe(reference, known=None):
    """Discover bounded safetensors metadata and pin exact file content."""
    if not accepts(reference):
        raise ValueError("local checkpoint reference must be an absolute server path")
    directory = Path(reference).resolve(strict=True)
    names = ["config.json"]
    if (directory / weight_files.INDEX).exists():
        names.append(weight_files.INDEX)
        names.extend(weight_files.read_index(directory / weight_files.INDEX))
    else:
        names.append("model.safetensors")
    inventory = tuple(_file(directory, name) for name in sorted(names))
    # Check metadata bounds before any tensor-payload scan or reader construction.
    headers.validate(directory, inventory)
    if known is None:
        manifest = tuple(_checksum(directory, file) for file in inventory)
    else:
        manifest = tuple(sorted(weight_files.files(known), key=lambda file: file.name))
        if [(f.name, f.size) for f in manifest] != [
            (f.name, f.size) for f in inventory
        ]:
            raise ValueError("local checkpoint files differ from the pinned manifest")
    metadata = read_config(directory / "config.json")
    for file in manifest:
        if file.name in ("config.json", weight_files.INDEX):
            weight_files.verify(directory / file.name, file)
    return str(directory), metadata, manifest


def verify(reference, manifest):
    directory = Path(reference)
    paths = {
        file.name: (directory / file.name).resolve(strict=True) for file in manifest
    }
    before = {name: _facts(path.stat()) for name, path in paths.items()}
    for file in manifest:
        if not stat.S_ISREG(paths[file.name].stat().st_mode):
            raise ValueError(f"local checkpoint requires a regular file: {file.name}")
        weight_files.verify(paths[file.name], file)
    if any(_facts(paths[name].stat()) != facts for name, facts in before.items()):
        raise ValueError("local checkpoint changed during verification")
    return paths, before


def disk_bytes(manifest):
    # Only directory entries and links to verified external files, no payload copy.
    return (8 << 20) + sum(
        (len(PurePosixPath(file.name).parts) + 3) << 12 for file in manifest
    )


@contextmanager
def snapshot(reference, manifest, runtime):
    if runtime.disk_bytes < disk_bytes(manifest):
        raise ValueError("local checkpoint links exceed admitted initialization disk")
    paths, before = verify(reference, manifest)
    with TemporaryDirectory(prefix="local-weights-", dir=runtime.cache_root) as root:
        directory = Path(root)
        for name, path in paths.items():
            target = directory.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(path)
        headers.validate(directory, manifest)
        # Readers see exactly the manifest, even if the source contains unrelated
        # checkpoints. Source files must stay unchanged while initialization runs.
        yield directory
        try:
            # Filesystem timestamps can be coarser than two consecutive writes.
            # Recheck content before publishing initialized state, not just stat.
            for file in manifest:
                weight_files.verify(paths[file.name], file)
            if any(
                _facts(paths[name].stat()) != facts for name, facts in before.items()
            ):
                raise ValueError("file identity changed")
        except ValueError as error:
            raise ValueError(
                "local checkpoint changed during initialization"
            ) from error
