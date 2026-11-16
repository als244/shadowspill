"""Bounded staging of compatible cases from a trusted measurement export."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

from shadowspill.artifacts._values import digest, encoded
from shadowspill_client._records import integer
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.training import ResolvedTraining

from .templates import TemplateInventory

REUSE_NAME = "reused.json"
METADATA_BYTES = 16 << 20
_STAGES = ("programs", "profiled", "calibrated", "communication")


def require_staged_reuse(root: Path, config: ResolvedTraining) -> None:
    """A reference alone cannot make a standalone phase silently ignore reuse."""
    if not config.preparation.reuse:
        return
    path = root / REUSE_NAME
    if not path.is_file() or path.stat().st_size > METADATA_BYTES:
        raise ValueError("preparation reuse must be staged by service storage first")
    value = json.loads(path.read_bytes())
    if (
        type(value.get("version")) is not int
        or value["version"] != 1
        or value.get("configuration") != digest(config.to_dict())
    ):
        raise ValueError("staged reuse belongs to another preparation configuration")


def stage_measurements(
    source: BinaryIO,
    reference: ArtifactRef,
    root: Path,
    inventory: TemplateInventory,
    *,
    context: Mapping[str, Any],
    max_disk_bytes: int,
    max_files: int,
) -> Mapping[str, Any]:
    """Copy original compatible files; phase workers still validate their target.

    The service must authenticate a completed preparation record before passing
    this stream. An uploaded archive is not sufficient provenance. Build/profile
    reuse follows exact capture identities and profiling controls. Calibration
    additionally requires the same complete inventory and physical assignment.
    The caller owns a fresh directory containing only its new preparation job,
    and the whole host/disk envelope through cleanup.
    """
    integer(max_disk_bytes, "reuse disk bytes", minimum=1)
    integer(max_files, "reuse file count", minimum=1)
    checksum = hashlib.sha256()
    size = 0
    while block := source.read(64 << 10):
        size += len(block)
        if size > reference.size_bytes:
            raise ValueError("measurement export exceeds its immutable reference")
        checksum.update(block)
    if size != reference.size_bytes or checksum.hexdigest() != reference.sha256:
        raise ValueError("measurement export failed its content checksum")
    source.seek(0)
    with zipfile.ZipFile(source) as archive:
        selected, calibration = _select(archive, inventory, context, max_files)
        requested = inventory.layouts.setup.config
        receipt = {
            "version": 1,
            "configuration": digest(requested.to_dict()),
            "source": reference.to_dict(),
            "files": selected,
            "calibration_reused": calibration,
        }
        body = encoded(receipt)
        if len(body) > METADATA_BYTES:
            raise ValueError("reuse manifest exceeds its metadata bound")
        existing = tuple(root.iterdir())
        if any(
            path.name not in {"preparation.ssp", ".preparation.lock"}
            or not path.is_file()
            or path.is_symlink()
            for path in existing
        ):
            raise ValueError("reuse requires a fresh owned preparation directory")
        directories = {Path(name).parent for name in selected}
        count = len(existing) + len(selected) + len(directories) + 1
        used = 4096 * count + sum(path.stat().st_size for path in existing)
        used += len(body) + sum(row["size_bytes"] for row in selected.values())
        if count > max_files or used > max_disk_bytes:
            raise ValueError(
                "measurement reuse exceeds the preparation storage envelope"
            )
        for name, description in selected.items():
            target = root / name
            target.parent.mkdir(exist_ok=True)
            checksum = hashlib.sha256()
            with archive.open(name) as stream, target.open("xb") as destination:
                while block := stream.read(64 << 10):
                    destination.write(block)
                    checksum.update(block)
            if checksum.hexdigest() != description["sha256"]:
                raise ValueError("measurement file failed its content checksum")
        # This completion marker is written only after every selected file agrees.
        with (root / REUSE_NAME).open("xb") as destination:
            destination.write(body)
    return receipt


def _select(
    archive: zipfile.ZipFile,
    inventory: TemplateInventory,
    context: Mapping[str, Any],
    max_files: int,
) -> tuple[dict[str, Any], bool]:
    entries = archive.infolist()
    by_name = {row.filename: row for row in entries}
    if len(entries) != len(by_name) or len(entries) > max_files:
        raise ValueError("measurement export has duplicate or excessive files")
    if (
        "manifest.json" not in by_name
        or by_name["manifest.json"].file_size > METADATA_BYTES
        or any(row.compress_type != zipfile.ZIP_STORED for row in entries)
    ):
        raise ValueError("measurement export has invalid framing or metadata")
    manifest = json.loads(archive.read("manifest.json"))
    if (
        manifest.get("format") != "shadowspill.preparation-export"
        or type(manifest.get("version")) is not int
        or manifest["version"] != 1
    ):
        raise ValueError("unsupported measurement export")
    original = ResolvedTraining.from_dict(manifest["configuration"])
    descriptions, captures = manifest["files"], manifest["captures"]
    if not isinstance(descriptions, dict) or not isinstance(captures, dict):
        raise ValueError("measurement export requires file/capture inventories")
    if set(by_name) != {"manifest.json", *descriptions}:
        raise ValueError("measurement export files differ from its manifest")
    for key in captures:
        if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ValueError("measurement export has an invalid capture identity")
    requested = inventory.layouts.setup.config
    common = inventory.templates.keys() & captures.keys()
    if not common:
        raise ValueError("measurement export has no compatible requested captures")
    profiles = original.preparation.profiling == requested.preparation.profiling
    calibration = (
        profiles
        and inventory.templates.keys() == captures.keys()
        and dict(context) == manifest.get("context")
        and all(set(_STAGES) <= captures[key]["files"].keys() for key in common)
    )
    stages = _STAGES if calibration else _STAGES[:2] if profiles else _STAGES[:1]
    selected = {}
    for key in sorted(common):
        for stage in stages:
            path = captures[key]["files"].get(stage)
            if path is None:
                # Retry failed cases; retain their old diagnostics in the source.
                continue
            if path != f"{stage}/{key}.ssp" or path not in descriptions:
                raise ValueError("measurement file is outside its declared capture")
            description, entry = descriptions[path], by_name[path]
            if (
                entry.is_dir()
                or type(description.get("size_bytes")) is not int
                or description["size_bytes"] != entry.file_size
                or re.fullmatch(r"[0-9a-f]{64}", description.get("sha256", "")) is None
            ):
                raise ValueError("measurement file differs from its manifest")
            selected[path] = description
    if not selected:
        raise ValueError("measurement export has no completed compatible files")
    return selected, calibration
