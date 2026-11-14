"""Framing and metadata must fail safely without touching runtime state."""

import io
import tarfile

import pytest

from engine_tests.execution.test_checkpoint import MemoryExecutor
from shadowspill import load_checkpoint, save_checkpoint
from shadowspill.checkpoint.archive import header


def test_tensor_header_supports_sizes_above_eight_gibibytes():
    stream = io.BytesIO()
    size = (9 << 30) + 7
    header(stream, "tensors/0", size)
    info = tarfile.TarInfo.frombuf(stream.getvalue(), "utf-8", "strict")
    assert info.name == "tensors/0" and info.size == size


def test_manifest_corruption_is_rejected_before_initialization(tmp_path):
    executor = MemoryExecutor()
    path = tmp_path / "checkpoint.ssp"
    save_checkpoint(executor, path, metadata={"updates": 7})
    with tarfile.open(path) as archive:
        offset = archive.getmember("manifest.json").offset_data
    with path.open("r+b") as stream:
        stream.seek(offset)
        stream.write(b"!")
    with pytest.raises(ValueError, match="manifest checksum"):
        load_checkpoint(executor, path)
    assert executor.initialized


@pytest.mark.parametrize("member", ["duplicate", "link"])
def test_nonordinary_or_duplicate_archive_members_are_rejected(tmp_path, member):
    executor = MemoryExecutor()
    path = tmp_path / "checkpoint.ssp"
    save_checkpoint(executor, path)
    with tarfile.open(path, "a:") as archive:
        info = tarfile.TarInfo("manifest.json" if member == "duplicate" else "link")
        if member == "link":
            info.type = tarfile.SYMTYPE
            info.linkname = "tensors/0"
        archive.addfile(info)
    with pytest.raises(ValueError, match="archive members"):
        load_checkpoint(executor, path)
    assert executor.initialized
