"""Archive metadata and complete geometry are bounded before state mutation."""

import io
import json
import os
import tarfile

import pytest
import torch

from engine_tests.checkpoint.archive_helpers import write_contents
from engine_tests.execution.test_checkpoint import MemoryExecutor
from shadowspill.checkpoint import inspection, load_checkpoint, save_checkpoint
from shadowspill.checkpoint.archive import header, read_manifest
from shadowspill.checkpoint.inspection import check_headers, open_checkpoint
from shadowspill.checkpoint.load import validate_checkpoint


def contents(path):
    with tarfile.open(path, "r:") as archive:
        return {name: archive.extractfile(name).read() for name in archive.getnames()}


def test_preflight_reads_descriptors_without_exporting_or_initializing(
    tmp_path, monkeypatch
):
    value = MemoryExecutor()
    path = tmp_path / "rank.ssp"
    save_checkpoint(value, path, metadata={"cursor": [1, 2]})

    def uncalled(*args, **kwargs):
        raise AssertionError("preflight touched tensor data")

    monkeypatch.setattr(value, "initialize", uncalled)
    monkeypatch.setattr(value, "export_tensors", uncalled)
    assert validate_checkpoint(value, path)["metadata"] == {"cursor": [1, 2]}
    with open_checkpoint(path) as archive:
        assert read_manifest(archive)["metadata"] == {"cursor": [1, 2]}


@pytest.mark.parametrize(
    "kind",
    [
        tarfile.XHDTYPE,
        tarfile.XGLTYPE,
        tarfile.GNUTYPE_LONGNAME,
        tarfile.GNUTYPE_LONGLINK,
        tarfile.GNUTYPE_SPARSE,
        tarfile.SYMTYPE,
    ],
)
def test_extensions_are_rejected_before_tarfile_parses_the_first_header(
    tmp_path, monkeypatch, kind
):
    info = tarfile.TarInfo("extension")
    info.type, info.size = kind, 1 << 40
    path = tmp_path / "extension.ssp"
    path.write_bytes(info.tobuf(format=tarfile.GNU_FORMAT))
    monkeypatch.setattr(
        tarfile, "open", lambda *a, **k: pytest.fail("unbounded TAR parser reached")
    )
    with pytest.raises(ValueError, match="archive members"), open_checkpoint(path):
        pass


def test_member_count_is_bounded_before_the_tar_directory_is_created(
    tmp_path, monkeypatch
):
    path = tmp_path / "rank.ssp"
    save_checkpoint(MemoryExecutor(), path)
    monkeypatch.setattr(inspection, "MEMBER_LIMIT", 2)
    monkeypatch.setattr(
        tarfile, "open", lambda *a, **k: pytest.fail("unbounded TAR directory reached")
    )
    with pytest.raises(ValueError, match="count bound"), open_checkpoint(path):
        pass


@pytest.mark.parametrize(
    "change", ["truncate", "missing_end", "hidden_data", "extra_padding"]
)
def test_payload_boundaries_and_end_padding_are_checked(tmp_path, change):
    path = tmp_path / "rank.ssp"
    save_checkpoint(MemoryExecutor(), path)
    original = path.read_bytes()
    data = {
        "truncate": original[:600],
        "missing_end": original[:-512],
        "hidden_data": original + b"not padding",
        "extra_padding": original + bytes(tarfile.RECORDSIZE),
    }[change]
    path.write_bytes(data)
    with (
        pytest.raises(ValueError, match=r"truncated|end padding"),
        open_checkpoint(path),
    ):
        pass


@pytest.mark.parametrize("metadata", [b'{"x":NaN}', b'{"x":1e9999}', b'{"x":1,"x":2}'])
def test_nonfinite_and_duplicate_json_are_rejected_before_loading(tmp_path, metadata):
    value = MemoryExecutor()
    path = tmp_path / "rank.ssp"
    save_checkpoint(value, path)
    rows = contents(path)
    rows["manifest.json"] = metadata
    write_contents(path, rows)
    with pytest.raises(ValueError, match=r"nonfinite|duplicate"):
        load_checkpoint(value, path)
    assert value.initialized


@pytest.mark.parametrize("kind", ["depth", "nodes"])
def test_json_growth_is_rejected_before_decoding(tmp_path, monkeypatch, kind):
    value = MemoryExecutor()
    path = tmp_path / "rank.ssp"
    save_checkpoint(value, path)
    rows = contents(path)
    if kind == "depth":
        rows["manifest.json"] = b'{"x":' + b"[" * 65 + b"0" + b"]" * 65 + b"}"
    else:
        rows["manifest.json"] = b'{"x":[' + b"{}," * 100 + b"{}]}"
        monkeypatch.setattr(inspection, "MANIFEST_NODES", 100)
    write_contents(path, rows)
    monkeypatch.setattr(
        json, "loads", lambda *a, **k: pytest.fail("unbounded JSON decoder reached")
    )
    with pytest.raises(ValueError, match=r"depth bound|node bound"):
        load_checkpoint(value, path)
    assert value.initialized


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "torch.float64"),
        ("storage_elements", True),
        ("blake3", "invalid"),
        ("unknown_region", 1),
    ],
)
def test_last_tensor_geometry_or_checksum_description_fails_before_any_mutation(
    tmp_path, field, value
):
    executor = MemoryExecutor()
    path = tmp_path / "rank.ssp"
    save_checkpoint(executor, path)
    rows = contents(path)
    manifest = json.loads(rows["manifest.json"])
    manifest["tensors"][-1][field] = value
    rows["manifest.json"] = json.dumps(manifest).encode()
    write_contents(path, rows)
    before = {key: tensor.clone() for key, tensor in executor.values.items()}
    for check in (validate_checkpoint, load_checkpoint):
        with pytest.raises(ValueError, match=r"geometry differs|checksum is invalid"):
            check(executor, path)
        assert executor.initialized
        for name, tensor in executor.values.items():
            torch.testing.assert_close(tensor.float(), before[name].float())


def test_large_tensor_size_inspection_skips_payload_without_allocating_it():
    size = (9 << 30) + 7
    entries = [(0, "tensors/0", size)]
    cursor = 512 + size + (-size % 512)
    entries += [(cursor, "manifest.json", 2), (cursor + 1024, "manifest.blake3", 64)]
    end = cursor + 2048 + 1024
    segments = {}
    for offset, name, length in entries:
        buffer = io.BytesIO()
        header(buffer, name, length)
        segments[offset] = buffer.getvalue()

    class Sparse:
        position = 0
        read_bytes = 0

        def tell(self):
            return self.position

        def seek(self, offset, whence=os.SEEK_SET):
            self.position = (end if whence == os.SEEK_END else 0) + offset
            return self.position

        def read(self, count):
            assert 0 < count <= tarfile.RECORDSIZE
            data = bytearray(min(count, end - self.position))
            for start, value in segments.items():
                first, last = (
                    max(start, self.position),
                    min(start + len(value), self.position + len(data)),
                )
                if first < last:
                    data[first - self.position : last - self.position] = value[
                        first - start : last - start
                    ]
            self.position += len(data)
            self.read_bytes += len(data)
            return bytes(data)

    source = Sparse()
    check_headers(source)
    assert source.tell() == 0 and source.read_bytes == 5 * 512
