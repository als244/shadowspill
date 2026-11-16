"""Pinned weight-file manifests and bounded, private Hub downloads."""

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from .weight_files import (
    CHUNK_BYTES,
    INDEX,
    WeightFile,
    _digest,
    disk_bytes,
    read_index,
    verify,
)


def metadata(reference, revision, name):
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    info = get_hf_file_metadata(hf_hub_url(reference, name, revision=revision))
    if info.commit_hash != revision:
        raise ValueError("pretrained file differs from its pinned Hub revision")
    return WeightFile(name, info.size, info.etag)


def download(reference, revision, file, destination):
    """Stream at most the pinned size; no Xet workers or full-file buffering."""
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

    digest, written = _digest(file), 0
    headers = build_hf_headers(headers={"Accept-Encoding": "identity"})
    with (
        get_session().stream(
            "GET",
            hf_hub_url(reference, file.name, revision=revision),
            headers=headers,
            follow_redirects=True,
            timeout=30,
        ) as response,
        destination.open("xb") as stream,
    ):
        hf_raise_for_status(response)
        if response.headers.get("content-encoding", "identity") != "identity":
            raise ValueError("pretrained download must use identity encoding")
        for data in response.iter_raw(chunk_size=CHUNK_BYTES):
            written += len(data)
            if written > file.size:
                raise ValueError(
                    f"pretrained download exceeds pinned size: {file.name}"
                )
            stream.write(data)
            digest.update(data)
    if written != file.size or digest.hexdigest() != file.checksum:
        raise ValueError(f"pretrained download size/checksum differs: {file.name}")


def fetch(reference, revision, file, root):
    from huggingface_hub import try_to_load_from_cache

    destination = root.joinpath(*PurePosixPath(file.name).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cached = try_to_load_from_cache(reference, file.name, revision=revision)
    if isinstance(cached, str):
        path = Path(cached)
        verify(path, file)
        destination.symlink_to(path.resolve())
    else:
        download(reference, revision, file, destination)
    return destination


def describe(reference, revision):
    """Read only config/index and HEAD metadata, never a tensor payload."""
    from huggingface_hub.errors import RemoteEntryNotFoundError

    result = [metadata(reference, revision, "config.json")]
    try:
        index = metadata(reference, revision, INDEX)
    except RemoteEntryNotFoundError:
        shards = ["model.safetensors"]
    else:
        with TemporaryDirectory(prefix="shadowspill-weight-index-") as root:
            shards = read_index(fetch(reference, revision, index, Path(root)))
        result.append(index)
    result.extend(metadata(reference, revision, name) for name in shards)
    return [asdict(file) for file in sorted(result, key=lambda file: file.name)]


@contextmanager
def snapshot(reference, revision, manifest, runtime):
    """Own temporary files per loader; external Hub caches remain read-only.

    Cleanup after each loader also bounds repeated restores of different bases.
    The node keeps the allowance/root until all workers exit, covering crashes.
    """
    if disk_bytes(manifest) > runtime.disk_bytes:
        raise ValueError("pretrained files exceed admitted initialization disk space")
    with TemporaryDirectory(prefix="weights-", dir=runtime.cache_root) as root:
        root = Path(root)
        for file in manifest:
            fetch(reference, revision, file, root)
        yield root
