"""Bounded Hub metadata access; model weights are opened only by workers."""

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from shadowspill.checkpoint.inspection import check_json

from .hub_files import CONFIG_BYTES, HubFile, fetch


def immutable_revision(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def accepts(reference):
    # The predicate performs no network access and accepts no local paths/URLs.
    from huggingface_hub.errors import HFValidationError
    from huggingface_hub.utils import validate_repo_id

    try:
        validate_repo_id(reference)
    except HFValidationError:
        return False
    return True


def read_config(path):
    with Path(path).open("rb") as stream:
        payload = stream.read(CONFIG_BYTES + 1)
    if len(payload) > CONFIG_BYTES:
        raise ValueError("pretrained config.json exceeds the metadata byte limit")
    check_json(payload)
    metadata = json.loads(payload)
    if not isinstance(metadata, dict):
        raise ValueError("pretrained config.json must be a JSON object")
    return metadata


def config(reference, revision):
    """Pin a branch/tag before fetching config.json; never download model weights.

    Reuse an existing exact-revision Hub cache entry read-only. Otherwise the
    small metadata download has a temporary cache, removed on success or failure.
    Authentication and endpoint selection use the server's ordinary Hub settings.
    """
    from huggingface_hub import (
        get_hf_file_metadata,
        hf_hub_url,
        try_to_load_from_cache,
    )

    if not accepts(reference):
        raise ValueError("pretrained reference must be a Hugging Face repository ID")
    if immutable_revision(revision):
        cached = try_to_load_from_cache(reference, "config.json", revision=revision)
        if isinstance(cached, str):
            return revision, read_config(cached)
    info = get_hf_file_metadata(hf_hub_url(reference, "config.json", revision=revision))
    if not immutable_revision(info.commit_hash):
        raise ValueError("pretrained metadata did not resolve to an immutable revision")
    if immutable_revision(revision) and info.commit_hash != revision:
        raise ValueError("pretrained metadata differs from the requested revision")
    if type(info.size) is not int or not 0 < info.size <= CONFIG_BYTES:
        raise ValueError("pretrained config.json exceeds the metadata byte limit")
    with TemporaryDirectory(prefix="shadowspill-model-metadata-") as cache:
        path = fetch(
            reference,
            info.commit_hash,
            HubFile("config.json", info.size, info.etag),
            Path(cache),
        )
        metadata = read_config(path)
    return info.commit_hash, metadata
