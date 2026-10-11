"""A pool image: frozen state as it lies in an SSD pool file, described so a
later runtime binds it in place instead of importing it again.

The image is the pool file a conversion wrote its state into. Its manifest,
``<image>.manifest.json``, records every storage root's offset and size and
the tensors viewing it, with their shapes, strides, storage offsets and
dtypes, under a caller-chosen ``frozen_state_id``. A model built on ``meta``
whose tensors of those names match is bound to those offsets by
:func:`shadowspill.pytorch.import_model_state`; everything else it owns is
allocated and initialized as usual, in the scratch file behind the image.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from shadowspill.runtime.configuration import RuntimeConfigurationError
from shadowspill.runtime.objects import pool_base_address

from .storage import persistent_state

SCHEMA = "shadowspill.pool_image/v1"
ALIGNMENT = 4096


@dataclass(frozen=True, slots=True)
class ImageView:
    """One tensor viewing a root: its state name and geometry."""

    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: str


@dataclass(frozen=True, slots=True)
class ImageRoot:
    """One storage root in the image: where it lies and who views it."""

    offset: int
    size_bytes: int
    views: tuple[ImageView, ...]


@dataclass(frozen=True, slots=True)
class PoolImage:
    """An image file and the manifest that describes the state in it."""

    path: Path
    frozen_state_id: str
    image_bytes: int
    roots: tuple[ImageRoot, ...]

    @staticmethod
    def manifest_path_for(path: str | os.PathLike[str]) -> Path:
        return Path(f"{Path(path)}.manifest.json")

    @property
    def manifest_path(self) -> Path:
        return self.manifest_path_for(self.path)

    def names(self) -> frozenset[str]:
        return frozenset(view.name for root in self.roots for view in root.views)

    def placements(self) -> dict[str, int]:
        """Each tensor name to the pool offset of the root it views."""

        return {view.name: root.offset for root in self.roots for view in root.views}

    def write_manifest(self) -> Path:
        document = {
            "schema": SCHEMA,
            "frozen_state_id": self.frozen_state_id,
            "image_bytes": self.image_bytes,
            "roots": [asdict(root) for root in self.roots],
        }
        self.manifest_path.write_text(json.dumps(document, indent=1) + "\n")
        return self.manifest_path

    def trim(self) -> None:
        """Cut the image file to the bytes its state reaches; the runtime closed."""

        os.truncate(self.path, self.image_bytes)

    @classmethod
    def read(cls, path: str | os.PathLike[str]) -> PoolImage:
        image = Path(path).expanduser().resolve()
        manifest = cls.manifest_path_for(image)
        if not image.is_file() or not manifest.is_file():
            raise RuntimeConfigurationError(
                f"a pool image needs both {image} and {manifest}"
            )
        document = json.loads(manifest.read_text())
        if document.get("schema") != SCHEMA:
            raise RuntimeConfigurationError(
                f"{manifest} is not a pool image manifest ({document.get('schema')!r})"
            )
        roots = tuple(
            ImageRoot(
                offset=int(root["offset"]),
                size_bytes=int(root["size_bytes"]),
                views=tuple(
                    ImageView(
                        name=str(view["name"]),
                        shape=tuple(int(n) for n in view["shape"]),
                        stride=tuple(int(n) for n in view["stride"]),
                        storage_offset=int(view["storage_offset"]),
                        dtype=str(view["dtype"]),
                    )
                    for view in root["views"]
                ),
            )
            for root in document["roots"]
        )
        result = cls(
            path=image,
            frozen_state_id=str(document["frozen_state_id"]),
            image_bytes=int(document["image_bytes"]),
            roots=roots,
        )
        actual = image.stat().st_size
        if actual != result.image_bytes:
            raise RuntimeConfigurationError(
                f"{image} holds {actual} bytes but its manifest describes "
                f"{result.image_bytes}; the image was not trimmed or has changed"
            )
        return result


def write_pool_image(
    target: object,
    *,
    runtime: object,
    pool: str,
    path: str | os.PathLike[str],
    frozen_state_id: str,
) -> PoolImage:
    """Describe the state ``target`` holds in ``pool`` as the image at ``path``.

    The pool's file is the image, so this writes only the manifest. Call it
    while the runtime is open and the state imported; trim the file after the
    runtime closes. Roots with trainable views are left out: a run initializes
    its own trainable state, and only the frozen roots are the image.
    """

    if not frozen_state_id:
        raise ValueError("frozen_state_id must be a nonempty string")
    state = persistent_state(runtime, target)
    if state is None:
        raise RuntimeConfigurationError(
            "the target's state is not imported into this runtime"
        )
    if state.pool != pool:
        raise RuntimeConfigurationError(
            f"the target's state is in pool {state.pool!r}, not {pool!r}"
        )
    base = pool_base_address(runtime, runtime.pools[pool].pool_id)
    roots = []
    for storage in state.storages:
        # Trainable state is each run's own; only frozen roots are the image.
        if any(view.requires_grad for view in storage.views):
            continue
        # Roots lie at the pool's own lease alignment; only the image's end is
        # rounded, since partial sectors are read and written as such.
        offset = int(storage.pool_pointer) - base
        if offset < 0:
            raise RuntimeConfigurationError(
                f"persistent object {storage.persistent_object_id} lies at offset "
                f"{offset}, which an image cannot describe"
            )
        roots.append(
            ImageRoot(
                offset=offset,
                size_bytes=int(storage.size_bytes),
                views=tuple(
                    ImageView(
                        name=view.name,
                        shape=tuple(view.shape),
                        stride=tuple(view.stride),
                        storage_offset=int(view.storage_offset),
                        dtype=str(view.tensor.dtype),
                    )
                    for view in storage.views
                ),
            )
        )
    roots.sort(key=lambda root: root.offset)
    end = max((root.offset + root.size_bytes for root in roots), default=0)
    image = PoolImage(
        path=Path(path).expanduser().resolve(),
        frozen_state_id=frozen_state_id,
        image_bytes=-(-end // ALIGNMENT) * ALIGNMENT,
        roots=tuple(roots),
    )
    image.write_manifest()
    return image


def image_placements(
    image: PoolImage, leaves: Mapping[str, torch.Tensor]
) -> dict[str, int]:
    """Check every tensor the image names against the model's, and place them.

    A name the model lacks, or a geometry or dtype that differs, means the image
    was made from other state, and is refused rather than bound.
    """

    problems = []
    for root in image.roots:
        for view in root.views:
            leaf = leaves.get(view.name)
            if leaf is None:
                problems.append(f"{view.name}: not in the model")
                continue
            expected = (
                view.shape,
                view.stride,
                view.storage_offset,
                view.dtype,
                root.size_bytes,
            )
            actual = (
                tuple(leaf.shape),
                tuple(leaf.stride()),
                int(leaf.storage_offset()),
                str(leaf.dtype),
                int(leaf.untyped_storage().nbytes()),
            )
            if expected != actual:
                problems.append(f"{view.name}: image {expected}, model {actual}")
    if problems:
        shown = "; ".join(problems[:4]) + (" and more" if len(problems) > 4 else "")
        raise RuntimeConfigurationError(
            f"pool image {image.path} does not describe this model's state: {shown}"
        )
    return image.placements()


__all__ = [
    "ImageRoot",
    "ImageView",
    "PoolImage",
    "image_placements",
    "write_pool_image",
]
