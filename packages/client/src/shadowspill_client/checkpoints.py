"""Checkpoint metadata and explicit local copies; no model or worker setup."""

import io
from pathlib import Path

from ._async import CallFuture
from .artifacts import ArtifactHandle
from .protocol.checkpoints import CHECKPOINT_MEDIA_TYPE, CheckpointManifest
from .protocol.codec import MAX_METADATA_BYTES, decode


class CheckpointHandle(ArtifactHandle):
    def _manifest(self) -> tuple[CheckpointManifest, bytes]:
        if (
            self.reference.media_type != CHECKPOINT_MEDIA_TYPE
            or self.reference.size_bytes > MAX_METADATA_BYTES
        ):
            raise ValueError("checkpoint reference exceeds its metadata bounds")
        with io.BytesIO() as stream:
            self._service._transport.download(
                "/v1/artifacts/" + self.id, self.reference, stream
            )
            body = stream.getvalue()
        value = decode(body)
        if not isinstance(value, CheckpointManifest):
            raise ValueError("artifact is not a checkpoint manifest")
        return value, body

    def manifest(self) -> CallFuture[CheckpointManifest]:
        """Verify the small generation manifest without downloading rank data."""
        return self._service._work.submit(lambda: self._manifest()[0])

    def download(self, path: str | Path) -> CallFuture[Path]:
        """Copy the manifest and rank files into a new directory, serially.

        Existing destinations are never overwritten. Publish the local manifest
        last; failure removes only files created by this call. Plans and frozen
        base dependencies are separate artifacts, not implicit large downloads.
        """
        destination = Path(path)

        def transfer() -> Path:
            manifest, body = self._manifest()
            destination.mkdir()
            files: list[Path] = []
            ranks = destination / "ranks"
            created_ranks = False
            try:
                ranks.mkdir()
                created_ranks = True
                for row in manifest.ranks:
                    target = ranks / f"rank-{row.rank:05d}.ssp"
                    with target.open("xb") as stream:
                        files.append(target)
                        self._service._transport.download(
                            "/v1/artifacts/" + row.archive.id, row.archive, stream
                        )
                target = destination / "manifest.json"
                with target.open("xb") as stream:
                    files.append(target)
                    stream.write(body)
            except BaseException:
                for target in reversed(files):
                    target.unlink()
                if created_ranks:
                    ranks.rmdir()
                destination.rmdir()
                raise
            return destination

        return self._service._work.submit(transfer)
