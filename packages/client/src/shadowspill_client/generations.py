"""Metadata and explicit directory copies for committed rank generations."""

import io
from pathlib import Path

from ._async import CallFuture
from .artifacts import ArtifactHandle
from .protocol.checkpoints import CheckpointManifest
from .protocol.codec import MAX_METADATA_BYTES, decode


class GenerationHandle[Manifest: CheckpointManifest](ArtifactHandle):
    @property
    def _manifest_type(self) -> type[Manifest]:
        raise NotImplementedError

    @property
    def _media_type(self) -> str:
        raise NotImplementedError

    def _manifest(self) -> tuple[Manifest, bytes]:
        if (
            self.reference.media_type != self._media_type
            or self.reference.size_bytes > MAX_METADATA_BYTES
        ):
            raise ValueError("checkpoint reference exceeds its metadata bounds")
        with io.BytesIO() as stream:
            self._service._transport.download(
                "/v1/artifacts/" + self.id, self.reference, stream
            )
            body = stream.getvalue()
        value = decode(body)
        if not isinstance(value, self._manifest_type):
            raise ValueError("artifact is not a checkpoint manifest")
        return value, body

    def manifest(self) -> CallFuture[Manifest]:
        """Verify the small generation manifest without downloading rank data."""
        return self._service._work.submit(lambda: self._manifest()[0])

    def download(self, path: str | Path, *, resume: bool = False) -> CallFuture[Path]:
        """Copy the manifest and rank files into a new directory, serially.

        Existing destinations are never overwritten. Publish the local manifest
        last; failure removes only files created by this call. Plans and frozen
        base dependencies are separate artifacts, not implicit large downloads.
        """
        if type(resume) is not bool:
            raise TypeError("download resume must be boolean")
        destination = Path(path)

        def transfer() -> Path:
            if resume:
                from .downloads import resume_checkpoint

                return resume_checkpoint(
                    self._service._transport,
                    self.reference,
                    destination,
                    self._manifest,
                    memory=self._service._memory,
                )
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
