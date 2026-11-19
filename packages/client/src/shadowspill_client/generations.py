"""Metadata and explicit directory copies for committed rank generations."""

import io
from pathlib import Path

from ._async import CallFuture
from .artifacts import ArtifactHandle
from .protocol.checkpoints import CheckpointManifest, WeightsManifest
from .protocol.codec import MAX_METADATA_BYTES, decode


class GenerationHandle[Manifest: CheckpointManifest | WeightsManifest](ArtifactHandle):
    """Saved per-worker state in service storage: a manifest and one file per worker.

    Calls return one of its two kinds, `CheckpointHandle` and
    `WeightsHandle`. Nothing is downloaded until `manifest` or `download` is
    called.

    Attributes
    ----------
    reference : ArtifactRef
        The manifest's artifact reference.
    """

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
            raise ValueError("saved generation exceeds its metadata bounds")
        with io.BytesIO() as stream:
            self._service._transport.download(
                "/v1/artifacts/" + self.id, self.reference, stream
            )
            body = stream.getvalue()
        value = decode(body)
        if not isinstance(value, self._manifest_type):
            raise ValueError("artifact is not the expected generation manifest")
        return value, body

    def manifest(self) -> CallFuture[Manifest]:
        """Download and verify the manifest, without the worker files.

        Returns
        -------
        CallFuture of CheckpointManifest or WeightsManifest
            The decoded manifest, of this handle's kind. It fails with
            `ValueError` if the artifact is not such a manifest.
        """
        return self._service._work.submit(lambda: self._manifest()[0])

    def download(self, path: str | Path, *, resume: bool = False) -> CallFuture[Path]:
        """Copy the manifest and every worker's file into a new local directory.

        The worker files are copied one at a time into ``ranks/`` and verified;
        ``manifest.json`` is written last, so its presence marks a complete
        copy. An existing destination is never overwritten. The Plans and
        frozen base weights the manifest names are separate artifacts and are
        not copied.

        Parameters
        ----------
        path : str or pathlib.Path
            The directory to create.
        resume : bool, default False
            Keep an interrupted copy, and continue one already at ``path``:
            complete files are verified instead of downloaded again, and a
            partial file continues where it stopped. Without it, a failed copy
            removes what this call created.

        Returns
        -------
        CallFuture of pathlib.Path
            The directory, once the copy is complete.

        Raises
        ------
        TypeError
            If ``resume`` is not a bool.
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
