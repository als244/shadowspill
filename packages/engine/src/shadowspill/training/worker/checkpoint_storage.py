"""Storage dependency supplied by deployment, not imported by training workers."""

from collections.abc import Callable, Sequence
from typing import Protocol

from shadowspill.orchestration.launch.publication import OutputWriter
from shadowspill.training.session.prepared import PreparedCall
from shadowspill.training.session.state import Transition
from shadowspill_client.protocol.checkpoints import CheckpointInfo
from shadowspill_client.protocol.requests import ArtifactRef

from .checkpoint_records import RankSave, RankSaved


class CheckpointFiles(Protocol):
    @property
    def references(self) -> tuple[ArtifactRef, ...]: ...

    def commit(self) -> tuple[ArtifactRef, ...]: ...

    def close(self) -> None: ...


class CheckpointStorage(Protocol):
    def describe(
        self, call: PreparedCall, transition: Transition
    ) -> CheckpointInfo: ...

    def stage(
        self,
        info: CheckpointInfo,
        ranks: Sequence[RankSave],
        write: Callable[[RankSave, OutputWriter], RankSaved],
        *,
        name: str | None,
    ) -> CheckpointFiles: ...
