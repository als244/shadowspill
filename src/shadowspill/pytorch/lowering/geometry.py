"""The compiler facts required to bind task inputs and outputs."""

from typing import Protocol

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.task.geometry import TaskStorageGeometry
from shadowspill.task.storage import TaskStorageContract


class TaskGeometry(Protocol):
    def contract(self, artifact: GraphArtifact) -> TaskStorageContract: ...

    def geometry(
        self, artifact: GraphArtifact, metadata_digest: str | None = None
    ) -> TaskStorageGeometry: ...
