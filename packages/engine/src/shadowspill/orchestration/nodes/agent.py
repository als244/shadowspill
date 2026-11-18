"""Independent assigned-node owner, registered through outbound service control."""

from __future__ import annotations

import json
import os
import ssl
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from threading import RLock
from uuid import uuid4

from shadowspill.orchestration.launch.admission import (
    NodeAdmission,
    NodeLease,
    ResourceRequest,
)
from shadowspill.orchestration.launch.node_capacity import discover_capacity
from shadowspill.training.definitions import load_registry
from shadowspill_client._records import Value, integer, names
from shadowspill_client.protocol.requests import identifier

from .hardware import node_hardware
from .membership import Membership
from .ownership import NodeOwnership
from .protocol import Command, Join, Joined, NodeScope
from .transport import NodeTransport


class AgentAssignments:
    """Reserve/stop local resources; installed phase/worker adapters attach here."""

    def __init__(self, admission: NodeAdmission) -> None:
        self.admission = admission
        self._leases: dict[str, NodeLease] = {}
        self._active: tuple[str, ...] = ()
        self._stopped: set[str] = set()
        self._lock = RLock()
        self.closed = False

    def active(self) -> tuple[str, ...]:
        # A cleanup can wait on process exit. Heartbeats must not take that lock
        # or incorrectly lose the membership while unrelated work is healthy.
        return self._active

    def handle(self, command: Command) -> Mapping[str, Value]:
        if command.operation == "claim":
            payload = command.payload
            if set(payload) != {"devices", "host_bytes"} or not isinstance(
                payload["devices"], Mapping
            ):
                raise ValueError("claim requires complete device and host requirements")
            request = ResourceRequest(
                {
                    key: integer(value, "device bytes", minimum=1)
                    for key, value in payload["devices"].items()
                },
                integer(payload["host_bytes"], "host bytes"),
            )
            with self._lock:
                if self.closed or command.assignment in self._stopped:
                    raise RuntimeError("assignment ownership is stopping")
                if command.assignment in self._leases:
                    raise ValueError("assignment was already claimed")
                if len(self._stopped) + len(self._leases) >= 1024:
                    raise RuntimeError(
                        "assignment history is full; rejoin after cleanup"
                    )
                self._leases[command.assignment] = self.admission.reserve(request)
                self._active = tuple(self._leases)
            return {"claimed": True}
        if command.operation == "stop":
            self.stop(command.assignment)
            return {"stopped": True}
        raise ValueError("node has no installed handler for this assignment operation")

    def stop(self, assignment: str) -> None:
        with self._lock:
            self._stopped.add(assignment)
            lease = self._leases.get(assignment)
            if lease is not None:
                lease.close()
                del self._leases[assignment]
                self._active = tuple(self._leases)

    def close(self) -> None:
        with self._lock:
            self.closed = True
            self.admission.close()
            errors = []
            for assignment in tuple(self._leases):
                try:
                    self.stop(assignment)
                except BaseException as error:
                    errors.append(error)
            if errors:
                raise BaseExceptionGroup(
                    "agent resources still require cleanup", errors
                )


class NodeAgent:
    """One operator-supplied machine/step; discovery never uses SSH host lists.

    Construction is resource-free. ``serve`` holds node/device ownership locks,
    probes local assignment and joins the configured service. ``close`` fences
    new work and stops local children. Failed cleanup keeps ownership retryable.
    """

    def __init__(
        self,
        service_url: str,
        *,
        node_id: str,
        work_root: str | Path,
        token: str | None = None,
        pool: str = "default",
        workload_packages: Sequence[str] = ("shadowspill",),
        worker_address: str | None = None,
        environment: Mapping[str, str] | None = None,
        ssl_context: ssl.SSLContext | None = None,
        host_bytes: int = 256 << 20,
    ) -> None:
        self.node_id = identifier(node_id, "node identity")
        self.root = Path(work_root).absolute()
        self.pool, self.worker_address = pool, worker_address
        self.packages = names(workload_packages, "definition packages")
        self.host_bytes = integer(host_bytes, "agent host allowance", minimum=16 << 20)
        self.environment = dict(os.environ if environment is None else environment)
        credential = (
            self.environment.pop("SHADOWSPILL_NODE_TOKEN", "")
            if token is None
            else token
        )
        self.environment.pop("SHADOWSPILL_TOKEN", None)
        self.environment.pop("SHADOWSPILL_NODE_TOKEN", None)
        self._membership = Membership(
            NodeTransport(service_url, credential, ssl_context=ssl_context)
        )
        self._locks: NodeOwnership | None = None
        self._assignments: AgentAssignments | None = None
        self._served = False
        self.closed = False

    def _joined(self, joined: Joined) -> None:
        # Persist only public scope identifiers, never credentials. A restart
        # can reconcile this membership after the shared ownership locks prove
        # that every previous supervisor/worker has exited.
        temporary = self.root / "membership.tmp"
        temporary.write_text(json.dumps(joined.scope.to_dict(), sort_keys=True))
        temporary.replace(self.root / "membership.json")

    def serve(self) -> None:
        if self._served or self.closed:
            raise RuntimeError("node agent can only serve once")
        self._served = True
        self._locks = NodeOwnership()
        try:
            self._locks.acquire("node:" + self.node_id)
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            previous_path = self.root / "membership.json"
            previous = None
            if previous_path.exists():
                with previous_path.open("rb") as stream:
                    raw = stream.read(4097)
                if len(raw) > 4096:
                    raise ValueError("saved membership exceeds its bound")
                previous = NodeScope.from_dict(json.loads(raw))
            capacity = discover_capacity(
                environment=self.environment,
                root=self.root,
                reserved_host_bytes=self.host_bytes,
            )
            self._locks.acquire(*(f"device:{device.id}" for device in capacity.devices))
            admission = NodeAdmission(
                replace(capacity, node_id=self.node_id),
                guard_fds=self._locks.descriptors,
            )
            self._assignments = AgentAssignments(admission)
            load_registry(self.packages)
            self._membership.run(
                Join(
                    uuid4().hex,
                    node_hardware(admission, pool=self.pool),
                    self.packages,
                    self.worker_address,
                    previous,
                ),
                handle=self._assignments.handle,
                cleanup=self._assignments.close,
                assignments=self._assignments.active,
                joined=self._joined,
            )
        finally:
            self.close()

    def close(self) -> None:
        self._membership.stop.set()
        if self._assignments is not None:
            self._assignments.close()
        if self._locks is not None:
            self._locks.close()
        self.closed = True
