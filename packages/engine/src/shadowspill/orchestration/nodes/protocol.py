"""Private, bounded node membership and command records; no executable payloads."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from shadowspill_client._records import (
    Record,
    Value,
    integer,
    mapping,
    names,
    number,
    record,
    text,
)
from shadowspill_client.protocol.codec import RecordCodec
from shadowspill_client.protocol.hardware import HardwareNode
from shadowspill_client.protocol.requests import identifier

VERSION = 1
CONTROL_BYTES = 64 << 10
HTTP_BYTES = 16 * CONTROL_BYTES


@dataclass(frozen=True)
class NodeScope(Record):
    controller: str
    membership: str

    def __post_init__(self) -> None:
        identifier(self.controller, "controller incarnation")
        identifier(self.membership, "node membership")


@dataclass(frozen=True)
class Join(Record):
    incarnation: str
    hardware: HardwareNode
    packages: Sequence[str]
    worker_address: str | None = None
    previous: NodeScope | None = None
    version: int = VERSION

    def __post_init__(self) -> None:
        identifier(self.incarnation, "node daemon incarnation")
        object.__setattr__(
            self, "hardware", record(HardwareNode, self.hardware, "node")
        )
        object.__setattr__(self, "packages", names(self.packages, "installed packages"))
        if self.worker_address is not None:
            address = text(self.worker_address, "worker address")
            if len(address) > 253 or any(c.isspace() for c in address):
                raise ValueError("invalid worker address")
        if self.previous is not None:
            object.__setattr__(
                self, "previous", record(NodeScope, self.previous, "previous")
            )
        if type(self.version) is not int or self.version != VERSION:
            raise ValueError("unsupported node protocol version")


@dataclass(frozen=True)
class Joined(Record):
    scope: NodeScope
    heartbeat_seconds: float
    lease_seconds: float
    poll_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", record(NodeScope, self.scope, "membership"))
        for name in ("heartbeat_seconds", "lease_seconds", "poll_seconds"):
            if not number(getattr(self, name), name, minimum=0):
                raise ValueError("membership durations must be positive")
        if self.heartbeat_seconds * 3 > self.lease_seconds:
            raise ValueError("membership needs at least three heartbeat intervals")


@dataclass(frozen=True)
class Heartbeat(Record):
    scope: NodeScope
    sequence: int
    assignments: Sequence[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", record(NodeScope, self.scope, "membership"))
        integer(self.sequence, "heartbeat sequence", minimum=1)
        object.__setattr__(self, "assignments", names(self.assignments, "assignments"))
        if len(self.assignments) > 1024:
            raise ValueError("too many active assignments")
        for assignment in self.assignments:
            identifier(assignment, "assignment")


@dataclass(frozen=True)
class Poll(Record):
    scope: NodeScope
    acknowledged: int = 0
    retired: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", record(NodeScope, self.scope, "membership"))
        integer(self.acknowledged, "acknowledged command")
        integer(self.retired, "retired command")
        if self.retired > self.acknowledged:
            raise ValueError("retirement exceeds acknowledged delivery")


@dataclass(frozen=True)
class Command(Record):
    scope: NodeScope
    sequence: int
    assignment: str
    operation: str
    payload: Mapping[str, Value] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", record(NodeScope, self.scope, "membership"))
        integer(self.sequence, "command sequence", minimum=1)
        identifier(self.assignment, "assignment")
        # Dispatch is installed node daemon code. No shell command, module or callable
        # can be selected by a client or deserialized from this record.
        if self.operation not in ("claim", "stop", "phase", "worker"):
            raise ValueError("unknown node operation")
        object.__setattr__(self, "payload", mapping(self.payload, "command payload"))


@dataclass(frozen=True)
class Receipt(Record):
    scope: NodeScope
    sequence: int
    payload: Mapping[str, Value] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", record(NodeScope, self.scope, "membership"))
        integer(self.sequence, "completed command", minimum=1)
        object.__setattr__(self, "payload", mapping(self.payload, "receipt payload"))
        if self.error is not None and len(text(self.error, "command error")) > 1024:
            raise ValueError("node error exceeds its bound")


@dataclass(frozen=True)
class Acknowledged(Record):
    scope: NodeScope
    sequence: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", record(NodeScope, self.scope, "membership"))
        integer(self.sequence, "acknowledged sequence")


@dataclass(frozen=True)
class Leave(Record):
    scope: NodeScope

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", record(NodeScope, self.scope, "membership"))


@dataclass(frozen=True)
class Reconcile(Record):
    """Confirm local cleanup of a journaled attempt, even if Join's reply was lost.

    The node daemon must first acquire its node/device locks, which supervisors retain
    until all old children exit. The identical reply acknowledges that cleanup;
    it grants no new membership or execution authority.
    """

    incarnation: str

    def __post_init__(self) -> None:
        identifier(self.incarnation, "previous node daemon incarnation")


CODEC = RecordCodec(
    {
        "node.join": Join,
        "node.joined": Joined,
        "node.heartbeat": Heartbeat,
        "node.poll": Poll,
        "node.command": Command,
        "node.receipt": Receipt,
        "node.acknowledged": Acknowledged,
        "node.leave": Leave,
        "node.reconcile": Reconcile,
    }
)
