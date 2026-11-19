"""Transport-independent membership, fencing and bounded command receipts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from threading import Condition
from time import monotonic
from uuid import uuid4

from shadowspill_client._records import Value, integer
from shadowspill_client.protocol.hardware import HardwareInventory, HardwareNode

from ..launch.admission import ResourceRequest
from .authentication import NodeAccess
from .protocol import (
    CODEC,
    CONTROL_BYTES,
    Acknowledged,
    Command,
    Heartbeat,
    Join,
    Joined,
    Leave,
    NodeScope,
    Poll,
    Receipt,
    Reconcile,
)


class NodeUnavailable(RuntimeError):
    """Membership has no authority; uncertain commands must never be replayed."""


@dataclass
class _Pending:
    command: Command
    receipt: Receipt | None = None


@dataclass
class _Member:
    join: Join
    joined: Joined
    expires: float
    heartbeat: int = 0
    acknowledged: int = 0
    next_sequence: int = 1
    retired: int = 0
    fenced: bool = False
    cleaned: bool = False
    polling: bool = False
    pending: dict[int, _Pending] = field(default_factory=dict)
    assignments: set[str] = field(default_factory=set)
    # Claimed capacity stays owned until confirmed Stop, even if the heartbeat
    # has not yet observed Claim. A timeout can fence it but cannot release it.
    devices: dict[str, tuple[str, ...]] = field(default_factory=dict)
    host_bytes: dict[str, int] = field(default_factory=dict)


class NodeRegistry:
    def __init__(
        self,
        incarnation: str,
        *,
        heartbeat_seconds: float = 5,
        lease_seconds: float = 30,
        poll_seconds: float = 10,
        max_nodes: int = 32,
        max_commands: int = 32,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._durations = Joined(
            NodeScope(incarnation, "configuration"),
            heartbeat_seconds,
            lease_seconds,
            poll_seconds,
        )
        self.incarnation = incarnation
        self.max_nodes = integer(max_nodes, "maximum nodes", minimum=1)
        self.max_commands = integer(max_commands, "maximum pending commands", minimum=1)
        self._clock, self._condition = clock, Condition()
        self._members: dict[str, _Member] = {}
        self.closed = False

    def _expire(self, member: _Member) -> None:
        if self._clock() >= member.expires:
            member.fenced = True
            self._condition.notify_all()

    def _member(self, node: str, scope: NodeScope, *, live: bool = True) -> _Member:
        member = self._members.get(node)
        if member is None or member.joined.scope != scope:
            raise NodeUnavailable("node belongs to another membership/controller")
        self._expire(member)
        if live and (self.closed or member.fenced):
            raise NodeUnavailable("node membership is fenced")
        return member

    def join(self, access: NodeAccess, request: Join) -> Joined:
        if (
            request.hardware.id != access.node_id
            or request.hardware.pool != access.pool
        ):
            raise PermissionError("credential does not authorize this node/pool")
        CODEC.encode(request, max_bytes=CONTROL_BYTES)
        with self._condition:
            if self.closed:
                raise NodeUnavailable("controller is closed")
            old = self._members.get(access.node_id)
            if old is not None:
                self._expire(old)
                if old.join.incarnation == request.incarnation:
                    if old.fenced or request != old.join:
                        raise NodeUnavailable(
                            "join cannot change or revive a membership"
                        )
                    return old.joined
                # A new local supervisor must reconcile the exact old scope
                # after taking its ownership lock and confirming child exit.
                if not old.cleaned and (
                    not old.fenced or request.previous != old.joined.scope
                ):
                    raise NodeUnavailable("previous node ownership is not reconciled")
            elif len(self._members) >= self.max_nodes:
                raise TimeoutError("node membership capacity is full")
            if (
                sum(
                    len(member.join.hardware.devices)
                    for node, member in self._members.items()
                    if node != access.node_id
                )
                + len(request.hardware.devices)
                > 1024
            ):
                raise TimeoutError("node inventory exceeds its device bound")
            joined = replace(
                self._durations, scope=NodeScope(self.incarnation, uuid4().hex)
            )
            self._members[access.node_id] = _Member(
                request, joined, self._clock() + joined.lease_seconds
            )
            self._condition.notify_all()
            return joined

    def heartbeat(self, node: str, request: Heartbeat) -> Acknowledged:
        with self._condition:
            member = self._member(node, request.scope)
            if request.sequence <= member.heartbeat:
                raise NodeUnavailable("heartbeat renewal is stale")
            if set(request.assignments) - member.assignments:
                raise NodeUnavailable("node daemon reports an unknown assignment")
            member.heartbeat = request.sequence
            member.expires = self._clock() + member.joined.lease_seconds
            return Acknowledged(request.scope, request.sequence)

    def inventory(self) -> HardwareInventory:
        with self._condition:
            nodes: list[HardwareNode] = []
            for member in self._members.values():
                self._expire(member)
                if member.fenced or self.closed:
                    continue
                busy = {
                    device for devices in member.devices.values() for device in devices
                }
                hardware = member.join.hardware
                nodes.append(
                    replace(
                        hardware,
                        host_available_bytes=max(
                            0,
                            hardware.host_available_bytes
                            - sum(member.host_bytes.values()),
                        ),
                        devices=tuple(
                            replace(
                                device,
                                available=device.available and device.id not in busy,
                            )
                            for device in hardware.devices
                        ),
                    )
                )
            return HardwareInventory(tuple(sorted(nodes, key=lambda node: node.id)))

    def submit(
        self,
        node: str,
        assignment: str,
        operation: str,
        payload: Mapping[str, Value] | None = None,
    ) -> Command:
        with self._condition:
            member = self._members[node]
            self._member(node, member.joined.scope)
            allowance = self.max_commands + (operation == "stop")
            if len(member.pending) >= allowance or sum(
                len(item.pending) for item in self._members.values()
            ) >= 128 + (operation == "stop"):
                raise TimeoutError("node command capacity is full")
            command = Command(
                member.joined.scope,
                member.next_sequence,
                assignment,
                operation,
                {} if payload is None else payload,
            )
            CODEC.encode(command, max_bytes=CONTROL_BYTES)
            if operation == "claim":
                if len(member.assignments) >= 1024:
                    raise TimeoutError(
                        "membership assignment history is full; rejoin after cleanup"
                    )
                if assignment in member.assignments:
                    raise ValueError(
                        "assignment already exists; reuse its command receipt"
                    )
                devices = command.payload.get("devices")
                if not isinstance(devices, Mapping) or not set(devices) <= {
                    device.id for device in member.join.hardware.devices
                }:
                    raise ValueError("claim requires assigned device requirements")
                demand = ResourceRequest(
                    {
                        key: integer(value, "device bytes", minimum=1)
                        for key, value in devices.items()
                    },
                    integer(command.payload.get("host_bytes", 0), "host bytes"),
                )
                hardware = member.join.hardware
                if demand.host_bytes > hardware.host_capacity_bytes or any(
                    demand.devices.get(device.id, 0) > device.capacity_bytes
                    for device in hardware.devices
                ):
                    raise ValueError("claim cannot fit assigned node capacity")
                if (
                    demand.host_bytes + sum(member.host_bytes.values())
                    > hardware.host_available_bytes
                ):
                    raise NodeUnavailable("node host capacity has live owners")
                if set(devices).intersection(
                    device for group in member.devices.values() for device in group
                ):
                    raise NodeUnavailable("selected devices have another assignment")
                member.assignments.add(assignment)
                member.devices[assignment] = tuple(devices)
                member.host_bytes[assignment] = demand.host_bytes
            elif assignment not in member.assignments:
                raise ValueError("command has no resource assignment")
            member.next_sequence += 1
            member.pending[command.sequence] = _Pending(command)
            self._condition.notify_all()
            return command

    def poll(self, node: str, request: Poll) -> Command | Acknowledged:
        with self._condition:
            member = self._member(node, request.scope)
            if member.polling:
                raise TimeoutError("node already has a command subscription")
            if not member.acknowledged <= request.acknowledged < member.next_sequence:
                raise ValueError("invalid command acknowledgement")
            member.acknowledged = request.acknowledged
            self._retire(member)
            member.polling = True
            deadline = self._clock() + member.joined.poll_seconds
            try:
                while True:
                    self._member(node, request.scope)
                    if member.retired > request.retired:
                        return Acknowledged(request.scope, member.retired)
                    for sequence, pending in member.pending.items():
                        if sequence > request.acknowledged:
                            return pending.command
                    remaining = min(deadline, member.expires) - self._clock()
                    if remaining <= 0:
                        return Acknowledged(request.scope, member.retired)
                    self._condition.wait(remaining)
            finally:
                member.polling = False

    def complete(self, node: str, receipt: Receipt) -> Acknowledged:
        CODEC.encode(receipt, max_bytes=CONTROL_BYTES)
        with self._condition:
            member = self._member(node, receipt.scope)
            pending = member.pending.get(receipt.sequence)
            if pending is None:
                # A result whose acknowledgement was lost is safely retired;
                # there is no code path from an old receipt to command execution.
                if receipt.sequence <= member.retired:
                    return Acknowledged(receipt.scope, member.retired)
                raise ValueError("receipt has no submitted command")
            if pending.receipt is not None and receipt != pending.receipt:
                raise ValueError("command receipt is immutable")
            pending.receipt = receipt
            command = pending.command
            if (command.operation == "stop" and receipt.error is None) or (
                command.operation == "claim" and receipt.error is not None
            ):
                member.devices.pop(command.assignment, None)
                member.host_bytes.pop(command.assignment, None)
            self._condition.notify_all()
            return Acknowledged(receipt.scope, member.retired)

    def wait(
        self, node: str, command: Command, *, timeout: float | None = None
    ) -> Receipt:
        deadline = None if timeout is None else self._clock() + timeout
        with self._condition:
            while True:
                member = self._member(node, command.scope)
                pending = member.pending.get(command.sequence)
                if pending is None or pending.command != command:
                    raise ValueError("command is not retained by this membership")
                if pending.receipt is not None:
                    return pending.receipt
                remaining = member.expires - self._clock()
                if deadline is not None:
                    remaining = min(remaining, deadline - self._clock())
                if remaining <= 0:
                    raise TimeoutError("node command has not completed")
                self._condition.wait(remaining)

    def retire(self, node: str, command: Command) -> None:
        """Caller consumed a receipt; preserve sequence tombstones after eviction."""
        with self._condition:
            member = self._member(node, command.scope, live=False)
            pending = member.pending.get(command.sequence)
            if pending is None or pending.command != command or pending.receipt is None:
                raise ValueError("retirement requires a completed command")
            # Ordered prefix retirement also bounds the node daemon's receipt cache.
            if command.sequence != member.retired + 1:
                raise ValueError("retire commands in submission order")
            member.retired = command.sequence
            self._retire(member)
            self._condition.notify_all()

    @staticmethod
    def _retire(member: _Member) -> None:
        for sequence in tuple(member.pending):
            if sequence <= min(member.retired, member.acknowledged):
                del member.pending[sequence]

    def leave(self, node: str, request: Leave) -> Acknowledged:
        with self._condition:
            member = self._member(node, request.scope, live=False)
            member.fenced = member.cleaned = True
            member.devices.clear()
            member.host_bytes.clear()
            self._condition.notify_all()
            return Acknowledged(request.scope, member.retired)

    def reconcile(self, node: str, request: Reconcile) -> Reconcile:
        """An authenticated node daemon has reaped its journaled predecessor locally.

        A dropped Join/reconciliation acknowledgement is safe to repeat. A stale
        journal never fences a different incarnation; a controller restart or an
        unaccepted Join has no matching ownership to retire.
        """
        with self._condition:
            member = self._members.get(node)
            if member is not None and member.join.incarnation == request.incarnation:
                member.fenced = member.cleaned = True
                member.devices.clear()
                member.host_bytes.clear()
                self._condition.notify_all()
            return request

    def close(self) -> None:
        with self._condition:
            self.closed = True
            for member in self._members.values():
                member.fenced = True
            self._condition.notify_all()
