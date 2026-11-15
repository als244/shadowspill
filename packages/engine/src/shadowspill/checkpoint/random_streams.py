"""Q24 logical RNG mapping, independent of framework or physical device identity.

Payloads use the existing small metadata-value codec at the runtime boundary.
This module treats their declared representations as opaque: a matching logical
stream retains its exact saved state, while an unmappable stream needs an
explicitly supported seed initializer. It cannot initialize other state roles.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from shadowspill.artifacts._values import digest, encoded, nonnegative_int
from shadowspill.artifacts.state import StateId

MAX_STREAM_BYTES = 64 << 10
MAX_STREAMS = 65536
MAX_SNAPSHOT_BYTES = 1 << 20


def default_streams(
    logical_rank: int, torch_version: str, *, accelerator: bool
) -> tuple["RandomStream", ...]:
    """Declare worker defaults without importing a framework or opening a device."""
    representations = {
        "python": "python.random@3",
        "numpy": "numpy.MT19937@1",
        "torch": "torch.cpu@" + torch_version,
    }
    if accelerator:
        representations["accelerator"] = "torch.cuda@" + torch_version
    return tuple(
        RandomStream(
            StateId("rng", "worker.default", (str(logical_rank), purpose, name)),
            representation,
            logical_rank,
        )
        for purpose in ("training", "evaluation")
        for name, representation in representations.items()
    )


def _name(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
        or len(value.encode()) > 1024
    ):
        raise ValueError(f"{name} requires a bounded nonempty name")


@dataclass(frozen=True)
class RandomStream:
    """Logical identity and representation; rank locates its current owner."""

    identity: StateId
    representation: str
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.identity, StateId) or self.identity.role != "rng":
            raise ValueError("random streams require a logical RNG state identity")
        _name(self.identity.name, "stream identity")
        if len(self.identity.component_path) > 16:
            raise ValueError("random stream component path exceeds its bound")
        for part in self.identity.component_path:
            _name(part, "stream component")
        _name(self.representation, "generator representation")
        nonnegative_int(self.rank, "logical stream owner")

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "representation": self.representation,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RandomStream":
        if not isinstance(value, Mapping) or set(value) != {
            "identity",
            "representation",
            "rank",
        }:
            raise ValueError("invalid random stream declaration")
        return cls(
            StateId.from_dict(value["identity"]), value["representation"], value["rank"]
        )


@dataclass(frozen=True)
class RandomState:
    """Saved codec payload or an explicit seed instruction, never both."""

    stream: RandomStream
    payload: str | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stream, RandomStream):
            raise TypeError("random state requires a declared stream")
        if (self.payload is None) == (self.seed is None):
            raise ValueError("random state requires exactly one payload or seed")
        if self.payload is not None and (
            type(self.payload) is not str
            or not self.payload
            or len(self.payload) > MAX_STREAM_BYTES
            or len(self.payload.encode()) > MAX_STREAM_BYTES
        ):
            raise ValueError("random state payload exceeds its byte bound")
        if self.seed is not None:
            nonnegative_int(self.seed, "generator seed")
            if self.seed >= 1 << 64:
                raise ValueError("generator seed must fit in 64 bits")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream": self.stream.to_dict(),
            "payload": self.payload,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RandomState":
        if not isinstance(value, Mapping) or set(value) != {
            "stream",
            "payload",
            "seed",
        }:
            raise ValueError("invalid random stream state")
        return cls(
            RandomStream.from_dict(value["stream"]), value["payload"], value["seed"]
        )


def snapshot(root_seed: int, states: Sequence[RandomState]) -> dict[str, Any]:
    """Bound one worker's checkpoint metadata before encoding/publication."""
    nonnegative_int(root_seed, "saved root seed")
    if (
        not isinstance(states, (list, tuple))
        or len(states) > 32
        or any(not isinstance(state, RandomState) for state in states)
    ):
        raise ValueError("RNG snapshot requires at most 32 declared streams")
    _unique(tuple(state.stream for state in states))
    result = {
        "version": 1,
        "root_seed": root_seed,
        "streams": [state.to_dict() for state in states],
    }
    if len(encoded(result)) > MAX_SNAPSHOT_BYTES:
        raise ValueError("RNG snapshot exceeds its metadata bound")
    return result


def read_snapshot(value: Mapping[str, Any]) -> tuple[int, tuple[RandomState, ...]]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"version", "root_seed", "streams"}
        or type(value["version"]) is not int
        or value["version"] != 1
        or not isinstance(value["streams"], (list, tuple))
        or len(value["streams"]) > 32
    ):
        raise ValueError("invalid RNG snapshot version or stream declarations")
    states = tuple(RandomState.from_dict(row) for row in value["streams"])
    snapshot(value["root_seed"], states)
    return value["root_seed"], states


@dataclass(frozen=True)
class RandomMapping:
    """Prepared states plus complete source/target mapping provenance."""

    root_seed: int
    restore_id: str
    sources: tuple[RandomStream, ...]
    states: tuple[RandomState, ...]
    derived: tuple[StateId, ...]

    def provenance(self) -> dict[str, Any]:
        derived = set(self.derived)
        return {
            "version": 1,
            "root_seed": self.root_seed,
            "restore_id": self.restore_id,
            "source_streams": [stream.to_dict() for stream in self.sources],
            "target_streams": [
                {
                    **state.stream.to_dict(),
                    "action": "derive"
                    if state.stream.identity in derived
                    else "preserve",
                    "seed": state.seed,
                }
                for state in self.states
            ],
        }


def _unique(streams: Sequence[RandomStream]) -> None:
    if len(streams) > MAX_STREAMS or any(
        not isinstance(stream, RandomStream) for stream in streams
    ):
        raise ValueError("random streams require a bounded declared inventory")
    if len({stream.identity for stream in streams}) != len(streams):
        raise ValueError("duplicate logical random stream identity")


def map_random_streams(
    sources: Sequence[RandomState],
    targets: Sequence[RandomStream],
    *,
    root_seed: int,
    restore_id: str,
    initializers: Collection[str],
) -> RandomMapping:
    """Preserve matching logical streams; derive only declared RNG initializers.

    The source root seed comes from the saved checkpoint, not the new run's
    execution settings. The owner supplies one stable restore identity before
    publication. Physical GPU IDs and source/target owner ranks never enter the
    derivation. A custom representation must declare a seed initializer or use
    its own explicit converter; equal payload sizes do not imply compatibility.
    """
    nonnegative_int(root_seed, "saved root seed")
    _name(restore_id, "restore identity")
    if not isinstance(sources, (tuple, list)) or len(sources) > MAX_STREAMS:
        raise ValueError("saved random states require a bounded sequence")
    if any(not isinstance(state, RandomState) for state in sources):
        raise TypeError("saved random states require validated declarations")
    if not isinstance(targets, (tuple, list)):
        raise TypeError("target random streams require a sequence")
    source_streams = tuple(state.stream for state in sources)
    _unique(source_streams)
    _unique(targets)
    if isinstance(initializers, (str, bytes)) or len(initializers) > MAX_STREAMS:
        raise ValueError("declare supported generator representations")
    for representation in initializers:
        _name(representation, "seed initializer representation")
    old = {state.stream.identity: state for state in sources}
    states, derived = [], []
    for stream in sorted(targets, key=lambda stream: stream.identity):
        previous = old.get(stream.identity)
        if (
            previous is not None
            and previous.stream.representation == stream.representation
        ):
            states.append(replace(previous, stream=stream))
            continue
        if stream.representation not in initializers:
            raise ValueError(
                "unmappable RNG representation needs a declared initializer "
                "or converter: " + stream.representation
            )
        seed = int(
            digest(
                {
                    "domain": "shadowspill.rng-restore",
                    "version": 1,
                    "root_seed": root_seed,
                    "restore_id": restore_id,
                    "stream": stream.identity.to_dict(),
                    "representation": stream.representation,
                }
            )[:16],
            16,
        )
        states.append(RandomState(stream, seed=seed))
        derived.append(stream.identity)
    return RandomMapping(
        root_seed,
        restore_id,
        tuple(sorted(source_streams, key=lambda stream: stream.identity)),
        tuple(states),
        tuple(derived),
    )
