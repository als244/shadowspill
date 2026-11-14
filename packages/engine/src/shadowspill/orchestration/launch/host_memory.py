"""Read Linux physical and hierarchical cgroup memory limits without CUDA."""

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class MemoryLimit:
    scope: str
    limit_bytes: int
    used_bytes: int


@dataclass(frozen=True)
class HostMemory:
    physical_bytes: int
    physical_available_bytes: int
    limits: tuple[MemoryLimit, ...]

    @property
    def capacity_bytes(self) -> int:
        """A hard ceiling, distinct from current availability and reservations."""
        return min((self.physical_bytes, *(limit.limit_bytes for limit in self.limits)))

    @property
    def available_bytes(self) -> int:
        return min(
            (
                self.physical_available_bytes,
                *(
                    max(0, limit.limit_bytes - limit.used_bytes)
                    for limit in self.limits
                ),
            )
        )


def _read(path: Path, *, maximum: int = 16 << 10) -> str:
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"system memory description exceeds its bound: {path}")
    return data.decode("utf-8")


def _bytes(value: str) -> int:
    if re.fullmatch(r"[0-9]+", value.strip()) is None:
        raise ValueError("system memory counters must be nonnegative integers")
    return int(value)


def _path(value: str) -> PurePosixPath:
    decoded = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)
    path = PurePosixPath(decoded)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("cgroup membership is not visible beneath its mounted root")
    return path


def _mount(
    membership: PurePosixPath, kind: str, mountinfo: str
) -> tuple[Path, Path, bool]:
    matches = []
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        fields, filesystem = before.split(), after.split()
        if not separator or len(fields) < 6 or len(filesystem) < 3:
            raise ValueError("invalid cgroup mount description")
        if filesystem[0] != kind:
            continue
        if kind == "cgroup" and "memory" not in filesystem[2].split(","):
            continue
        root, mounted = _path(fields[3]), Path(_path(fields[4]))
        if membership.is_relative_to(root):
            matches.append(
                (
                    len(root.parts), mounted, mounted / membership.relative_to(root),
                    root == PurePosixPath("/"),
                )
            )
    if not matches:
        raise ValueError("cannot locate the process's memory cgroup mount")
    # Prefer the widest visible hierarchy, not a more specific bind mount that
    # would hide stricter ancestors above its root.
    _, mount_root, current, hierarchy_root = min(matches, key=lambda item: item[0])
    return mount_root, current, hierarchy_root


def host_memory(*, proc_root: Path = Path("/proc")) -> HostMemory:
    """Respect every visible ancestor, including limits above the current scope.

    Cgroup v1 memory and unified v2 are supported. Missing/inaccessible membership
    mounts fail instead of treating a container/job as the whole physical host.
    A hierarchy without an enabled memory controller has no cgroup memory files.
    The resulting availability is a snapshot, not a reservation or peak proof.
    """
    memory = {}
    for line in _read(proc_root / "meminfo").splitlines():
        name, separator, value = line.partition(":")
        if name in {"MemTotal", "MemAvailable"}:
            amount, unit = value.split()
            if not separator or unit != "kB" or name in memory:
                raise ValueError("invalid physical memory description")
            memory[name] = _bytes(amount) * 1024
    if memory.keys() != {"MemTotal", "MemAvailable"}:
        raise ValueError("physical memory requires MemTotal and MemAvailable")
    if not 0 <= memory["MemAvailable"] <= memory["MemTotal"] or not memory["MemTotal"]:
        raise ValueError("physical memory availability exceeds capacity")
    memberships = []
    for line in _read(proc_root / "self/cgroup").splitlines():
        hierarchy, controllers, value = line.split(":", 2)
        if "memory" in controllers.split(","):
            memberships.append(("cgroup", _path(value)))
        elif hierarchy == "0" and not controllers:
            memberships.append(("cgroup2", _path(value)))
    limits = []
    if memberships:
        mountinfo = _read(proc_root / "self/mountinfo", maximum=1 << 20)
        visited = set()
        for kind, member in memberships:
            root, current, hierarchy_root = _mount(member, kind, mountinfo)
            if not current.is_dir():
                raise ValueError("the process's memory cgroup is not accessible")
            names = (
                ("memory.max", "memory.current")
                if kind == "cgroup2"
                else ("memory.limit_in_bytes", "memory.usage_in_bytes")
            )
            while True:
                maximum, usage = (current / name for name in names)
                if current not in visited:
                    visited.add(current)
                    bounded = maximum.exists()
                    unbounded_root = (
                        current == root and hierarchy_root
                        and kind == "cgroup2" and not bounded
                    )
                    if not unbounded_root and bounded != usage.exists():
                        raise ValueError(
                            "memory cgroup has incomplete limit/usage files"
                        )
                    if bounded:
                        limit = _read(maximum).strip()
                        used = _bytes(_read(usage))
                        if limit != "max":
                            limits.append(
                                MemoryLimit(str(current), _bytes(limit), used)
                            )
                if current == root:
                    break
                current = current.parent
    return HostMemory(memory["MemTotal"], memory["MemAvailable"], tuple(limits))
