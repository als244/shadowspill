"""Bounded read-only facts for an existing Slurm job; never create an allocation."""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType

from shadowspill_client._records import integer, number
from shadowspill_client.protocol.codec import _constant, _depth, _object
from shadowspill_client.protocol.requests import identifier

from .supervision import OwnedProcess

# Pin the scheduler schema, not the installed Slurm release's changing default.
DATA_PARSER = "data_parser/v0.0.44"
MAX_JOB_BYTES = 16 << 20


@dataclass(frozen=True)
class SlurmNode:
    name: str
    cpus: int
    memory_bytes: int

    def __post_init__(self) -> None:
        identifier(self.name, "allocated Slurm node")
        integer(self.cpus, "allocated CPUs", minimum=1)
        integer(self.memory_bytes, "allocated node memory", minimum=1)


@dataclass(frozen=True)
class SlurmJob:
    """A current scheduler snapshot, not a new reservation or proof of free GPUs.

    Per-node memory is the scheduler's allocated value, not requested memory per
    CPU times a CPU count. Node daemon admission must also apply local physical/cgroup
    limits and retain process/artifact overhead. GPU UUIDs come from the isolated
    probe inside each assigned step, never from these global Slurm GPU indices.
    """

    job_id: int
    user_id: int
    nodes: Mapping[str, SlurmNode]

    def __post_init__(self) -> None:
        integer(self.job_id, "existing Slurm job ID", minimum=1)
        integer(self.user_id, "Slurm job owner UID")
        if not isinstance(self.nodes, Mapping) or not 0 < len(self.nodes) <= 65536:
            raise ValueError("Slurm job requires a bounded nonempty node allocation")
        if any(
            not isinstance(node, SlurmNode) or name != node.name
            for name, node in self.nodes.items()
        ):
            raise ValueError("Slurm node records differ from their names")
        object.__setattr__(self, "nodes", MappingProxyType(dict(self.nodes)))


def parse_job(data: bytes, *, job_id: int, user_id: int) -> SlurmJob:
    """Extract only matching active-owner allocation facts from the pinned schema."""
    integer(job_id, "existing Slurm job ID", minimum=1)
    integer(user_id, "Slurm job owner UID")
    if not isinstance(data, bytes) or len(data) > MAX_JOB_BYTES:
        raise ValueError("Slurm job response exceeds its byte bound")
    try:
        value = json.loads(data, object_pairs_hook=_object, parse_constant=_constant)
        _depth(value)
        if value["meta"]["plugin"]["data_parser"] != DATA_PARSER:
            raise ValueError("Slurm response uses a different data parser")
        if value["errors"] != []:
            raise ValueError("Slurm returned scheduler errors")
        jobs = value["jobs"]
        if not isinstance(jobs, list) or len(jobs) != 1:
            raise ValueError("Slurm query must identify exactly one existing job")
        job = jobs[0]
        if type(job["job_id"]) is not int or job["job_id"] != job_id:
            raise ValueError("Slurm response belongs to a different job")
        if type(job["user_id"]) is not int or job["user_id"] != user_id:
            raise ValueError("Slurm job belongs to another allocation owner")
        if job["job_state"] != ["RUNNING"]:
            raise ValueError("Slurm job is not in a stable running state")
        resources = job["job_resources"]
        count = integer(resources["nodes"]["count"], "allocated node count", minimum=1)
        rows = resources["nodes"]["allocation"]
        if not isinstance(rows, list) or not len(rows) == count <= 65536:
            raise ValueError("Slurm per-node allocation is incomplete")
        nodes = {}
        for index, row in enumerate(rows):
            if type(row["index"]) is not int or row["index"] != index:
                raise ValueError("Slurm allocated-node indices are not complete")
            allocated = integer(row["memory"]["allocated"], "allocated MiB", minimum=1)
            node = SlurmNode(row["name"], row["cpus"]["count"], allocated << 20)
            if node.name in nodes:
                raise ValueError("Slurm returned duplicate allocated nodes")
            nodes[node.name] = node
        cpus = integer(resources["cpus"], "total allocated CPUs", minimum=1)
        if sum(node.cpus for node in nodes.values()) != cpus:
            raise ValueError("Slurm per-node CPU counts differ from the allocation")
        return SlurmJob(job_id, user_id, nodes)
    except (KeyError, TypeError, IndexError, UnicodeError, RecursionError) as error:
        raise ValueError(
            "Slurm returned malformed or incomplete allocation facts"
        ) from error


def query_job(
    job_id: int,
    *,
    environment: Mapping[str, str],
    root: Path,
    timeout: float = 30,
) -> SlurmJob:
    """Run only a bounded scontrol show for this node daemon's existing job and UID."""
    integer(job_id, "existing Slurm job ID", minimum=1)
    if number(timeout, "Slurm query timeout", minimum=0) == 0:
        raise ValueError("Slurm query timeout must be positive")
    with TemporaryDirectory(prefix="slurm-job-", dir=root) as temporary:
        directory = Path(temporary) / "process"
        child = OwnedProcess(
            [
                "scontrol",
                f"--json={DATA_PARSER}",
                "--local",
                "show",
                "job",
                str(job_id),
            ],
            environment={**environment, "SLURM_JSON": "compact"},
            directory=directory,
            cwd=Path(temporary),
            log_limit_bytes=MAX_JOB_BYTES,
        )
        try:
            status = child.wait(timeout)
        finally:
            child.close(grace_seconds=0)
        if status:
            with (directory / "stderr.log").open("rb") as stream:
                detail = stream.read(4096).decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Slurm allocation query failed ({status}): {detail}")
        if child.logs[0].truncated:
            raise ValueError("Slurm allocation response was truncated or unreadable")
        with (directory / "stdout.log").open("rb") as stream:
            return parse_job(
                stream.read(MAX_JOB_BYTES + 1), job_id=job_id, user_id=os.getuid()
            )
