"""Scheduler memory comes from actual node allocations, not CPU-count arithmetic."""

import json
import os
import subprocess
import sys

import pytest

from shadowspill.orchestration.launch import slurm_job
from shadowspill.orchestration.launch.slurm_job import DATA_PARSER, parse_job, query_job


def snapshot():
    return {
        "meta": {"plugin": {"data_parser": DATA_PARSER}},
        "errors": [],
        "jobs": [
            {
                "job_id": 123,
                "user_id": os.getuid(),
                "job_state": ["RUNNING"],
                "memory_per_cpu": {"set": True, "infinite": False, "number": 100},
                "job_resources": {
                    "cpus": 10,
                    "nodes": {
                        "count": 2,
                        "allocation": [
                            {
                                "index": 0,
                                "name": "node-a",
                                "cpus": {"count": 6},
                                "memory": {"allocated": 300},
                            },
                            {
                                "index": 1,
                                "name": "node-b",
                                "cpus": {"count": 4},
                                "memory": {"allocated": 200},
                            },
                        ],
                    },
                },
            }
        ],
    }


def parse(value):
    return parse_job(json.dumps(value).encode(), job_id=123, user_id=os.getuid())


def test_heterogeneous_node_memory_uses_assigned_mib_not_hardware_thread_count():
    value = snapshot()
    job = parse(value)
    assert job.nodes["node-a"].cpus == 6
    assert job.nodes["node-a"].memory_bytes == 300 << 20
    assert job.nodes["node-b"].memory_bytes == 200 << 20
    value["jobs"][0]["job_resources"]["nodes"]["allocation"].clear()
    assert len(job.nodes) == 2
    with pytest.raises(TypeError):
        job.nodes["new"] = job.nodes["node-a"]


@pytest.mark.parametrize(
    "fault",
    [
        "parser",
        "errors",
        "missing",
        "multiple",
        "job",
        "owner",
        "state",
        "count",
        "index",
        "duplicate",
        "memory",
        "boolean",
        "cpus",
    ],
)
def test_incomplete_wrong_owner_and_ambiguous_facts_fail_closed(fault):
    value = snapshot()
    job = value["jobs"][0]
    nodes = job["job_resources"]["nodes"]
    if fault == "parser":
        value["meta"]["plugin"]["data_parser"] = "data_parser/v0.0.45"
    elif fault == "errors":
        value["errors"] = [{"error": "missing job"}]
    elif fault == "missing":
        del job["job_resources"]
    elif fault == "multiple":
        value["jobs"].append(job)
    elif fault == "job":
        job["job_id"] += 1
    elif fault == "owner":
        job["user_id"] += 1
    elif fault == "state":
        job["job_state"] = ["RUNNING", "COMPLETING"]
    elif fault == "count":
        nodes["count"] = 3
    elif fault == "index":
        nodes["allocation"][1]["index"] = 0
    elif fault == "duplicate":
        nodes["allocation"][1]["name"] = "node-a"
    elif fault in ("memory", "boolean"):
        nodes["allocation"][0]["memory"]["allocated"] = 0 if fault == "memory" else True
    else:
        job["job_resources"]["cpus"] += 1
    with pytest.raises(ValueError):
        parse(value)


@pytest.mark.parametrize(
    "payload", [b'{"jobs":[],"jobs":[]}', b'{"jobs":NaN}', b'{"jobs":', b"[]"]
)
def test_strict_bounded_json(payload, monkeypatch):
    with pytest.raises(ValueError):
        parse_job(payload, job_id=123, user_id=os.getuid())
    monkeypatch.setattr(slurm_job, "MAX_JOB_BYTES", 4)
    with pytest.raises(ValueError, match="byte bound"):
        parse_job(b" " * 5, job_id=123, user_id=os.getuid())


@pytest.mark.parametrize("fault", [None, "exit", "timeout", "large"])
def test_owned_read_only_query_has_bounded_output_deadline_and_cleanup(
    tmp_path, monkeypatch, fault
):
    binary, response, arguments = (
        tmp_path / "scontrol",
        tmp_path / "job.json",
        tmp_path / "args.json",
    )
    response.write_text(json.dumps(snapshot()))
    binary.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, sys, time
from pathlib import Path
Path(os.environ['QUERY_ARGS']).write_text(json.dumps(sys.argv[1:]))
assert os.environ['SLURM_JSON'] == 'compact'
fault = os.environ.get('QUERY_FAULT')
if fault == 'timeout':
    time.sleep(60)
if fault == 'exit':
    sys.stderr.write('scheduler unavailable')
    raise SystemExit(1)
data = 'x' * 5000 if fault == 'large' else Path(os.environ['QUERY_DATA']).read_text()
sys.stdout.write(data)
"""
    )
    binary.chmod(0o700)
    environment = {
        **os.environ,
        "PATH": str(tmp_path),
        "QUERY_ARGS": str(arguments),
        "QUERY_DATA": str(response),
        "QUERY_FAULT": fault or "",
    }
    monkeypatch.setattr(slurm_job, "MAX_JOB_BYTES", 4096)
    if fault:
        with pytest.raises((ValueError, RuntimeError, subprocess.TimeoutExpired)):
            query_job(123, environment=environment, root=tmp_path, timeout=0.2)
    else:
        assert (
            query_job(123, environment=environment, root=tmp_path)
            .nodes["node-a"]
            .memory_bytes
            == 300 << 20
        )
    assert json.loads(arguments.read_text()) == [
        f"--json={DATA_PARSER}",
        "--local",
        "show",
        "job",
        "123",
    ]
    assert not list(tmp_path.glob("slurm-job-*"))
