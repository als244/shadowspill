import json
from pathlib import Path
from types import SimpleNamespace

from baselines.common import write_json
from baselines.moe_layer import persistent


def test_oom_restarts_remaining_cases_without_repeating_success(tmp_path, monkeypatch):
    jobs = [
        {
            "case": f"sonicmoe/{i}",
            "attempt": str(tmp_path / f"case-{i}"),
            "python": "/python",
            "ranks": 8,
            "distributed_tuning": True,
            "arguments": [],
            "command": [],
        }
        for i in range(4)
    ]
    launches = []

    class Process:
        def __init__(self, command, **kwargs):
            queue_path = Path(command[-1])
            queue = json.loads(queue_path.read_text())
            launches.append([j["case"].split("/")[-1] for j in queue["jobs"]])
            self.returncode = 1 if len(launches) == 1 else 0
            completed = queue["jobs"][:1] if self.returncode else queue["jobs"]
            for job in completed:
                write_json(
                    Path(job["attempt"]) / "outcome.json",
                    {"status": "completed", "attempt": job["attempt"]},
                )
            if self.returncode:
                failed = queue["jobs"][1]
                write_json(queue_path.parent / "current.json", {**failed, "started": 0})
                write_json(
                    Path(failed["attempt"]) / "rank-00000/failure.json",
                    {"status": "out_of_memory", "stage": "warmup"},
                )

        def poll(self):
            return self.returncode

    monkeypatch.setattr(persistent, "emit_start", lambda *args: None)
    monkeypatch.setattr(persistent, "emit_finished", lambda *args: None)
    monkeypatch.setattr(persistent.subprocess, "Popen", Process)
    cases = []
    persistent.run(
        SimpleNamespace(output=tmp_path, cpu_threads=1, timeout=60), jobs, cases
    )
    assert launches == [["0", "1", "2", "3"], ["2", "3"]]
    assert [(c["case"].split("/")[-1], c["status"]) for c in cases] == [
        ("0", "completed"),
        ("1", "out_of_memory"),
        ("2", "completed"),
        ("3", "completed"),
    ]
    assert cases[1]["failure_stages"] == ["warmup"]


def test_native_nccl_oom_terminates_live_workers_and_continues(tmp_path, monkeypatch):
    jobs = [dict(case=f"ncclep/bf16/full/tokens-{t}", attempt=str(tmp_path/f"case-{t}"),
                 python="/python", ranks=8, distributed_tuning=False, arguments=[], command=[])
            for t in (65536, 2048)]
    launched=[]; stopped=[]
    class Process:
        def __init__(self, command, **kwargs):
            queue_path=Path(command[-1]); queue=json.loads(queue_path.read_text())
            launched.append(queue['jobs'])
            self.returncode=None if len(launched)==1 else 0
            if self.returncode is None:
                current=queue['jobs'][0]
                write_json(queue_path.parent/'current.json', {**current,'started':0})
                path=Path(current['attempt'])/'console.log';path.parent.mkdir(parents=True)
                path.write_text('NCCL EP: failed to allocate the HT intra-LSA staging buffer (50208.01 MiB)\nNCCL error 1 at nccl_ep.cc:1296\n')
            else:
                for job in queue['jobs']:
                    write_json(Path(job['attempt'])/'outcome.json', {'status':'completed','attempt':job['attempt']})
        def poll(self): return self.returncode
    def stop(process):
        stopped.append(process);process.returncode=-15
    monkeypatch.setattr(persistent,'stop',stop)
    monkeypatch.setattr(persistent,'emit_start',lambda *args:None)
    monkeypatch.setattr(persistent,'emit_finished',lambda *args:None)
    monkeypatch.setattr(persistent.subprocess,'Popen',Process)
    cases=[]
    persistent.run(SimpleNamespace(output=tmp_path,cpu_threads=1,timeout=60),jobs,cases)
    assert len(stopped)==1 and len(launched)==2
    assert [c['status'] for c in cases]==['out_of_memory','completed']
    assert cases[0]['failure_stages']==['native_dispatcher_setup']
    assert 'failed to allocate' in cases[0]['native_failure']['error']
