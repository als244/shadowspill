"""NumPy-only whole/split update oracle and explicit native capture download."""

import importlib.abc
import json
import runpy
import socket
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import mkdtemp
from threading import Event
from time import monotonic

import numpy as np

from shadowspill_client import SSH, ServiceClient
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.runs import RunInfo
from shadowspill_client.protocol.training import ResolvedTraining


def main():
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, *args):
            if fullname.split(".")[0] in {
                "torch",
                "triton",
                "pynvml",
                "shadowspill",
                "shadowspill_workloads",
            }:
                raise AssertionError(
                    "engine/framework import in demo client: " + fullname
                )

    sys.meta_path.insert(0, Block())
    helper = runpy.run_path(str(Path(__file__).with_name("_rng_service_client.py")))
    Oracle, records, outputs = (
        helper[name] for name in ("Oracle", "records", "outputs")
    )
    raw = sys.stdin.buffer.read((1 << 20) + 1)
    assert len(raw) <= 1 << 20
    options = json.loads(raw)
    config = ResolvedTraining.from_dict(options["config"])
    connection = (
        {"ssh": SSH(options["ssh_host"], options["port"])}
        if options.get("ssh_host")
        else {"base_url": f"http://127.0.0.1:{options['port']}"}
    )
    memory = BufferBudget(64 << 20, 8)
    receipts, samples = [], []
    with ServiceClient(
        **connection, token=options["token"], session_id="nsys-session", timeout=30
    ) as client:
        for enabled in (False, True):
            settings = replace(
                config,
                execution=replace(
                    config.execution,
                    nsys={"start_step": 2, "steps": 2, "gpu_metrics": False}
                    if enabled
                    else None,
                ),
            )
            training = client.create_training_client(config=settings).result(180)
            oracle, observed, labels, latencies = Oracle(options["initial"]), [], [], []
            if enabled:
                assert training.nsys().result(30).record.status == "pending"

            def finish(
                future,
                *,
                temperature=1.0,
                backward=False,
                capture=False,
                oracle=oracle,
                observed=observed,
                labels=labels,
                latencies=latencies,
            ):
                started = monotonic()
                result = future.result(120)
                latencies.append(
                    {
                        "operation": result.operation,
                        "seconds": monotonic() - started,
                        "update_after": result.completed_updates,
                        "capture_active": capture,
                    }
                )
                assert (
                    result.state_valid
                    and not result.artifact_issues
                    and not result.traces
                )
                if result.outputs:
                    observed.append(
                        oracle.check(
                            outputs(result, memory),
                            result,
                            temperature=temperature,
                            backward=backward,
                        )
                    )
                if capture:
                    number = result.completed_updates + (
                        result.operation not in ("step", "optim_step")
                    )
                    labels.append(
                        f"train/update={number}/operation={result.operation}"
                        f"/request={result.identity.request_id}/rank=0"
                    )
                return result

            first = finish(training.step(records(), optim={"lr": 0.03}), backward=True)
            oracle.update(0.03, 1)
            assert first.completed_updates == 1
            masked = finish(training.step(records(masked=True)))
            assert masked.completed_updates == 1 and masked.outcome == "skipped"
            whole = finish(
                training.step(
                    records(),
                    optim={"lr": 0.04},
                    grad_scale=1.25,
                    loss_params={"temperature": 1.5},
                ),
                temperature=1.5,
                backward=True,
                capture=enabled,
            )
            oracle.update(0.04, 1.25)
            assert whole.completed_updates == 2
            finish(
                training.forward_backward(records(), loss_params={"temperature": 0.5}),
                temperature=0.5,
                backward=True,
                capture=enabled,
            )
            finish(
                training.forward(records(targets=False), outputs="predictions"),
                capture=enabled,
            )
            skipped = finish(
                training.forward_backward(records(masked=True)), capture=enabled
            )
            assert skipped.outcome == "skipped" and skipped.completed_updates == 2
            finish(training.forward_backward(records()), backward=True, capture=enabled)
            split = finish(
                training.optim_step(optim={"lr": 0.02}, grad_scale=-0.5),
                capture=enabled,
            )
            oracle.update(0.02, -0.5)
            assert split.completed_updates == 3
            after = finish(training.step(records(), optim={"lr": 0.01}), backward=True)
            oracle.update(0.01, 1)
            assert after.completed_updates == 4
            finish(training.forward(records(targets=False), outputs="predictions"))
            if enabled:
                assert training.nsys().result(30).record.status == "pending"
            training.close().result(180)
            deadline = monotonic() + 120
            while True:
                info = client._request(
                    "GET", f"/v1/runs/nsys-session/{training.run_id}", RunInfo
                )
                if info.state == "closed":
                    break
                assert monotonic() < deadline, info
                Event().wait(0.05)
            receipt = {
                "run_id": training.run_id,
                "captured": enabled,
                "expected_labels": labels,
                "latencies": latencies,
                "independent_loss_gradient_updates": True,
            }
            if enabled:
                deadline = monotonic() + 120
                while True:
                    capture = training.nsys().result(30)
                    if capture.record.status != "pending":
                        break
                    assert monotonic() < deadline
                    Event().wait(0.05)
                assert capture.record.status == "ready", capture.record
                window = capture.record.window
                assert window.completed_updates == 2 and window.state == "complete"
                assert window.history_id == whole.history_id == split.history_id
                assert window.first_request == whole.identity.request_id
                assert window.last_request == split.identity.request_id
                path = capture.report.download(
                    Path(mkdtemp(prefix="shadowspill-nsys-client-")) / "rank-0.nsys-rep"
                ).result(120)
                receipt.update(manifest=capture.record.to_dict(), download=str(path))
                with ServiceClient(
                    **connection, token=options["token"], session_id="nsys-session"
                ) as reconnected:
                    assert (
                        reconnected.get_nsys(training.run_id).result(30).record
                        == capture.record
                    )
            receipts.append(receipt)
            samples.append(observed)
        for before, captured in zip(*samples, strict=True):
            np.testing.assert_array_equal(before, captured)
    assert memory.used == client._memory.used == client._work.pending.used == (0, 0)
    assert not any(name in sys.modules for name in ("torch", "triton", "shadowspill"))
    print(
        json.dumps(
            {
                "runs": receipts,
                "client_host": socket.gethostname(),
                "numpy": np.__version__,
                "python": sys.version,
                "exact_training_and_eval_rng": True,
                "all_client_claims_released": True,
            }
        )
    )


if __name__ == "__main__":
    main()
