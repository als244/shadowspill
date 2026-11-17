"""Cold worker for arbitrary nested inputs, unequal shapes and per-group controls."""

import sys
from pathlib import Path

import torch

from engine_tests.execution._cold import forbid_preparation

forbid_preparation()
from shadowspill import Executor, Plan  # noqa: E402
from shadowspill.training.observations import StepObservations  # noqa: E402

directory = Path(sys.argv[1])
oracle = torch.load(directory / "oracle.pt", weights_only=True)


def fill(request, chunk):
    value = oracle["initial"][request.name].reshape(-1)
    chunk.copy_(value[request.offset : request.offset + chunk.numel()])


def snapshot(executor):
    values = {}

    def collect(request, chunk):
        value = values.setdefault(
            request.name, torch.empty(request.storage_elements, dtype=request.dtype)
        )
        value[request.offset : request.offset + chunk.numel()].copy_(chunk)

    executor.export_tensors(collect, chunk_bytes=20)
    return values


with Executor(Plan.load(directory / "plan.ssp"), parameter_norms=True) as executor:
    executor.initialize(fill, chunk_bytes=20)
    actual = snapshot(executor)
    for name, value in oracle["initial"].items():
        torch.testing.assert_close(
            actual[name].reshape_as(value), value, rtol=0, atol=0
        )
    retained = []
    for index, calls in enumerate(oracle["examples"]):
        args, kwargs = [call[0] for call in calls], [call[1] for call in calls]
        # Validation must reject a nonexistent runtime control without an update.
        try:
            executor.run(args=args, kwargs=kwargs, hyperparams={"not_declared": 0.1})
        except ValueError as error:
            assert "unknown runtime hyperparameters" in str(error)
        else:
            raise AssertionError("unknown runtime control was accepted")
        raw = executor.run(
            args=args,
            kwargs=kwargs,
            hyperparams=oracle["controls"][index],
            collect_parameter_norms=True,
        ).result()
        result = StepObservations.collect(
            raw.losses, raw.metrics, raw.parameter_metrics
        )
        executor.synchronize()
        del raw
        expected = oracle["expected"][index]
        torch.testing.assert_close(torch.tensor(sum(result.losses)), expected["loss"])
        torch.testing.assert_close(result.metrics, expected["metrics"])
        torch.testing.assert_close(result.parameter_metrics, expected["norms"])
        assert all(
            value.device.type == "cpu" and not value.requires_grad
            for metrics in result.parameter_metrics.values()
            for value in metrics.values()
        )
        retained.append(result)
        actual = snapshot(executor)
        for name, value in expected["values"].items():
            torch.testing.assert_close(actual[name].reshape_as(value), value)
        for name, value in oracle["controls"][index].items():
            torch.testing.assert_close(actual[name], torch.tensor([value]))
    for result, expected in zip(retained, oracle["expected"], strict=True):
        torch.testing.assert_close(result.metrics, expected["metrics"])
        torch.testing.assert_close(result.parameter_metrics, expected["norms"])
assert "shadowspill.build" not in sys.modules
print("Qualified mixed-input Executor")
