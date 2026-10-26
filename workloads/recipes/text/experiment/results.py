"""Small CPU observations returned from the execution subprocess as JSON."""

import json
from dataclasses import asdict

from shadowspill.training import _state
from shadowspill.training.observations import MetricSummary, MetricTable
from shadowspill.training.results import StepResult


def save(result, path):
    path.write_text(
        json.dumps(_state.encode(None if result is None else asdict(result))) + "\n"
    )


def load(path):
    values = _state.decode(json.loads(path.read_text()))
    if values is None:
        return None
    summary = values["summary"]
    values["summary"] = MetricSummary(
        summary["scalars"],
        {name: MetricTable(**table) for name, table in summary["tables"].items()},
    )
    return StepResult(**values)
