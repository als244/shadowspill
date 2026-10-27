"""Fresh worker: load an executable Plan, initialize tensors, run the gate."""

import json
import sys
from pathlib import Path

from qualification.pipeline.observe import Observe
from qualification.pipeline.state import initialize, load_inputs
from qualification.runtime_evidence import check_physical_budget
from shadowspill import Executor, Plan
from shadowspill.runtime.plan.annotations import set_profiler_annotations

from .baseline import regression_comparison
from .phases import measure_groups, optimizer_steps, warm_step
from .prepare import read_request
from .verdict import measured_result


def run(directory: Path) -> None:
    arguments, manifest = read_request(directory)
    plan = Plan.load(directory / "plan.ssp")
    inputs = load_inputs(directory)
    regression = regression_comparison(
        manifest, arguments, device_name=plan.program.setup["gpu"]
    )
    print("THROUGHPUT BASELINE: " + regression.reason, flush=True)
    with Executor(
        plan, external_headroom_bytes=manifest.external_headroom_bytes
    ) as executor:
        if arguments.profiler_annotations:
            set_profiler_annotations(executor._paths[0].bridge, True)
        initialize(executor, inputs)
        observer = Observe(executor, steps=arguments.steps_per_group)
        statuses = [check_physical_budget()]
        try:
            warm = warm_step(
                executor, observer, inputs["args"], arguments, directory, statuses
            )
            measured = measure_groups(
                executor, observer, inputs["args"], manifest, arguments, statuses
            )
            result = measured_result(
                manifest,
                arguments,
                plan,
                warm,
                measured,
                runtime=executor.runtime,
                logical_steps=optimizer_steps(executor),
                physical_statuses=statuses,
                regression=regression,
                directory=directory,
            )
        finally:
            observer.close()
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    run(Path(sys.argv[1]))
