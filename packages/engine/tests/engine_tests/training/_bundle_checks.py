"""Real saved bundle qualification with model frameworks and search forbidden."""

import sys
from dataclasses import replace
from pathlib import Path

for name in (
    "torch",
    "triton",
    "mlops",
    "shadowspill_workloads",
    "shadowspill.build",
    "shadowspill.profiling.measure",
    "shadowspill.planning.plan",
):
    sys.modules[name] = None

from engine_tests.training.test_setup_resolution import installed  # noqa: E402
from shadowspill.training.preparation.bundle import TrainingBundle  # noqa: E402

bundle = TrainingBundle.load(Path(sys.argv[1]))
prepared = bundle.recheck(installed())
assert len(prepared.plans[0]) == 14 and len(prepared.coverage) == 3
plans = prepared.plans[0]
backward_id = next(
    key for key, plan in plans.items() if plan._accumulation_plan is not None
)
backward = plans[backward_id]
for replacement, expected in (
    (replace(backward, _accumulation_plan=None), "accumulation path"),
    (
        replace(
            backward,
            profiled_program=replace(
                backward.profiled_program,
                program=replace(backward.program, name="changed saved graph"),
            ),
        ),
        "source operation",
    ),
):
    changed = replace(bundle, plans={0: {**plans, backward_id: replacement}})
    try:
        changed.recheck(installed())
    except ValueError as error:
        assert expected in str(error), str(error)
    else:
        raise AssertionError("changed saved Plan was accepted")
print("Qualified saved bundle CPU recheck")
