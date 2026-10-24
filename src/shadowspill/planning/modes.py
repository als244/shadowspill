"""Both gradient invocation paths share one serialized Plan and one GPU arena."""

from dataclasses import replace

from shadowspill.artifacts.plan import Plan
from shadowspill.errors import PlanInfeasibleError


def bundle_gradient_modes(fresh: Plan, accumulated: Plan) -> Plan:
    plans = [fresh, accumulated]
    if all(p.memory_layout is not None for p in plans):
        # The reusable arena must fit either path. Expanding an arena adds no
        # new placements, but its reserved bytes count even on the smaller path.
        span = max(
            p.memory_layout.fixed_slice_bytes
            for p in plans
            if p.memory_layout is not None
        )
        for index, plan in enumerate(plans):
            layout = plan.memory_layout
            assert layout is not None
            required = span + layout.required_bytes - layout.fixed_slice_bytes
            capacity = (
                plan.memory_budgets.execution_bytes
                - plan.profiled_program.fixed_device_bytes
            )
            if required > capacity:
                raise PlanInfeasibleError(
                    "shared gradient-mode arena exceeds the GPU allowance",
                    kind="physical_layout",
                    required_bytes=required,
                    capacity_bytes=capacity,
                )
            plans[index] = replace(
                plan,
                memory_layout=replace(
                    layout,
                    fixed_slice_bytes=span,
                    required_bytes=required,
                ),
            )
    return replace(plans[0], _accumulation_plan=plans[1])
