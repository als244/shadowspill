"""What counts as a refusal, and which refusals a store records as verdicts."""

from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.simulator import SimulationInfeasibleError

_INFEASIBLE = (PlanInfeasibleError, SimulationInfeasibleError)
_EXHAUSTED = (PlanSearchExhaustedError,)
# what a store records as a verdict, and raises again for the same question
_REFUSED = (*_INFEASIBLE, *_EXHAUSTED)
