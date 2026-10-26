"""Small CPU coordination records before workers enter ordered NCCL calls."""

from collections.abc import Callable

from shadowspill.errors import AdmissionError
from shadowspill.runtime import Runtime

from .worker import DistributedWorker


def coordinated[T](worker: DistributedWorker | None, function: Callable[[], T]) -> T:
    if worker is None:
        return function()
    result = None
    failure: Exception | None = None
    try:
        result = function()
    except Exception as error:
        failure = error
    try:
        worker.groups.check(None if failure is None else str(failure))
    except Exception as error:
        if failure is not None:
            raise error from failure
        raise
    return result  # type: ignore[return-value]


def check_invocation(
    worker: DistributedWorker,
    plan: str | None,
    accumulate: bool,
    parameter_norms: bool = False,
) -> None:
    choice = (plan, accumulate, parameter_norms)
    if any(value != choice for value in worker.groups.gather(choice)):
        raise ValueError(
            "workers selected different Plans, gradient accumulation modes "
            "or observations"
        )


def check_external_budget(runtime: Runtime) -> None:
    status = int(runtime._installed.library.shadowspill_pytorch_check_physical_budget())
    if status:
        raise AdmissionError(
            f"worker exceeded its physical memory reservation (status {status})"
        )
