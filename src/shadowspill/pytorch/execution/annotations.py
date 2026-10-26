"""Profiler and completion methods retained for older task executors."""

from collections.abc import Callable

from shadowspill.pytorch.invocation import ReusableCompletionEvent
from shadowspill.runtime.plan import RuntimeBridge
from shadowspill.runtime.plan.annotations import TaskBoundaryAnnotations


class AnnotatedExecutor:
    """What every task executor does about annotations and completion.

    Both executors own a `TaskBoundaryAnnotations`, a `RuntimeBridge` and a
    completion recorder, and do the same three things with them. The
    behaviour belongs to the policy, not to forward or training.
    """

    _task_annotations: TaskBoundaryAnnotations
    _bridge: RuntimeBridge
    _completion: ReusableCompletionEvent

    def set_profiler_annotations(self, enabled: bool) -> None:
        """Toggle provider annotations for this callable."""

        self._task_annotations.set_enabled(enabled)

    def finish_profiler_annotations(self) -> None:
        """Drain annotated asynchronous work before disabling its provider."""

        if not self._task_annotations.enabled:
            return
        self._bridge.wait_until_idle()
        self.set_profiler_annotations(False)

    def record_invocation_completion(self) -> Callable[[], None]:
        """Record the explicit public-result completion boundary."""

        return self._completion.record()
