"""One saved Plan, one worker, one outstanding asynchronous invocation."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

import torch

from shadowspill.artifacts.plan import Plan
from shadowspill.pytorch.invocation import InvocationResult, ReusableCompletionEvent
from shadowspill.runtime.plan import clear_tasks
from shadowspill.runtime.plan.lifecycle import abort_plan, adopt_plan, release_plan

from .admission import prepare
from .dispatch import dispatch
from .inputs import stage_inputs
from .results import collect_outputs
from .setup import open_worker
from .storage import Storage, TensorChunk


class Executor:
    def __init__(
        self,
        plan: Plan,
        *,
        device: int = 0,
        rank_map: Mapping[int, int] | None = None,
        external_headroom_bytes: int = 512 << 20,
    ) -> None:
        if rank_map is not None and dict(rank_map) != {0: 0}:
            raise NotImplementedError("this milestone supports a single logical rank")
        self.plan = plan
        self._pending: InvocationResult[Any] | None = None
        self._closed = False
        self._plan_released = False
        self._initialized = False
        self._failure: BaseException | None = None
        self.runtime, self._memory, self._bridge = open_worker(
            plan, device, external_headroom_bytes
        )
        try:
            self._storage = Storage(plan, self._bridge, torch.device("cuda", device))
            self._tasks, self._acquisition = prepare(plan, self._bridge)
            self._completion = ReusableCompletionEvent(
                self.runtime._runtime_handle, self._storage.device
            )
            adopt_plan(self.runtime, self._memory.plan_handle)
        except BaseException:
            self._bridge.objects.unregister(self._bridge.objects.registered_aliases())
            abort_plan(self.runtime, self._memory.plan_handle)
            self.runtime.close()
            raise

    def initialize(
        self,
        fill_tensor: Callable[[TensorChunk, torch.Tensor], None],
        *,
        chunk_bytes: int = 64 << 20,
    ) -> None:
        self.synchronize()
        self._storage.visit_state(fill_tensor, chunk_bytes, initialize=True)
        self._initialized = True

    def export_tensors(
        self,
        write_chunk: Callable[[TensorChunk, torch.Tensor], None],
        *,
        chunk_bytes: int = 64 << 20,
    ) -> None:
        self.synchronize()
        self._storage.visit_state(write_chunk, chunk_bytes, initialize=False)

    def run(
        self,
        *,
        args: Sequence[tuple[Any, ...]] | None = None,
        kwargs: Sequence[Mapping[str, Any]] | None = None,
        input_loader: Callable[[int], tuple[tuple[Any, ...], Mapping[str, Any]]]
        | None = None,
        hyperparams: Mapping[str, Any] | None = None,
        accumulate_gradients: bool = False,
    ) -> InvocationResult[Any]:
        self.synchronize()
        if not self._initialized:
            raise RuntimeError("initialize persistent tensors before run")
        if hyperparams:
            raise NotImplementedError(
                "runtime hyperparameter bindings are not implemented yet"
            )
        if accumulate_gradients:
            raise NotImplementedError(
                "cross-invocation gradient accumulation is not implemented yet"
            )
        self._bridge.require_initial_residency(self.plan.schedule.initial_residency)
        stage_inputs(self._storage, args, kwargs, input_loader)
        try:
            for task in self._tasks:
                dispatch(task, self._storage)
            payload = collect_outputs(self._storage, self._acquisition)
            wait = self._completion.record()

            def finish() -> None:
                wait()
                self._bridge.wait_until_idle()

            self._pending = InvocationResult(
                payload,
                finish,
                on_resolved=lambda result: None,
                on_failure=self._failed,
            )
            return self._pending
        except BaseException as error:
            self._failed(error)
            raise

    def _failed(self, error: BaseException) -> None:
        self._failure = error

    def synchronize(self) -> None:
        if self._closed or self._plan_released:
            raise RuntimeError("Executor is closed")
        if self._failure is not None:
            raise RuntimeError(
                "Executor is unusable after an execution failure"
            ) from self._failure
        if self._pending is not None:
            self._pending.result()
            self._pending = None

    def close(self) -> None:
        if self._closed:
            return
        if not self._plan_released:
            try:
                self.synchronize()
            finally:
                self._completion.release()
                self._storage.clear_views()
                clear_tasks(self._bridge)
                self._bridge.objects.unregister(
                    self._bridge.objects.registered_aliases()
                )
                release_plan(self.runtime, self._memory.plan_handle)
                self._plan_released = True
        # A caller-held output may prevent runtime shutdown. Allow close() to
        # be retried after it is released without destroying plan handles twice.
        self.runtime.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
