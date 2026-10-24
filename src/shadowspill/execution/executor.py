"""One saved Plan, one worker, one outstanding asynchronous invocation."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

import torch

from shadowspill.artifacts.plan import Plan
from shadowspill.pytorch.invocation import InvocationResult, ReusableCompletionEvent

from .dispatch import dispatch
from .inputs import stage_inputs
from .results import collect_outputs
from .tensors import TensorChunk, visit_tensors
from .worker import create_paths


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
        self.device = torch.device("cuda", device)
        self._pending: InvocationResult[Any] | None = None
        self._closed = False
        self._plan_released = False
        self._initialized = False
        self._failure: BaseException | None = None
        self.runtime, self._paths = create_paths(plan, device, external_headroom_bytes)
        self._completion = ReusableCompletionEvent(
            self.runtime._runtime_handle, torch.device("cuda", device)
        )

    def initialize(
        self,
        fill_tensor: Callable[[TensorChunk, torch.Tensor], None],
        *,
        chunk_bytes: int = 64 << 20,
        restore: bool = False,
    ) -> None:
        self.synchronize()
        self._initialized = False
        path = self._paths[0]
        visit_tensors(
            path.storage, fill_tensor, chunk_bytes, initialize=True, restore=restore
        )
        path.controls.initialize(fill_tensor, chunk_bytes=chunk_bytes, restore=restore)
        self._initialized = True

    def export_tensors(
        self,
        write_chunk: Callable[[TensorChunk, torch.Tensor], None],
        *,
        chunk_bytes: int = 64 << 20,
    ) -> None:
        self.synchronize()
        if not self._initialized:
            raise RuntimeError("initialize persistent tensors before exporting them")
        visit_tensors(
            self._paths[0].storage, write_chunk, chunk_bytes, initialize=False
        )
        self._paths[0].controls.export(write_chunk, chunk_bytes=chunk_bytes)

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
        if accumulate_gradients and len(self._paths) != 2:
            raise ValueError("this Plan has no gradient-accumulation invocation path")
        path = self._paths[int(accumulate_gradients)]
        path.bridge.require_initial_residency(path.plan.schedule.initial_residency)
        path.controls.apply(hyperparams or {})
        stage_inputs(path.storage, args, kwargs, input_loader)
        try:
            for task in path.tasks:
                dispatch(task, path.storage, path.controls)
            payload = collect_outputs(path.storage, path.acquisition)
            wait = self._completion.record()

            def finish() -> None:
                wait()
                path.bridge.wait_until_idle()

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
                for path in reversed(self._paths):
                    path.close(self.runtime)
                self._plan_released = True
        # A caller-held output may prevent runtime shutdown. Allow close() to
        # be retried after it is released without destroying plan handles twice.
        self.runtime.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            exc.add_note(f"Executor cleanup also failed: {cleanup}")
