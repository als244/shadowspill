"""Worker-only CUDA profiler boundaries shared by execution adapters."""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Literal

import torch
import torch.distributed as dist

from shadowspill_client.protocol.captures import NsightWindow

from .nsys import NsysCapture


def capture_boundary(action: Callable[[], object]) -> None:
    distributed = dist.is_initialized()
    if distributed:
        dist.barrier()
    action()
    if distributed:
        dist.barrier()


class UpdateCapture:
    """One profiler interval over complete updates, including split contributions.

    The execution owner supplies the existing update number and completion fact.
    This object neither advances training state nor decides whether an update ran.
    After completion, interruption or a history change it never restarts capture.
    GPU errors propagate to the execution owner; metadata is inspected separately.
    """

    def __init__(
        self,
        settings: NsysCapture,
        *,
        publish: Callable[[NsightWindow], None] | None = None,
    ) -> None:
        self.settings = settings
        self.state: Literal[
            "pending", "running", "complete", "incomplete", "missing"
        ] = "pending"
        self.history_id: str | None = None
        self.first_request: str | None = None
        self.last_request: str | None = None
        self.completed_updates = 0
        self.error: str | None = None
        self._active = False
        self._publish = publish
        self.publication_error: str | None = None
        self._changed()

    @property
    def progress(self) -> NsightWindow:
        return NsightWindow(
            self.settings.start_step,
            self.settings.steps,
            self.state,
            self.completed_updates,
            self.history_id,
            self.first_request,
            self.last_request,
            self.error,
        )

    def _changed(self) -> None:
        if self._publish is not None:
            try:
                self._publish(self.progress)
            except Exception as error:
                # Optional host metadata cannot undo a completed update or retain
                # a callback traceback. Missing/stale evidence fails finalization.
                self.publication_error = (str(error).strip() or type(error).__name__)[
                    :1024
                ].rstrip()

    @property
    def active(self) -> bool:
        return self._active

    def begin(
        self,
        number: int,
        *,
        can_start: bool,
        request_id: str,
        history_id: str | None = None,
    ) -> bool:
        if self.state not in ("pending", "running"):
            return False
        if self._active and history_id != self.history_id:
            self.interrupt("Training history changed during the capture window.")
            return False
        if number < self.settings.start_step:
            if self._active:
                self.interrupt("Update counter moved before the capture window.")
            return False
        if number > self.settings.start_step + self.settings.steps - 1:
            self.interrupt("The selected update window was not completed.")
            return False
        if not self._active:
            if not can_start:
                return False
            if number != self.settings.start_step:
                self.interrupt("The first selected update was not captured.")
                return False
            capture_boundary(torch.cuda.profiler.start)
            self._active = True
            self.state = "running"
            self.history_id, self.first_request = history_id, request_id
            self._changed()
        return True

    def annotation(self, label: str) -> AbstractContextManager[object]:
        return torch.cuda.nvtx.range(label) if self._active else nullcontext()

    def complete(self, number: int, *, request_id: str) -> None:
        if not self._active:
            return
        if number != self.settings.start_step + self.completed_updates:
            self.interrupt("Captured update completions were not contiguous.")
            return
        self.completed_updates += 1
        self.last_request = request_id
        if self.completed_updates == self.settings.steps:
            capture_boundary(torch.cuda.profiler.stop)
            self._active = False
            self.state = "complete"
        self._changed()

    def interrupt(self, message: str) -> None:
        """End an incomplete window at a known, coordinated execution boundary."""
        if self.state in ("complete", "incomplete", "missing"):
            return
        if self._active:
            capture_boundary(torch.cuda.profiler.stop)
            self._active = False
        self.state = "incomplete" if self.first_request is not None else "missing"
        self.error = message[:1024]
        self._changed()

    def failed(self, message: str) -> None:
        """An uncertain worker cannot enter another distributed capture barrier."""
        if self.state != "complete":
            self.state = "incomplete"
            self.error = message[:1024]
            self._active = False
            self._changed()
