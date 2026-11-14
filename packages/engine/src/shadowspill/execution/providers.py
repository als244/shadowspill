"""Worker ownership for process resources and task-bound operator binaries."""

from contextlib import ExitStack

import torch

from shadowspill.artifacts.plan import Plan
from shadowspill.preparation.lifecycle import LoadedPreparations
from shadowspill.resources.lifecycle import ResourceBindings


class WorkerProviders:
    def __init__(self, device: int):
        self.resources = ResourceBindings()
        self.preparations = LoadedPreparations(torch.device("cuda", device))

    def prepare(self, plans: tuple[Plan, ...]) -> None:
        for plan in plans:
            if not plan.program.bindings.get("operator_preparation"):
                continue
            selected = {
                task.task_id
                for task in plan._costed_program.selected_tasks(plan.selections)
            }
            keys = {
                task.profile_id
                for task in plan.program.tasks
                if task.task_id in selected
                and task.profile_id in plan.program.executables
            }
            self.preparations.add(plan.program, tuple(sorted(keys)))

    def close(self) -> None:
        # The worker has already drained invocation streams and removed callbacks.
        with ExitStack() as cleanup:
            cleanup.callback(self.resources.close)
            cleanup.callback(self.preparations.close)
