"""Traverse example/stage coordinates without building new computations."""

from collections.abc import Iterator

from shadowspill.step.ordering import StepDataOrdering


def traversal(
    ordering: StepDataOrdering, stages: int
) -> Iterator[tuple[str, int, int]]:
    head = stages - 1
    for pass_index in range(ordering.depth):
        positions = ordering.positions(pass_index)
        for stage in range(head if ordering.pair_loss else stages):
            for example in positions:
                yield "forward", example, stage
        if ordering.pair_loss:
            for example in positions:
                yield "forward", example, head
                yield "backward", example, head
        for stage in range(head - 1 if ordering.pair_loss else head, -1, -1):
            for example in ordering.backward_positions(pass_index):
                yield "backward", example, stage


def task_id(phase: str, example: int, stage: int, variant: str) -> str:
    return f"example_{example:04d}.stage_{stage:04d}.{variant}.{phase}"
