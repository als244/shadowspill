"""Physical tensor enumeration and checkpoint copy descriptors."""

from dataclasses import dataclass


def physical_state(model):
    """Enumerate ordinary tensors, including public tensor-wrapper components."""
    result = {}

    def add(name, value):
        flatten = getattr(value, "__tensor_flatten__", None)
        if flatten is None:
            result[name] = value
        else:
            for child in flatten()[0]:
                add(name + "." + child, getattr(value, child))

    for name, value in (*model.named_parameters(), *model.named_buffers()):
        add(name, value)
    return result


@dataclass(frozen=True)
class Copy:
    """``source`` into rows ``start:stop`` of ``destination`` (of its expert
    ``index``, for a stacked bank)."""

    destination: str
    source: str
    start: int | None = None
    stop: int | None = None
    reciprocal: bool = False
    index: int | None = None

    def view(self, tensors):
        value = tensors[self.destination]
        if self.index is not None:
            value = value[self.index]
        return value if self.start is None else value[self.start : self.stop]


def check_rows(destination, copies, shape):
    """The copies into ``destination`` cover each row (of each expert) once,
    in order."""
    stacked = copies[0].index is not None
    rows = shape[1] if stacked else shape[0]
    spans = {}
    for copy in copies:
        spans.setdefault(copy.index, []).append(copy)
    if stacked and sorted(spans) != list(range(shape[0])):
        raise ValueError(f"Incomplete experts for {destination}")
    for group in spans.values():
        position = 0
        for copy in group:
            start = 0 if copy.start is None else copy.start
            stop = rows if copy.stop is None else copy.stop
            if start != position:
                raise ValueError(f"Incomplete/overlapping rows for {destination}")
            position = stop
        if position != rows:
            raise ValueError(f"Incomplete rows for {destination}")
