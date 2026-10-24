"""Observe real Executor dispatch streams without inserting GPU work."""

import torch


class StreamChecks:
    def __init__(self, executor):
        self.callers = [torch.cuda.Stream(), torch.cuda.Stream()]
        self.observed = {}
        self.active = False
        self.expected = executor._streams
        for path in (*executor._paths, *executor._named_paths.values()):
            for bound in path.tasks:
                if bound.function is not None:
                    bound.function = self.wrap(bound.function, bound.task.resource)

    def wrap(self, function, resource):
        def observe(*args):
            if self.active:
                stream = torch.cuda.current_stream().cuda_stream
                assert self.observed.setdefault(resource, stream) == stream
            return function(*args)

        return observe

    def begin(self, index):
        self.observed.clear()
        self.active = True
        return self.callers[index % len(self.callers)]

    def check(self):
        self.active = False
        assert len(self.observed) >= 2
        assert len(set(self.observed.values())) == len(self.observed)
        caller = torch.cuda.current_stream().cuda_stream
        for resource, stream in self.observed.items():
            assert stream == self.expected[resource].cuda_stream
            assert stream != caller
