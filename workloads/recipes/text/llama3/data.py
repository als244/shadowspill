"""Adapt fixed text windows to ordinary per-example calls, with resume position."""

from workloads.recipes.text.fixed import TokenUpdates


class Inputs:
    def __init__(self, options, *, split="train"):
        self.data = TokenUpdates(
            options["data"],
            sequence_length=options["sequence_length"],
            sequences_per_step=options["sequences_per_step"],
            split=split,
        )
        self.size = options["sequences_per_example"]
        if self.size < 1 or self.data.sequences_per_step % self.size:
            raise ValueError("sequences_per_example must divide sequences_per_step")

    def arguments(self, update):
        args, kwargs = [], []
        for (tokens, targets, lengths), normalizer in self.data.microbatches(
            update, sequences_per_microbatch=self.size
        ):
            args.append((tokens, targets))
            kwargs.append({"seq_lens": lengths, "normalizer": normalizer})
        return {"args": args, "kwargs": kwargs}

    def example(self):
        return self.arguments(self.data.example())

    def __iter__(self):
        return self

    def __next__(self):
        return self.arguments(next(self.data))

    def state_dict(self):
        return {"data": self.data.state_dict(), "sequences_per_example": self.size}

    def load_state_dict(self, state):
        if state["sequences_per_example"] != self.size:
            raise ValueError("checkpoint example geometry differs")
        self.data.load_state_dict(state["data"])
