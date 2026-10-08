"""In-place normal initialization also used after meta materialization."""

from torch import nn


class Linear(nn.Linear):
    def __init__(self, input_width, output_width, *, std=0.02):
        self.initializer_range = std
        super().__init__(input_width, output_width, bias=False)

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=self.initializer_range)


class Embedding(nn.Embedding):
    def __init__(self, vocabulary, width, *, std=0.02):
        self.initializer_range = std
        super().__init__(vocabulary, width)

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=self.initializer_range)


class Conv1d(nn.Conv1d):
    def __init__(self, width, kernel, *, std=0.02):
        self.initializer_range = std
        super().__init__(width, width, kernel, groups=width, bias=False)

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=self.initializer_range)
