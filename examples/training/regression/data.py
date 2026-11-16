"""Caller-owned deterministic global batches, with no framework imports."""

import numpy as np

from shadowspill_client import Datum


class SyntheticBatches:
    def __init__(self, *, ranks, samples_per_rank=8, seed=711):
        self.ranks, self.samples_per_rank, self.seed = ranks, samples_per_rank, seed
        self.identity = {
            "generator": "numpy.PCG64.standard_normal.float32@1",
            "numpy": np.__version__,
            "ranks": ranks,
            "samples_per_rank": samples_per_rank,
            "seed": seed,
        }

    def __getitem__(self, index):
        if type(index) is not int or index < 0:
            raise IndexError("update index must be a nonnegative integer")
        batch = []
        for rank in range(self.ranks):
            generator = np.random.Generator(
                np.random.PCG64(self.seed + rank * 1_000_003 + index)
            )
            values = generator.standard_normal((self.samples_per_rank, 16), np.float32)
            targets = generator.standard_normal((self.samples_per_rank, 8), np.float32)
            batch.extend(
                Datum(
                    {"value": value},
                    {
                        "targets": target,
                        "weights": 1 / (self.ranks * self.samples_per_rank * 8),
                    },
                )
                for value, target in zip(values, targets, strict=True)
            )
        return batch
