"""Importable mixed-shape factory for the independent quickstart workers."""

import torch

from workloads.recipes.regression import experiment as regression


def mixed_examples(*, device, rows=8, width=8, outputs=8):
    setup = regression(device=device, rows=rows, width=width, outputs=outputs)
    source = setup["candidates"][f"rows_{rows // 2}"]
    features = torch.cat([args[0] for args in source["args"]])
    targets = torch.cat([kwargs["targets"] for kwargs in source["kwargs"]])
    setup["candidates"] = {
        f"split_{split}": {
            "args": [(features[:split],), (features[split:],)],
            "kwargs": [{"targets": targets[:split]}, {"targets": targets[split:]}],
        }
        for split in (rows // 4, rows // 2 - 1)
    }
    return setup
