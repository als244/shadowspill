"""Record optimizer snapshots around the existing DP oracle test."""

import argparse
import inspect
import json
import os
from pathlib import Path

import torch

from tests.shadowspill.pytorch.distributed import _training_case as case

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
parser.add_argument("--optimizer", choices=("torch", "mlops"), default="torch")
parser.add_argument("--variant", choices=("auto", "save", "recompute"), default="save")
parser.add_argument("--sharded", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--parameter-metrics", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--symmetric-planning", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument("--masters", action="store_true")
args = parser.parse_args()
args.stochastic = args.diagnostics = False
directory = args.out / f"rank-{int(os.environ['RANK']):05d}"
directory.mkdir(parents=True, exist_ok=True)
snapshot = case.snapshot
assert_close = torch.testing.assert_close
count = 0


def recorded_snapshot(trainer, **kwargs):
    global count
    result = snapshot(trainer, **kwargs)
    torch.save(result, directory / f"snapshot-{count:03d}.pt")
    count += 1
    return result


def checked(actual, expected, **kwargs):
    try:
        return assert_close(actual, expected, **kwargs)
    except AssertionError:
        context = inspect.currentframe().f_back.f_locals
        details = {
            key: context.get(key)
            for key in ("step", "name", "key", "index")
            if isinstance(context.get(key), (str, int))
        }
        for key, value in (("actual", actual), ("expected", expected)):
            if isinstance(value, torch.Tensor):
                details[key] = value.detach().cpu().tolist()
        reference = context.get("reference")
        if reference is not None:
            torch.save(
                {
                    "model": reference.state_dict(),
                    "gradients": {
                        name: p.grad for name, p in reference.named_parameters()
                    },
                    "optimizer": context["reference_optimizer"].state_dict(),
                },
                directory / "reference.pt",
            )
        (directory / "mismatch.json").write_text(json.dumps(details, indent=2) + "\n")
        print("MISMATCH", json.dumps(details), flush=True)
        raise


case.snapshot = recorded_snapshot
torch.testing.assert_close = checked
case.run(args)
