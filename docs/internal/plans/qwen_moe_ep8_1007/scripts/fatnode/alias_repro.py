"""Check mutation-result lifetime without a trainer, allocator, or collectives."""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.compilation.compiler import compile_artifact

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if int(os.environ.get("RANK", "0")) != 0:
    raise SystemExit(0)
args.out.mkdir(parents=True, exist_ok=True)
torch.cuda.set_device(0)
torch.set_num_threads(2)


def operation(weight, first, second, other_weight, other_first, gradient):
    padded = torch.nn.functional.pad(weight, (0, 3))
    owned = padded[:3].clone()
    updated = torch.ops.aten.addcdiv_.default(owned, first, second)
    scratch = -(gradient.abs() + 0.0001) / 0.025
    other = torch.ops.aten.addcdiv_.default(other_weight.clone(), other_first, scratch)
    return torch.cat((updated, other))


torch.manual_seed(42)
inputs = tuple(torch.randn(n, device="cuda") for n in (9, 3, 3, 12, 12, 12))
inputs[2].abs_().add_(1)
expected = operation(*inputs)
original_inputs = tuple(value.detach().cpu().clone() for value in inputs)
graph = make_fx(operation)(*inputs)
(args.out / "graph.py").write_text(graph.code)
artifact = GraphArtifact.capture(
    kind="optimizer", graph_module=graph, example_inputs=inputs
)
explicit = compile_artifact(artifact, device_ordinal=0).function
ordinary = torch.compile(operation, fullgraph=True)
rows = {}
for name, function in (("explicit", explicit), ("torch_compile", ordinary)):
    arguments = tuple(value.cuda() for value in original_inputs)
    actual = function(*arguments)
    delta = float((actual - expected).abs().max())
    rows[name] = {
        "maximum_error": delta,
        "actual": actual.tolist(),
        "input_mutations": [
            float((value.cpu() - original).abs().max())
            for value, original in zip(arguments, original_inputs, strict=True)
        ],
    }
    print(name, json.dumps(rows[name]), flush=True)
rows["expected"] = expected.tolist()
(args.out / "result.json").write_text(json.dumps(rows, indent=2) + "\n")
