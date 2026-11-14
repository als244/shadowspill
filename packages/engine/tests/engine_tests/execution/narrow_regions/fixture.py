"""Actual FP16 storage through generic regions, independent DP references.

CPU readiness precedes any device allocation. No model/provider math or existing
numerical criteria are changed. These cases complement the FP32 DP fixtures.
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

CASES = {
    "full_save": dict(
        kind="step",
        variant="save",
        lora=False,
        storage="replicated",
        nested=True,
        checkpoint_cuts=False,
        retain_sequence=False,
        lifetime_mode="replay_compact",
        batch=2,
        sequence=17,
    ),
    "lora_retained": dict(
        kind="step",
        variant="recompute",
        lora=True,
        storage="sharded",
        nested=True,
        checkpoint_cuts=True,
        retain_sequence=True,
        lifetime_mode="replay_compact",
        batch=2,
        sequence=17,
    ),
    "lora_batch_accumulating": dict(
        kind="forward_backward",
        variant="recompute",
        lora=True,
        storage="sharded",
        nested=True,
        checkpoint_cuts=False,
        retain_sequence=True,
        lifetime_mode="replay_compact",
        chunk_axis="batch",
        batch=13,
        sequence=17,
    ),
}


def write(directory, name, value):
    (directory / name).write_text(json.dumps(value, indent=2) + "\n")


def model_for(case, *, device="cpu", dtype=None):
    import torch

    from engine_tests.execution._distributed_regions import Network

    torch.manual_seed(731)
    return Network(
        case["lora"],
        device=device,
        dtype=dtype or torch.float16,
        lifetime_mode=case["lifetime_mode"],
    )


def calls_for(case, step):
    import torch

    from engine_tests.execution._distributed_regions import examples

    calls = examples(case["batch"], case["sequence"])
    for local in calls:
        for index, (args, kwargs) in enumerate(local):
            # Vary real inputs across updates, preserving global mask weights.
            local[index] = (
                (torch.roll(args[0], step, dims=1).to(torch.float16),),
                kwargs,
            )
    assert all(not torch.equal(calls[0][i][0][0], calls[1][i][0][0]) for i in range(2))
    assert (
        abs(
            sum(
                float(kw["sample_weights"].sum()) * 7
                for local in calls
                for _, kw in local
            )
            - 1
        )
        < 1e-6
    )
    return calls


def prepare(directory, case):
    import torch

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    model = model_for(case)
    assert all(p.dtype == torch.float16 for p in model.parameters())
    assert model.blocks[0].out is model.blocks[1].out
    assert model.blocks[0].b is model.blocks[1].b
    assert all(p.count_nonzero() for p in model.parameters())
    torch.save(
        dict(
            initial={n: p.detach().clone() for n, p in model.named_parameters()},
            trainable=[n for n, p in model.named_parameters() if p.requires_grad],
            frozen=[n for n, p in model.named_parameters() if not p.requires_grad],
            examples=[calls_for(case, step) for step in range(3)],
        ),
        directory / "inputs.pt",
    )


def preflight(directory, case):
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx

    from engine_tests.execution._distributed_regions import (
        Block,
        check_batch_chunk_metadata,
        check_lifetime_reports,
        policies,
    )
    from engine_tests.pytorch.chunking.test_composition import executable
    from qualification.numerical.metrics import tensor_metrics
    from qualification.numerical.tolerances import meets_tensor_tolerance
    from shadowspill.pytorch.capture.aot import capture_graph_pair
    from shadowspill.pytorch.chunking.regions import region_artifact

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    results = []
    for dtype in (torch.float64, torch.float16):
        torch.manual_seed(731)
        block = Block(case["lora"], dtype=dtype, lifetime_mode=case["lifetime_mode"])
        weights = dict(block.named_parameters())
        inputs = (
            torch.randn(
                case["batch"], case["sequence"], 7, dtype=dtype, requires_grad=True
            ),
            *weights.values(),
        )

        def layer(x, *values, block=block, weights=weights):
            return torch.func.functional_call(
                block, dict(zip(weights, values, strict=True)), (x,)
            )

        expected = layer(*inputs)
        tangent = torch.randn_like(expected)
        gradients = torch.autograd.grad(
            expected, [v for v in inputs if v.requires_grad], tangent
        )
        controls = policies(case)
        policy = getattr(controls["graph_pair_policy"], case["variant"])
        pair = capture_graph_pair(
            make_fx(layer)(*inputs),
            inputs,
            original_output=layer(*inputs),
            memory_budget=float(case["variant"] == "save"),
            retention=policy,
        )
        forward, backward = [
            region_artifact(
                a, controls["chunking"], stage_id="block", variant=case["variant"]
            )
            for a in (pair.forward, pair.backward)
        ]
        assert forward.storage_contract == pair.forward.storage_contract
        assert backward.storage_contract == pair.backward.storage_contract
        reports = [a.regions.lifetime_report for a in (forward, backward)]
        if case["variant"] == "recompute":
            lifetime = check_lifetime_reports(reports, case["lifetime_mode"])
        else:
            # The cost-SAVE cut can already remove the relevant retained view.
            # Record its actual effects; demand compaction in the RECOMPUTE
            # controls where a larger live backing really crosses the barrier.
            lifetime = {
                kind: sum(
                    d["applied"]
                    for r in reports
                    for d in (r.get(kind) or {}).get("decisions", ())
                )
                for kind in ("schedule", "compaction", "replay")
            }
            assert lifetime["replay"] > 0
            assert all(
                after <= before
                for r in reports
                for kind in ("schedule", "compaction", "replay")
                for _, before, after in (r.get(kind) or {}).get("barrier_bytes", ())
            )
        if case.get("chunk_axis") == "batch":
            check_batch_chunk_metadata(
                [a.regions.identity for a in (forward, backward)]
            )
        selector = controls["chunking"].lifetimes.select
        selector.building = False
        before = selector.calls
        with torch.no_grad():
            actual = executable(forward)(*inputs)
            saved = actual[-pair.saved_value_count :] if pair.saved_value_count else ()
            actual_gradients = tuple(
                v for v in executable(backward)(*saved, tangent) if v is not None
            )
        assert selector.calls == before
        comparisons = []
        for name, reference, value in [
            ("output", expected, actual[0]),
            *[
                (f"gradient-{i}", a, b)
                for i, (a, b) in enumerate(
                    zip(gradients, actual_gradients, strict=True)
                )
            ],
        ]:
            metric = tensor_metrics(reference, value)
            passed = meets_tensor_tolerance(metric, key="state/gradient/" + name)
            if dtype == torch.float64:
                torch.testing.assert_close(value, reference, rtol=1e-10, atol=1e-10)
            comparisons.append(dict(name=name, passed=passed, **asdict(metric)))
        results.append(
            dict(
                dtype=str(dtype),
                lifetime_decisions=lifetime,
                comparisons=comparisons,
                saved_values=pair.saved_value_count,
            )
        )
        write(
            directory,
            "capture-preflight.json",
            dict(cases=results, cuda_initialized=torch.cuda.is_initialized()),
        )
        assert all(r["passed"] for r in comparisons), comparisons
    write(
        directory,
        "preflight-ready.json",
        dict(
            status="PASS_FP64_AND_ACTUAL_FP16_REGION_VJPS",
            cases=results,
            cuda_initialized=False,
            limits=(
                "CPU reference/capture only; no compilation, physical admission "
                "or distributed device execution yet."
            ),
        ),
    )


if __name__ == "__main__":
    import sys

    directory = Path(sys.argv[1])
    case = json.loads((directory / "case.json").read_text())
    {"prepare": prepare, "preflight": preflight}[sys.argv[2]](directory, case)
