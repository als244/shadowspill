"""Stock MoonEP reproduction: torchrun --standalone --nproc-per-node=2 this.py.

Only PyTorch and MoonEP are imported. Synthetic routing is independent of any
model or router implementation. No package patch or compiler override is applied.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from moonep import Buffer
from moonep import planning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--experts", type=int, default=192)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--check-roundtrip", action="store_true")
    parser.add_argument("--exact-payload", action="store_true",
                        help="Use multiples of 1/16 whose partial K-sums are exactly BF16 representable")
    parser.add_argument("--operation", choices=("dispatch", "planning"), default="dispatch")
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--compiler-options")
    parser.add_argument("--memory-clobber", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group(args.backend, timeout=timedelta(seconds=120))
    device = torch.device("cuda", local)
    args.outdir.mkdir(parents=True, exist_ok=True)
    events = []

    def record(phase, **values):
        event = dict(utc=datetime.now(UTC).isoformat(), rank=rank, phase=phase, **values)
        events.append(event)
        (args.outdir / f"rank-{rank:05d}.json").write_text(json.dumps(events, indent=2) + "\n")
        print(json.dumps(event), flush=True)

    unexpected = [name for name in sys.modules if name.split(".")[0] in ("mlops", "shadowspill")]
    assert not unexpected, unexpected
    assert not getattr(planning, "_moe_rank1_patched", False)
    record("stock_import", torch=torch.__version__, cuda=torch.version.cuda,
           cutlass_dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
           moon_source=importlib.metadata.distribution("moonep").read_text("direct_url.json"),
           planning_sha256=hashlib.sha256(Path(planning.__file__).read_bytes()).hexdigest(),
           imported_mlops_or_shadowspill=unexpected)
    if args.compiler_options or args.memory_clobber:
        import probe_moonep_compat
        if args.compiler_options:
            probe_moonep_compat.compiler_options(args.compiler_options)
        if args.memory_clobber:
            probe_moonep_compat.memory_clobber()
        record("compiler_experiment", compiler_options=args.compiler_options,
               memory_clobber=args.memory_clobber)
    buffer = Buffer(S=args.tokens, H=args.dim, K=args.top_k, E=args.experts,
                    num_ep_ranks=dist.get_world_size(), group=dist.group.WORLD,
                    enable_pdl=False, explicitly_destroy=True)
    torch.manual_seed(4100 + rank)
    ids = torch.randn(args.tokens, args.experts, device=device).topk(args.top_k, dim=-1).indices.to(torch.int32)
    histogram = torch.bincount(ids.flatten().long(), minlength=args.experts).to(torch.int32)
    x = torch.randn(args.tokens, args.dim, device=device, dtype=torch.bfloat16)
    if args.exact_payload:
        assert args.top_k <= 8
        x = torch.randint(-16, 17, x.shape, device=device, dtype=torch.int16).to(torch.bfloat16) / 16
    weights = torch.full((args.tokens, args.top_k), 1.0 / args.top_k, device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    record("inputs_valid", tokens=args.tokens, assignments=int(histogram.sum()),
           expert_min=int(ids.min()), expert_max=int(ids.max()),
           operation=args.operation, experts=args.experts, top_k=args.top_k,
           dim=args.dim, world_size=dist.get_world_size(), iterations=args.iterations,
           exact_payload=args.exact_payload)
    dist.barrier()
    for iteration in range(args.iterations):
        record("operation_begin", iteration=iteration)
        if args.operation == "planning":
            ctx = buffer._require_ctx()
            plan, ends = planning.allocate_planning_outputs(ctx)
            planning.launch_planning(ctx, ids.flatten(), histogram, ends, plan)
        else:
            received, probabilities, ends, plan = buffer.dispatch(x, weights, ids, histogram)
        torch.cuda.synchronize()
        record("operation_end", iteration=iteration, max_padded_rows=int(ends[-1]))
        if args.check_roundtrip:
            assert args.operation == "dispatch", "Roundtrip requires dispatch"
            combined, returned_weights, _ = buffer.combine(
                plan=plan, hidden_nvsh=received, route_weights_nvs=probabilities)
            expected = (x.float() * args.top_k).to(x.dtype)
            torch.testing.assert_close(combined, expected, rtol=0, atol=0)
            torch.testing.assert_close(returned_weights, weights, rtol=0, atol=0)
            del received, probabilities, combined, returned_weights, expected
            # The same saved-plan operations used in backward, with new values.
            dy = -x
            received_dy, _, _, _ = buffer.dispatch(dy, plan=plan)
            dx, _, _ = buffer.combine(plan=plan, hidden_nvsh=received_dy)
            torch.testing.assert_close(dx, (dy.float() * args.top_k).to(dy.dtype), rtol=0, atol=0)
            del dy, received_dy, dx
            record("roundtrip_passed", iteration=iteration, exact=True,
                   saved_plan_backward=True)
        elif args.operation == "dispatch":
            del received, probabilities
        del ends, plan
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()
    record("completed", passed=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # A poisoned CUDA context can fail again in Buffer/NCCL destructors and
        # obscure the original exception. Preserve it before process teardown.
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
