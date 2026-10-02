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
    buffer = Buffer(S=args.tokens, H=1024, K=4, E=192,
                    num_ep_ranks=dist.get_world_size(), group=dist.group.WORLD,
                    enable_pdl=False, explicitly_destroy=True)
    torch.manual_seed(4100 + rank)
    ids = torch.randn(args.tokens, 192, device=device).topk(4, dim=-1).indices.to(torch.int32)
    histogram = torch.bincount(ids.flatten().long(), minlength=192).to(torch.int32)
    x = torch.randn(args.tokens, 1024, device=device, dtype=torch.bfloat16)
    weights = torch.full((args.tokens, 4), 0.25, device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    record("inputs_valid", tokens=args.tokens, assignments=int(histogram.sum()),
           expert_min=int(ids.min()), expert_max=int(ids.max()),
           operation=args.operation)
    dist.barrier()
    record("operation_begin")
    if args.operation == "planning":
        ctx = buffer._require_ctx()
        plan, ends = planning.allocate_planning_outputs(ctx)
        planning.launch_planning(ctx, ids.flatten(), histogram, ends, plan)
    else:
        received, probabilities, ends, plan = buffer.dispatch(x, weights, ids, histogram)
    torch.cuda.synchronize()
    record("operation_end", max_padded_rows=int(ends[-1]))
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()
    record("completed", passed=True)


if __name__ == "__main__":
    main()
