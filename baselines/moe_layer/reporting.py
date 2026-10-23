"""Detailed case-boundary logs; no periodic progress messages."""

import itertools
import json
from datetime import datetime

from .config import requested_shape
from .implementations.registry import BACKENDS


def field(name, value):
    return f"  {name:<14} {value}"


def position(args, name):
    names = [
        f"{b}/{p}/{m}/tokens-{t}"
        for b, p, m, t in itertools.product(
            args.backends, args.precisions, args.training, args.tokens
        )
    ]
    return f"[{names.index(name) + 1}/{len(names)}]"


def emit_start(args, name, attempt):
    backend, precision, training, token_label = name.split("/")
    ranks = args.ep_size if BACKENDS[backend].parallelism == "ep" else args.dp_size
    shape = requested_shape(args, ranks)
    command = attempt / "command.json"
    stamp = command.stat().st_mtime if command.exists() else datetime.now().timestamp()
    lines = [
        f"CASE START {position(args, name)} {name}",
        field(
            "settings",
            f"{BACKENDS[backend].label}; {precision}; {training}; "
            f"router={args.router_weight_dtype}",
        ),
        field(
            "geometry",
            f"{int(token_label.removeprefix('tokens-')):,} tokens/rank; "
            f"{ranks} GPUs; E={shape.experts}, top-k={shape.top_k}, "
            f"D={shape.width}, H={shape.hidden}",
        ),
        field(
            "measurement",
            f"{args.warmup} warmup + {args.iterations} timed iterations; "
            "synchronized rank boundaries",
        ),
        field(
            "LoRA", f"rank={shape.lora_rank}, alpha={shape.lora_alpha:g}; router frozen"
        )
        if training == "lora"
        else field("gradients", "FP32 accumulation"),
        field("candidates", args.sonicmoe_candidate_policy)
        if backend == "sonicmoe"
        else field("dispatcher", backend),
        field("log", attempt / "console.log"),
        field(
            "start",
            datetime.fromtimestamp(stamp).astimezone().isoformat(timespec="seconds"),
        ),
    ]
    print("\n".join(lines) + "\n", flush=True)


def emit_finished(args, name, attempt, outcome):
    status = outcome["status"]
    lines = [f"CASE {status.replace('_', ' ').upper()} {position(args, name)} {name}"]
    path = attempt / "result.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    phases = result.get("phase_timings", {})
    if status == "completed" and phases:
        ordered = [phases[p] for p in ("forward", "backward", "total")]
        lines += [
            field(
                "runtime",
                " / ".join(f"{p['median_seconds'] * 1000:.3f}" for p in ordered)
                + " ms (FWD / BWD / combined)",
            ),
            field(
                "TFLOP/s",
                " / ".join(f"{p['effective_tflops_per_gpu']:.1f}" for p in ordered)
                + " per GPU",
            ),
        ]
        memory = list(result.get("memory_by_rank", {}).values())
        for phase in ("forward", "backward"):
            if not memory:
                break
            allocated = [r[phase]["peak_allocated_bytes"] / 2**30 for r in memory]
            reserved = [r[phase]["peak_reserved_bytes"] / 2**30 for r in memory]
            sampled = [r[phase].get("sampled_device_peak_bytes") for r in memory]
            lines.append(
                field(
                    phase + " mem",
                    f"{min(allocated):.2f}-{max(allocated):.2f} GiB allocated; "
                    f"{min(reserved):.2f}-{max(reserved):.2f} GiB reserved "
                    "(peak rank range)",
                )
            )
            if all(v is not None for v in sampled):
                lines.append(
                    field(
                        "device peak",
                        f"{min(sampled) / 2**30:.2f}-{max(sampled) / 2**30:.2f} GiB "
                        "(sampled lower bound, rank range)",
                    )
                )
        lines.append(
            field(
                "validated",
                f"{result.get('all_ranks_validated')} ranks; "
                "finite outputs and gradients",
            )
        )
        request = result.get("request", {})
        lines.append(
            field(
                "reuse",
                f"process group={request.get('process_group_reused')}; "
                f"model={request.get('model_reused')}",
            )
        )
    else:
        failures = [
            json.loads(p.read_text()) for p in attempt.glob("rank-*/failure.json")
        ]
        if failures:
            lines.append(
                field(
                    "failure", f"{failures[0].get('stage')}: {failures[0].get('error')}"
                )
            )
        lines.append(field("log", attempt / "console.log"))
    if "elapsed_seconds" in outcome:
        seconds = int(outcome["elapsed_seconds"])
        lines.append(field("duration", f"{seconds // 60}m {seconds % 60:02d}s"))
    lines.append(field("result", path))
    print("\n".join(lines) + "\n", flush=True)
