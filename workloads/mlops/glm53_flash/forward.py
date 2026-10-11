"""Run text or image/text forwards with a directly imported HF GLM checkpoint."""

import argparse
import json
import math
import resource
import shutil
import time
from pathlib import Path

import torch
from torch._inductor import config as inductor_config

from .checkpoint import GLMCheckpoint
from .partition import GLMStages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--ssd-directory", type=Path)
    parser.add_argument("--execution-gib", type=float, default=24)
    parser.add_argument("--spill-gib", type=float)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument(
        "--image", type=Path, help="One still image; uses the HF chat template"
    )
    parser.add_argument(
        "--assistant-prefix",
        default="",
        help="Optional assistant continuation for an image prompt",
    )
    parser.add_argument(
        "--minimum-evict-kib",
        type=int,
        default=1024,
        help="Reserve GPU slots for smaller objects (default: 1024 KiB); "
        "zero disables this small-object reservation",
    )
    parser.add_argument("--gemm-precision", choices=("bf16",), default="bf16")
    parser.add_argument(
        "--layer-limit", type=int, help="Debug only: execute a decoder prefix"
    )
    parser.add_argument(
        "--text",
        action="append",
        help="Prompt; repeat to reuse one model and plan across several prompts",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Validate metadata and report storage without loading weights",
    )
    args = parser.parse_args()
    if args.passes < 1:
        parser.error("--passes must be positive")
    if args.minimum_evict_kib < 0:
        parser.error("--minimum-evict-kib must be non-negative")
    if args.image is not None and args.text is not None and len(args.text) != 1:
        parser.error("--image accepts exactly one --text prompt")
    if args.assistant_prefix and args.image is None:
        parser.error("--assistant-prefix requires --image")
    args.outdir.mkdir(parents=True, exist_ok=True)
    with GLMCheckpoint(
        args.checkpoint, gemm_precision=args.gemm_precision
    ) as checkpoint:
        model = checkpoint.build_model(
            layer_limit=args.layer_limit, include_vision=args.image is not None
        )
        (args.outdir / "inventory.json").write_text(
            json.dumps(checkpoint.inventory, indent=2) + "\n"
        )
        print(json.dumps(checkpoint.inventory), flush=True)
        if args.inspect:
            return

        from tokenizers import Tokenizer

        from shadowspill.memory import device, transfer_route
        from shadowspill.planner import GenericPlanningOptions, SearchOptions
        from shadowspill.pytorch import (
            Runtime,
            import_model_state,
            plan_forward,
            release_model_state,
        )
        from shadowspill.pytorch.planning.common import estimate_spill_reservation
        from shadowspill.ssd import ssd
        from shadowspill.task.profiling import ProfilingOptions

        tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
        images = None
        if args.image is None:
            texts = args.text or ["The capital of France is"]
            token_lists = [tokenizer.encode(text).ids for text in texts]
        else:
            from .prompts import prepare_image_prompt

            text, tokens, images = prepare_image_prompt(
                args.checkpoint,
                args.image,
                (args.text or ["Describe this image."])[0],
                assistant_prefix=args.assistant_prefix,
            )
            texts, token_lists = [text], [tokens]
        if any(not tokens for tokens in token_lists):
            parser.error("Each --text must produce at least one token")
        length = max(map(len, token_lists))
        # Right padding does not influence preceding positions in this causal
        # model. Evaluate each prompt at its last real token, never at padding.
        batches = [
            [
                torch.tensor(tokens + [0] * (length - len(tokens)), dtype=torch.int64),
                torch.tensor([0, length], dtype=torch.int64),
                torch.tensor(
                    [(0, i) for i in range(math.ceil(length / 64))], dtype=torch.int64
                ),
            ]
            for tokens in token_lists
        ]
        if images is not None:
            batches[0].append(images)
        inputs = batches[0]
        (args.outdir / "invocation.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint.resolve()),
                    "text": texts[0] if len(texts) == 1 else texts,
                    "token_ids": token_lists[0] if len(texts) == 1 else token_lists,
                    "planned_tokens": length,
                    "image": None if args.image is None else str(args.image.resolve()),
                    "assistant_prefix": args.assistant_prefix,
                    "padding_token_id": 0,
                    "gemm_precision": args.gemm_precision,
                    "torch_version": torch.__version__,
                    "inductor_emulate_precision_casts": bool(
                        inductor_config.emulate_precision_casts
                    ),
                    "layer_limit": args.layer_limit,
                    "execution_gib": args.execution_gib,
                    "minimum_evict_kib": args.minimum_evict_kib,
                    "passes": args.passes,
                },
                indent=2,
            )
            + "\n"
        )
        minimum = estimate_spill_reservation(model, inputs, 1 << 62)
        spill_bytes = (
            int(args.spill_gib * 2**30)
            if args.spill_gib
            else (math.ceil(minimum / 2**30) + 4) * 2**30
        )
        estimate_spill_reservation(model, inputs, spill_bytes)
        directory = args.ssd_directory or args.outdir
        directory.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(directory).free
        if free < spill_bytes + 2**30:
            raise RuntimeError(
                f"SSD pool needs {spill_bytes / 2**30:.1f} GiB plus 1 GiB free margin; "
                f"directory has {free / 2**30:.1f} GiB"
            )
        started = time.monotonic()
        last_progress = started

        def progress(done, total, name):
            nonlocal last_progress
            now = time.monotonic()
            if now - last_progress > 5 or done == total:
                print(
                    f"import {done}/{total} [{now - started:.1f}s] {name}", flush=True
                )
                last_progress = now

        with Runtime(
            pools={
                "execution": device(physical_capacity=int(args.execution_gib * 2**30)),
                "spill": ssd(capacity=spill_bytes, directory=directory),
            },
            routes={
                "fetch": transfer_route(source="spill", destination="execution"),
                "evict": transfer_route(source="execution", destination="spill"),
            },
            calibrate=False,
        ) as runtime:
            runtime.calibrate_transfer_capabilities(
                large_copy_bytes=32 << 20, warmup_copies=1, measured_copies=3
            )
            model = import_model_state(
                model,
                runtime=runtime,
                pool="spill",
                initialize=lambda module: checkpoint.load_into(
                    module, progress=progress
                ),
            )
            checkpoint.close()
            peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            print(
                f"import complete: {time.monotonic() - started:.3f}s; "
                f"peak RSS {peak_rss_gib:.3f} GiB",
                flush=True,
            )
            try:
                with plan_forward(
                    model,
                    forward_fn=lambda m, x, cumulative, chunks, images=None: m(
                        x, (0, x.numel()), cumulative, chunks, images=images
                    ),
                    example_inputs=inputs,
                    runtime=runtime,
                    execution="execution",
                    spill="spill",
                    partition=GLMStages(),
                    artifact_store=args.outdir / "artifacts",
                    search_options=SearchOptions(
                        generic=GenericPlanningOptions(
                            minimum_object_bytes_evict_eligible=args.minimum_evict_kib
                            * 1024
                        )
                    ),
                    profiling_options=ProfilingOptions(
                        conditioning_seconds=0,
                        conditioning_wall_seconds=0,
                        measurement_seconds=0,
                        measurement_wall_seconds=0,
                    ),
                ) as forward:
                    results, first = [], {}
                    for i in range(args.passes):
                        for prompt_index, (batch, tokens) in enumerate(
                            zip(batches, token_lists, strict=True)
                        ):
                            begin = time.monotonic()
                            logits = forward(batch).float().cpu()[: len(tokens)].clone()
                            elapsed = time.monotonic() - begin
                            if not torch.isfinite(logits).all():
                                raise AssertionError("Non-finite logits")
                            if prompt_index not in first:
                                first[prompt_index] = logits.clone()
                            else:
                                torch.testing.assert_close(
                                    logits, first[prompt_index], rtol=0, atol=0
                                )
                            values, ids = logits[-1].topk(5)
                            log_probabilities = logits[-1].log_softmax(dim=-1)[ids]
                            result = {
                                "pass": i,
                                "prompt": prompt_index,
                                "text": texts[prompt_index],
                                "tokens": len(tokens),
                                "seconds": elapsed,
                                "top_ids": ids.tolist(),
                                "top_logits": values.tolist(),
                                "top_log_probabilities": log_probabilities.tolist(),
                                "top_tokens": [
                                    tokenizer.decode([j]) for j in ids.tolist()
                                ],
                            }
                            results.append(result)
                            display = dict(result)
                            if images is not None:
                                # Keep expanded chat tokens in the artifacts, not
                                # hundreds of image placeholders in the console.
                                display["text"] = (
                                    args.text or ["Describe this image."]
                                )[0]
                            print(json.dumps(display), flush=True)
                            (args.outdir / "forwards.json").write_text(
                                json.dumps(results, indent=2) + "\n"
                            )
                            name = (
                                "logits.pt"
                                if len(texts) == 1
                                else f"logits-{prompt_index}.pt"
                            )
                            torch.save(logits, args.outdir / name)
                    print(
                        "PASS: finite, repeatable checkpoint forwards. "
                        "Reference-model parity is a separate validation.",
                        flush=True,
                    )
            finally:
                release_model_state(model, runtime=runtime)


if __name__ == "__main__":
    main()
