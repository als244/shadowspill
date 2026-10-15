"""Import a GLM-5.3-Flash checkpoint into a pool image once.

The image is an SSD pool file holding the frozen model as ShadowSpill lays it
out, with a manifest beside it. A later quickstart or runtime opens the pool on
that image and binds the frozen tensors where they lie, so a run writes only
the state it creates: LoRA factors, optimizer state and activations.

    python -m workloads.mlops.glm53_flash.convert \\
        --checkpoint ~/models/GLM-5.3-Flash-NVFP4 --image /ssd/glm53.image
"""

import argparse
import math
from pathlib import Path

from shadowspill.memory import device, transfer_route
from shadowspill.pytorch import Runtime, import_model_state, release_model_state
from shadowspill.pytorch.state.image import ALIGNMENT, write_pool_image
from shadowspill.ssd import ssd

from .checkpoint import GLMCheckpoint, physical_state


def frozen_bytes(model):
    """The pool bytes the model's distinct storages take, each root aligned."""
    seen = set()
    total = 0
    for tensor in physical_state(model).values():
        storage = tensor.untyped_storage()
        if id(storage) in seen or storage.nbytes() == 0:
            continue
        seen.add(id(storage))
        total += math.ceil(storage.nbytes() / ALIGNMENT) * ALIGNMENT
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--image", type=Path, required=True, help="the image file to create"
    )
    parser.add_argument("--layer-limit", type=int, default=None)
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="build the model with LoRA factors, as a LoRA run builds it; the "
        "frozen tensors' names do not depend on the rank (default: 8)",
    )
    parser.add_argument(
        "--frozen-state-id",
        default=None,
        help="the identity recorded in the manifest (default: the checkpoint's name)",
    )
    parser.add_argument(
        "--gemm-precision", default="bf16", help="the checkpoint's GEMM precision"
    )
    parser.add_argument(
        "--expert-storage",
        choices=("bf16", "fp8", "fp8_block"),
        help="how the routed experts are held (default: bf16 for a dense "
        "checkpoint, fp8_block -- one copy -- for a compressed one; fp8 is the "
        "two-copy container)",
    )
    arguments = parser.parse_args(argv)
    checkpoint = arguments.checkpoint.expanduser().resolve()
    image = arguments.image.expanduser().resolve()
    identity = arguments.frozen_state_id or checkpoint.name
    if arguments.layer_limit is not None and arguments.frozen_state_id is None:
        identity = f"{identity}:layers={arguments.layer_limit}"
    if arguments.frozen_state_id is None:
        identity = f"{identity}:{'lora' if arguments.lora_rank else 'full'}"
    with GLMCheckpoint(
        checkpoint,
        gemm_precision=arguments.gemm_precision,
        expert_storage=arguments.expert_storage,
    ) as source:
        model = source.build_model(
            layer_limit=arguments.layer_limit, lora_rank=arguments.lora_rank
        )
        capacity = frozen_bytes(model) + (256 << 20)
        print(f"image capacity {capacity / 2**30:.2f} GiB for {identity}")
        with Runtime(
            pools={
                "execution": device(physical_capacity=1 << 30),
                "spill": ssd(
                    capacity=capacity,
                    directory=image.parent,
                    image=image,
                    create_image=True,
                ),
            },
            # A runtime needs routes to exist; the import moves nothing over them.
            routes={
                "fetch": transfer_route(source="spill", destination="execution"),
                "evict": transfer_route(source="execution", destination="spill"),
            },
            calibrate=False,
        ) as runtime:
            done = {"count": 0}

            def progress(count, total, name):
                done["count"] = count
                if count % 200 == 0 or count == total:
                    print(f"  {count}/{total} {name}")

            model = import_model_state(
                model,
                runtime=runtime,
                pool="spill",
                initialize=lambda target: source.load_into(target, progress=progress),
            )
            try:
                written = write_pool_image(
                    model,
                    runtime=runtime,
                    pool="spill",
                    path=image,
                    frozen_state_id=identity,
                )
            finally:
                # The bytes stay in the image; only the runtime's leases go.
                release_model_state(model, runtime=runtime)
    written.trim()
    print(
        f"wrote {written.path} ({written.image_bytes / 2**30:.2f} GiB, "
        f"{len(written.roots)} roots) and {written.manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
