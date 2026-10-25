"""Discover unique graph pairs through the same partition/capture as quickstart."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.pytorch.capture.export import _flatten_inputs
from shadowspill.pytorch.capture.fake import fake_device_model
from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill.pytorch.graph_pairs.capture import capture_training_stages
from shadowspill.pytorch.graph_pairs.serialization import CachedAotGraphPair
from shadowspill.pytorch.partition import partition_export
from shadowspill.pytorch.representations import materialize_meta_state

from .api import resolve
from .inputs import policy


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def registration_modules(pair):
    """Imports needed to recreate custom operator registrations in a fresh worker."""
    from torch._library.custom_ops import OPDEFS

    modules = set()
    for artifact in (pair.forward, pair.backward):
        for node in artifact.graph_module.graph.nodes:
            schema = getattr(node.target, "_schema", None)
            definition = OPDEFS.get(schema.name) if schema is not None else None
            if definition is not None:
                modules.add(definition._init_fn.__module__)
    return sorted(modules)


def capture(
    factory,
    factory_args,
    *,
    sequence_length,
    batch_size,
    output,
    variants=("save", "recompute"),
    exclude_embedding=True,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    print("Building meta model and small input controls", flush=True)
    workload = resolve(factory)(
        sequence_length=sequence_length, batch_size=batch_size, **factory_args
    )
    metadata = {
        "factory": factory,
        "factory_args": factory_args,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "variants": list(variants),
        "exclude_embedding": exclude_embedding,
        "workload": workload.metadata,
        "torch": torch.__version__,
    }
    with workload.context():
        mode = FakeTensorMode(allow_non_fake_inputs=True)
        with mode:
            # Meta storage becomes fake CUDA storage; no model payload is allocated.
            materialize_meta_state(workload.model)
            workload.model = fake_device_model(workload.model, mode)
            fake_inputs = tuple(
                v.to("cuda") if isinstance(v, torch.Tensor) else v
                for v in workload.inputs
            )
            exported = capture_training_objective(
                workload.model, workload.objective, fake_inputs
            )
            print(f"Exported objective ({time.monotonic() - started:.1f}s)", flush=True)
            roots = _flatten_inputs(exported.exported.exported_program, workload.inputs)
            partitioned = partition_export(
                exported.exported,
                exported.capture_module,
                representative_root_inputs=roots,
                fold_control_only_stages=True,
            )
            print(f"Partitioned into {len(partitioned.stages)} occurrences", flush=True)
            differentiated = capture_training_stages(
                partitioned, gradient_dtype=torch.float32, variants=tuple(variants)
            )
    stages, identities, skipped = [], {}, []
    for occurrence, stage in enumerate(differentiated):
        forward = stage.graph_pairs.reference.forward
        sources = sorted({p.source for p in forward.input_provenance if p.source})
        description = {
            "index": occurrence,
            "stage_id": stage.example.stage.stage_id,
            "state_sources": [s for s in sources if s.startswith("model.")],
        }
        if exclude_embedding and any(
            "aten.embedding." in op for op in forward.operator_targets
        ):
            skipped.append({**description, "reason": "contains embedding"})
            continue
        identity = tuple(
            (
                v.option_id,
                v.pair.forward.compatibility_digest,
                v.pair.backward.compatibility_digest,
            )
            for v in stage.graph_pairs.variants
        )
        if identity in identities:
            stages[identities[identity]]["occurrences"].append(description)
            continue
        index = len(stages)
        identities[identity] = index
        name = stage.example.stage.stage_id
        directory = output / name
        directory.mkdir()
        item = {"name": name, "occurrences": [description], "variants": []}
        for variant in stage.graph_pairs.variants:
            pair = variant.pair
            # Input policies carry only small authentic controls or recipes.
            policies = [
                policy(pair.forward, workload, i)
                for i in range(pair.forward.argument_count)
            ]
            payload = {
                "pair": CachedAotGraphPair.capture(pair),
                "registration_modules": registration_modules(pair),
                "policies": policies,
                "variant": variant.option_id,
                "workload": metadata,
            }
            path = directory / (variant.option_id + ".pt")
            torch.save(payload, path)
            for phase in ("forward", "backward"):
                artifact = getattr(pair, phase)
                (directory / f"{variant.option_id}-{phase}.py").write_text(
                    artifact.graph_module.code
                )
            item["variants"].append(
                {
                    "name": variant.option_id,
                    "artifact": str(path),
                    "forward_digest": pair.forward.compatibility_digest,
                    "backward_digest": pair.backward.compatibility_digest,
                    "saved_value_count": pair.saved_value_count,
                    "specialized_unit_tangents": pair.specialized_unit_tangent_count,
                    "forward_inputs": [
                        {
                            "source": p.source,
                            "role": p.role.value,
                            "shape": list(v.shape),
                            "stride": list(v.stride()),
                            "dtype": str(v.dtype),
                            "device": v.device.type,
                        }
                        for v, p in zip(
                            pair.forward.example_arguments,
                            pair.forward.input_provenance,
                            strict=True,
                        )
                    ],
                }
            )
        stages.append(item)
        print(f"Unique {name}: {len(item['variants'])} variants", flush=True)
    manifest = {
        **metadata,
        "stage_count": len(differentiated),
        "stages": stages,
        "skipped": skipped,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "manifest.json", manifest)
    print(
        f"Saved {len(stages)} unique non-embedding stage pairs in {output}", flush=True
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--factory-args", type=json.loads, default={})
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("save", "recompute"),
        default=["save", "recompute"],
    )
    parser.add_argument("--include-embedding", action="store_true")
    args = parser.parse_args()
    capture(
        args.factory,
        args.factory_args,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        output=args.output,
        variants=args.variants,
        exclude_embedding=not args.include_embedding,
    )


if __name__ == "__main__":
    main()
