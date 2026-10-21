"""Write a Hugging Face checkpoint of a catalog model, initialized under a
seed, that every harness can start from.

    python -m baselines.scripts.write_checkpoint --model NAME --output DIR \\
        [--model-config JSON] [--seed N]

The configuration is the family's ``to_huggingface`` of the catalog entry,
so the recipe systems build the same model from it that our implementations
do; the weights are the ``transformers`` model's own initialization at
bfloat16, saved as safetensors. The family's checkpoint reader then reads it
back, which refuses a configuration other than the catalog entry's, so a
written checkpoint is one every harness reads. A tiny model's checkpoint is
what a LoRA case that must start from pretrained weights starts from, and
the same weights let the harnesses' loss curves be compared.
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path


def write(model, output, *, config_overrides=None, seed=0):
    """The checkpoint directory of ``model``; the entry's configuration,
    read back through the family's reader."""

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from baselines.common.models import ModelRequest
    from workloads.models import catalog

    entry = catalog.entry(model)
    config = entry.config(**dict(config_overrides or {}))
    family = import_module(f"workloads.models.{entry.family}.config")
    value = family.to_huggingface(config)
    fields = {key: item for key, item in value.items() if key != "model_type"}
    torch.manual_seed(seed)
    network = AutoModelForCausalLM.from_config(
        AutoConfig.for_model(value["model_type"], **fields), dtype=torch.bfloat16
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    network.save_pretrained(output, safe_serialization=True)
    (output / "catalog.json").write_text(
        json.dumps(
            {
                "model": model,
                "model_config": dict(config_overrides or {}),
                "seed": seed,
            },
            indent=2,
        )
        + "\n"
    )
    implementation = entry.implementations[0]
    request = ModelRequest(model, implementation, config=dict(config_overrides or {}))
    request.checkpoint(str(output)).close()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="a catalog model name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=json.loads,
        default={},
        metavar="JSON",
        help="field overrides on the catalog entry's configuration",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    path = write(
        args.model, args.output, config_overrides=args.model_config, seed=args.seed
    )
    print(path)


if __name__ == "__main__":
    main()
