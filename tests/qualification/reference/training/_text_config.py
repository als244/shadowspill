"""Packed-text configuration and log reader for the PyTorch reference."""

import json

from tests.integration._text import write_dataset


def _config(tmp_path):
    raw = {
        "settings": [
            {"@call": "tests.integration._text:note", "text": "first"},
            {"@call": "tests.integration._text:note", "text": "second"},
        ],
        "run_dir": str(tmp_path / "run"),
        "model": {
            "@call": "shadowspill_workloads.models.construction:build_on_meta",
            "model": "@tests.integration._text:TinyModel",
        },
        "objective": (
            "@shadowspill_workloads.objectives.next_token:next_token_prediction"
        ),
        "optimizer": "@torch.optim:AdamW",
        "optimizer_args": {"lr": 0.01},
        "data": {
            "@call": "shadowspill_workloads.inputs.text.packed:PackedTokens",
            "directory": str(write_dataset(tmp_path / "tokens")),
        },
        "steps": 6,
        "max_seq_len": 512,
        "max_tokens_per_step": 4096,
        "max_tokens_per_microbatch": 2048,
        "schedules": {
            "lr": {
                "@call": "shadowspill_client.schedules:WarmupCosine",
                "lr": 0.01,
                "min_lr": 0.001,
                "warmup_steps": 2,
                "total_steps": 6,
            }
        },
        "engine": "pytorch",
        "compile": False,
        "device": "cpu",
        "eval_every": 3,
        "eval_tokens": 4096,
        "checkpoint_every": 3,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    return path


def _records(root, key):
    return {
        row["step"]: row
        for line in (root / "metrics.jsonl").read_text().splitlines()
        if key in (row := json.loads(line))
    }
