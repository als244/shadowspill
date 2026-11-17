"""Packed text keeps exact resume and observations across saved phase workers."""

import json

import pytest
import torch

from reference.training import config
from reference.training.text.inputs import (
    call_inputs,
    candidates,
    source,
)
from shadowspill.pytorch.guards import capture_input_signature
from tests.integration.training._text_recipe import _config


def test_packed_normalizer_is_a_runtime_tensor():
    parts = [[torch.ones(1, 8, dtype=torch.int64)] * 3] * 2
    first, second = [
        call_inputs({"candidates": {"batch": parts}, "normalizer": n}, "batch")
        for n in (7, 13)
    ]
    for calls, denominator in ((first, 7), (second, 13)):
        assert len(calls["args"]) == 2
        assert all(
            args[1].item() == pytest.approx(1 / denominator) for args in calls["args"]
        )
        assert all(args[1].shape == () for args in calls["args"])


def test_build_examples_match_actual_packed_layouts(tmp_path):
    options = json.loads(_config(tmp_path).read_text())
    data = config.resolve(options["data"])
    example, _, geometries, _ = candidates(options, data)
    for name, (tokens, count) in geometries.items():
        expected = capture_input_signature(call_inputs(example, name)["args"][0])
        actual = next(
            source(
                data,
                options,
                {
                    "candidate": name,
                    "tokens_per_microbatch": tokens,
                    "microbatches": count,
                },
            )
        )
        for args in actual["args"]:
            expected.validate(args)
