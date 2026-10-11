"""Pool-streamed checkpoints can refer to an already initialized frozen base."""

from collections.abc import Mapping
from functools import partial

import pytest
import torch
from torch import nn

from qualification.profiling import CORRECTNESS_PROFILING
from shadowspill.memory import device, transfer_route
from shadowspill.pytorch import (
    Runtime,
    import_model_state,
    plan_step,
    release_model_state,
)
from shadowspill.pytorch.materialization.training import TrainingMaterializedState
from shadowspill.ssd import ssd

pytestmark = [pytest.mark.cuda, pytest.mark.fresh_process]


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Parameter(torch.randn(16, 128), requires_grad=False)
        self.adapter = nn.Parameter(torch.zeros(16, 128))
        self.moving = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.register_buffer("calls", torch.tensor(0.0))

    def forward(self, x):
        with torch.no_grad():
            self.calls.add_(1)
            self.moving.add_(0.01)
        return x @ (self.frozen + self.adapter) + self.moving + self.calls


def _objective(model, x):
    return model(x).square().mean()


def _assert_same_state(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, Mapping):
        assert set(actual) == set(expected)
        for key in actual:
            try:
                _assert_same_state(actual[key], expected[key])
            except AssertionError as error:
                error.add_note(f"checkpoint key: {key}")
                raise
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            _assert_same_state(a, b)
    else:
        assert actual == expected


def test_partial_checkpoint_keeps_mutations_and_replays_from_ssd(tmp_path, monkeypatch):
    torch.manual_seed(112)
    with Runtime(
        pools={
            "execution": device(physical_capacity=2 << 30),
            "spill": ssd(capacity=512 << 20, directory=tmp_path),
        },
        routes={
            "fetch": transfer_route(source="spill", destination="execution"),
            "evict": transfer_route(source="execution", destination="spill"),
        },
        calibrate=False,
    ) as runtime:
        runtime.calibrate_transfer_capabilities(
            large_copy_bytes=1 << 20, warmup_copies=1, measured_copies=2
        )
        model = import_model_state(_Model(), runtime=runtime, pool="spill")
        inputs = [[torch.randn(4, 16)]]
        try:
            with plan_step(
                model,
                objective=_objective,
                optimizer=partial(torch.optim.AdamW, lr=0.003, foreach=False),
                example_inputs=inputs,
                runtime=runtime,
                execution="execution",
                spill="spill",
                artifact_store=tmp_path / "artifacts",
                profiling_options=CORRECTNESS_PROFILING,
            ) as step:
                del_result = step(inputs)
                del del_result
                initial = step.state_dict()
                checkpoint = step.state_dict(frozen_state_id="base-test-v1")
                assert set(checkpoint["model"]) == {"adapter", "moving", "calls"}
                assert checkpoint["frozen_state_id"] == "base-test-v1"
                path = tmp_path / "partial.pt"

                def refuse_snapshot(*args, **kwargs):
                    raise AssertionError("save must stream; no full model snapshot")

                with monkeypatch.context() as patch:
                    patch.setattr(
                        TrainingMaterializedState,
                        "_read_model_aliases",
                        refuse_snapshot,
                    )
                    step.save(path, frozen_state_id="base-test-v1")
                saved = torch.load(path, mmap=True, weights_only=True)
                _assert_same_state(saved, checkpoint)
                for identity in (None, "wrong-base"):
                    with pytest.raises(ValueError, match="frozen_state_id"):
                        step.load_state_dict(saved, frozen_state_id=identity)
                with pytest.raises(RuntimeError, match="model state_dict keys differ"):
                    step.load_state_dict(
                        {**saved, "model": {"adapter": saved["model"]["adapter"]}},
                        frozen_state_id="base-test-v1",
                    )
                with pytest.raises(ValueError, match="nonempty"):
                    step.save(path, frozen_state_id="")
                _assert_same_state(step.state_dict(), initial)
                del_result = step(inputs)
                del del_result
                uninterrupted = step.state_dict()
                step.load_state_dict(saved, frozen_state_id="base-test-v1")
                _assert_same_state(step.state_dict(), initial)
                del_result = step(inputs)
                del del_result
                replayed = step.state_dict()
                _assert_same_state(replayed, uninterrupted)
                torch.testing.assert_close(
                    uninterrupted["model"]["frozen"],
                    initial["model"]["frozen"],
                    rtol=0,
                    atol=0,
                )
        finally:
            release_model_state(model, runtime=runtime)
