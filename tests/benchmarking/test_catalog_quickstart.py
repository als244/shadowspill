"""Text factories preserve shapes, initialization and DP normalization across phases."""

from types import SimpleNamespace

import pytest
import torch

from workloads.recipes.text.benchmark.factory import experiment


@pytest.mark.parametrize(
    "family", ["llama3", "qwen3", "qwen35", "olmoe", "qwen3moe", "qwen35moe"]
)
@pytest.mark.parametrize("lora", [None, {}])
def test_initializer_covers_model_and_preserves_chunk_identity(family, lora):
    setup = experiment(
        device="cuda:0",
        model=family + "-tiny",
        sequence_length=8,
        sequences_per_step=2,
        lora=lora,
    )
    model = setup["model_factory"]()
    initialize = setup["fill_tensor"]
    initialize.block_elements = 31
    tensors = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    for name, tensor in tensors.items():
        count = min(tensor.numel(), 67)
        request = SimpleNamespace(
            name=name,
            shape=tuple(tensor.shape),
            storage_offset=0,
            storage_elements=tensor.numel(),
            offset=0,
        )
        whole = torch.empty(count, dtype=tensor.dtype)
        initialize(request, whole)
        split = torch.empty_like(whole)
        for start, stop in ((0, count // 2), (count // 2, count)):
            request.offset = start
            initialize(request, split[start:stop])
        assert torch.equal(whole, split), name
        assert torch.isfinite(whole).all(), name
        if name.endswith(("lora_b", "lora_gate_up_b", "lora_down_b")):
            assert not torch.count_nonzero(whole), name


def test_candidates_reuse_tokens_and_normalize_over_global_step():
    options = dict(
        device="cuda:0",
        model="llama3-tiny",
        sequence_length=8,
        sequences_per_step=4,
        ranks=2,
    )
    first, second = (experiment(**options, logical_rank=rank) for rank in range(2))
    for setup in (first, second):
        for tensor_index in (0, 1):
            whole = setup["candidates"]["4"]["args"][0][tensor_index]
            for candidate in setup["candidates"].values():
                pieces = [args[tensor_index] for args in candidate["args"]]
                assert torch.equal(torch.cat(pieces, 1), whole)
                assert all(piece.stride() == (piece.numel(), 1) for piece in pieces)
    assert not torch.equal(
        first["candidates"]["4"]["args"][0][0], second["candidates"]["4"]["args"][0][0]
    )
    assert first["units_per_step"] == 64

    class Model:
        config = SimpleNamespace(vocab_size=32)
        lm_head = SimpleNamespace(
            loss=lambda hidden, targets, **kwargs: targets.float().sum()
        )

        def hidden(self, tokens, lengths):
            return tokens

    values = (
        torch.zeros(1, 8, dtype=torch.long),
        torch.ones(1, 8, dtype=torch.long),
        (8,),
    )
    loss, metrics = first["objective"](Model(), *values)
    assert loss.item() == metrics["head_loss"].item() == 8 / 64


def test_initialize_qwen35_norms_and_decay():
    setup = experiment(
        device="cuda:0", model="qwen35moe-tiny", sequence_length=8, sequences_per_step=1
    )
    initialize = setup["fill_tensor"]

    def values(name, shape):
        return initialize.values(
            name, shape, 0, shape[-1], torch.Generator().manual_seed(1)
        )

    assert torch.equal(values("blocks.0.attn_norm.weight", (8,)), torch.zeros(8))
    assert torch.equal(values("blocks.0.mixer.lin_norm.weight", (8,)), torch.ones(8))
    assert torch.equal(values("blocks.0.mixer.dt_bias", (8,)), torch.ones(8))
    decay = values("blocks.0.mixer.A_log", (32,)).exp()
    assert ((decay >= 0.01) & (decay <= 16)).all()


@pytest.mark.cuda
@pytest.mark.parametrize("lora", [None, {}])
@pytest.mark.parametrize("ranks", [1, 2])
def test_catalog_training_from_saved_phases(tmp_path, lora, ranks):
    import json

    from benchmarking.quickstart import run
    from shadowspill.artifacts.sweep import SweepResult

    if not torch.cuda.is_available() or torch.cuda.device_count() < ranks:
        pytest.skip(f"requires {ranks} distinct CUDA devices")
    assert (
        run(
            "workloads.recipes.text.benchmark.factory:experiment",
            factory_args={
                "model": "llama3-tiny",
                "sequence_length": 32,
                "sequences_per_step": 2,
                "sequences_per_example": 1,
                "lora": lora,
                "ranks": ranks,
                "model_config": {
                    "d_model": 128,
                    "d_ff": 256,
                    "n_layers": 1,
                    "n_heads": 2,
                    "n_kv_heads": 1,
                    "vocab_size": 128,
                    "max_seq_len": 64,
                },
                "precision": {
                    "master_dtype": "float32",
                    "grad_dtype": "float32",
                    "opt_state_dtype": "float32",
                },
            },
            # Two H100 workers retain about 2.46 GiB of calibrated fixed
            # overhead. Leave room for the task instead of weakening admission.
            search_budget_gib=[2 if ranks == 1 else 3],
            spill_gib=0.125,
            steps=3,
            output_dir=tmp_path,
            warmup_seconds=0.001,
            measure_seconds=0.002,
            external_headroom_gib=0.75,
            plots=False,
        )
        == 0
    )
    sweep = SweepResult.load(tmp_path / "planning/sweep.ssp")
    assert {case.candidate for case in sweep.cases} == {"1: 1x2rp", "1: 2x1rp"}
    assert all(case.plan is not None for case in sweep.cases), {
        case.candidate: dict(case.failure) for case in sweep.cases if case.plan is None
    }
    result = json.loads((tmp_path / "execution/budget-0000/result.json").read_text())
    assert result["units_per_step"] == 64 * ranks
    assert result["units_per_second"] > 0
    losses = [row["loss"] for row in result["steps"]]
    assert losses[-1] < losses[0]
    assert all(torch.isfinite(torch.tensor(losses)))
    assert len(list((tmp_path / "programs").glob("*.ssp"))) == 1
