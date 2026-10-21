"""The recipe configuration the AutoModel trainer builds for a case: the
shared settings and AutoModel's options in the sections AutoModel reads
them from."""

import types
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from baselines.engines.automodel.settings import Settings
from baselines.engines.automodel.train import recipe_config

BASE = {
    "model": "llama3-tiny",
    "sequence_length": 64,
    "tokens_per_step": 512,
    "tokens_per_microbatch": 128,
}


def built(own=True, attention="sdpa", world=1, targets=(), **fields):
    settings = Settings.from_dict({**BASE, **fields})
    geometry = settings.validate(world)
    resolved = settings.resolve(world)
    case = types.SimpleNamespace(
        settings=settings,
        geometry=geometry,
        resolved=resolved,
        directory=Path("/nonexistent"),
    )
    return recipe_config(
        case, resolved["precision_mechanism"], 128, attention, own, targets
    )


def test_the_shared_settings_become_the_recipes_sections():
    config = built()
    assert config["step_scheduler"]["global_batch_size"] == 8
    assert config["step_scheduler"]["local_batch_size"] == 2
    assert config["model"]["torch_dtype"] is torch.bfloat16
    assert config["model"]["config"]["attn_implementation"] == "sdpa"
    assert config["loss_fn"]["_target_"].endswith("ChunkedCrossEntropy")
    assert config["model"]["backend"]["linear"] == "torch"
    optimizer = config["optimizer"]
    assert optimizer["_target_"].endswith("FusedAdam")
    assert optimizer["master_weight_dtype"] is torch.float32
    assert optimizer["exp_avg_dtype"] is torch.float32
    mp_policy = config["distributed"]["mp_policy"]
    assert mp_policy["param_dtype"] is torch.bfloat16
    assert config["checkpoint"]["enabled"] is False
    dataset = config["dataset"]
    assert dataset["_target_"] == "baselines.engines.automodel.data.CaseTokens"
    assert dataset["steps"] == 8 and dataset["tokenizer"] is None


def test_automodels_options_go_where_automodel_reads_them():
    config = built(
        activation_checkpointing="full",
        attn="te",
        mok_precision="mxfp8",
        te_fp8="block",
        cuda_graph_modules="attn,te_dpa",
        moe_reshard_after_forward=True,
        gc_every_steps=5,
        loss="masked",
        masked_fp32_upcast=False,
    )
    assert config["distributed"]["activation_checkpointing"] is True
    backend = config["model"]["backend"]
    assert backend["attn"] == "te" and backend["mok"] == {"precision": "mxfp8"}
    assert backend["te_fp8"] == {"recipe": "block"}
    assert backend["cuda_graph"] == {"modules": ["attn", "te_dpa"]}
    assert "linear" not in backend
    assert config["distributed"]["moe"] == {"reshard_after_forward": True}
    assert config["step_scheduler"]["gc_every_steps"] == 5
    assert config["loss_fn"]["fp32_upcast"] is False


def test_hugging_faces_classes_take_no_backend():
    assert "backend" not in built(own=False)["model"]
    with pytest.raises(ValueError, match="no backend options"):
        built(own=False, rms_norm="te")


def test_te_attention_initializes_through_sdpa():
    config = built(attention="te")
    assert config["model"]["attn_implementation"] == "te"
    assert config["model"]["config"]["attn_implementation"] == "sdpa"


def test_lora_takes_the_shared_options_and_automodels_own():
    targets = ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.experts"]
    config = built(
        mode="lora", lora={"rank": 4, "alpha": 8}, lora_dropout=0.1, targets=targets
    )
    peft = config["peft"]
    assert peft["dim"] == 4 and peft["alpha"] == 8 and peft["dropout"] == 0.1
    assert peft["target_modules"] == targets
    assert built(mode="lora", lora={"rank": 4})["peft"]["dropout"] == 0.0


def test_the_dataset_serves_each_replica_the_shared_tokens():
    from baselines.common.runner import Tokens
    from baselines.engines.automodel import data

    settings = Settings.from_dict({**BASE, "warmup_steps": 1, "steps": 1})
    geometry = settings.validate(2)
    record = {}

    class Case:
        def __init__(self):
            self.settings, self.geometry, self.record = settings, geometry, record

        def data(self, vocabulary, *, replica, replicas):
            return Tokens(
                settings,
                geometry,
                vocabulary,
                replica=replica,
                replicas=replicas,
                device=None,
            )

    data.serve(Case(), 128)
    whole = data.CaseTokens(2)
    first, second = whole.shard(2, 0), whole.shard(2, 1)
    batches = list(first)
    assert len(batches) == 2 * geometry.accumulation
    assert batches[0]["input_ids"].shape == (2, 64)
    assert torch.equal(batches[0]["position_ids"][1], torch.arange(64))
    expected = Tokens(settings, geometry, 128, replica=0, replicas=2, device=None)(0)
    assert torch.equal(batches[0]["input_ids"].reshape(-1), expected[0][0, :128])
    assert torch.equal(batches[0]["labels"].reshape(-1), expected[1][0, :128])
    assert not torch.equal(batches[0]["input_ids"], next(iter(second))["input_ids"])
    assert record["automodel_replica"] == [1, 2]
