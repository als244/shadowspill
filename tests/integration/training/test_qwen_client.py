"""Qwen application setup and composed-objective normalization on a light client."""

import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from examples.train_qwen35_lora import arguments, main
from examples.training.qwen35_lora.loop import update
from examples.training.qwen35_lora.settings import Settings
from examples.training.qwen35_lora.setup import (
    APPLICATION,
    data_identity,
    read_resume,
    record,
    resolve,
    start,
)
from examples.training.text_data import TokenBatches
from shadowspill.parallelism import Parallelism
from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_client.protocol.requests import ArtifactRef
from workload_tests.test_local_pretrained import checkpoint


def future(value):
    return SimpleNamespace(result=lambda: value)


def corpus(root):
    root.mkdir()
    tokens = np.arange(97, dtype=np.uint32) % 16
    tokens.tofile(root / "train.bin")
    ((tokens + 1) % 16).tofile(root / "val.bin")
    (root / "meta.json").write_text(json.dumps({"vocab_size": 16}))
    return root


def test_setup_keeps_exact_chunk_choices_global_capacity_and_checkpoint_coefficient(
    tmp_path,
):
    base = tmp_path / "base"
    base.mkdir()
    checkpoint(base, "qwen35moe")
    metadata = json.loads((base / "config.json").read_text())
    metadata.get("text_config", metadata)["router_aux_loss_coef"] = 0.25
    (base / "config.json").write_text(json.dumps(metadata))
    settings = Settings(
        str(tmp_path),
        str(base),
        sequence_length=4,
        sequences_per_step=4,
        ranks=2,
        microbatch_sizes=(1, 2, 1),
        chunk_sizes=(2,),
        moe_chunk_sizes=(2, 4),
        seed=45,
        graph_pair_policies=("cost-input", "cost-attention"),
        resolution_options=("0", "1/2", "1"),
        tail_save_counts=(0, 2),
    )
    registry = load_registry(["shadowspill", "shadowspill-workloads"])
    requests = []

    def describe(**request):
        requests.append(request)
        return future(
            resolve_training(
                registry,
                **{
                    **request,
                    "parallelism": Parallelism.from_dict(request["parallelism"]),
                },
            ).preview()
        )

    preview = resolve(SimpleNamespace(describe_training=describe), settings)
    assert len(requests) == 2
    config = preview.config
    assert config.model.config["lora_seed"] == 45
    assert config.model.initialize.revision.startswith("local-sha256:")
    assert config.objective.defaults["balance.weight"] == 0.25
    assert not config.objective.config["terms"]["balance"]["config"]["include_padding"]
    choices = config.preparation
    assert len(choices.cases) == 8
    assert set(choices.planning.microbatch_counts) == {1, 2}
    search = choices.planning.to_dict()["search"]["algorithm"]["options"]
    assert search["resolution_options"] == ["0", "1/2", "1"]
    assert search["tail_save_counts"] == [0, 2]
    for case in choices.cases:
        assert (
            case.build.to_dict()["chunking"]["config"]["token_capacity"]
            == case.microbatch_size
        )
    layouts = prepare_layouts(resolve_training(registry, config))
    for family in layouts.families.values():
        for row in family:
            assert (
                sum(
                    buffer.fields["inputs", "tokens"][0]
                    for buffer in row.candidate.schedule.buffers
                )
                <= 16
            )


@pytest.mark.parametrize("pieces", [1, 2, 3])
def test_complete_composed_objective_is_normalized_once(tmp_path, pieces):
    source = TokenBatches(corpus(tmp_path / "data"), 4, 3, normalize=False)
    accumulated = 0.0
    scales = []

    def backward(batch):
        nonlocal accumulated
        # Derivative of theta * (2 * weighted_tokens + 3 * valid_tokens).
        # The secondary objective deliberately ignores primary loss weights.
        weights = sum(float(row.loss_inputs["weights"].sum()) for row in batch)
        positions = sum(len(row.inputs["tokens"]) for row in batch)
        loss = 2 * weights + 3 * positions
        accumulated += loss
        return future(SimpleNamespace(loss=loss))

    def optimize(*, optim, grad_scale):
        scales.append(grad_scale)
        assert optim == {"lr": 0.25}
        return future(SimpleNamespace(delta=-optim["lr"] * accumulated * grad_scale))

    def step(batch, **options):
        loss = backward(batch).result().loss
        result = optimize(**options).result()
        result.loss = loss
        return future(result)

    training = SimpleNamespace(
        step=step, forward_backward=backward, optim_step=optimize
    )
    result, loss = update(training, source[0], pieces=pieces, lr=0.25, positions=12)
    assert loss == 5 and result.delta == -1.25
    assert scales == [1 / 12]


@pytest.mark.parametrize("change", ["train", "validation", "settings", "next_update"])
def test_resume_rejects_changed_corpus_settings_and_progress_before_workers(
    tmp_path, change
):
    data = corpus(tmp_path / "data")
    settings = Settings(
        str(data),
        "/server/base",
        sequence_length=4,
        sequences_per_step=2,
        microbatch_sizes=(1, 2),
    )
    source = TokenBatches(data, 4, 2, normalize=False)
    validation = TokenBatches(data, 4, 2, split="val", normalize=False)
    reference = ArtifactRef("checkpoint", "a" * 64, 1)
    metadata = {
        "application": APPLICATION,
        "settings": record(settings),
        "data": data_identity(source, validation),
        "next_update": 1,
    }
    if change in ("train", "validation"):
        metadata["data"][change] = {}
    else:
        metadata[change] = 2 if change == "next_update" else {}
    info = SimpleNamespace(
        to_dict=lambda: {"metadata": metadata}, completed_updates=1, plans=reference
    )
    service = SimpleNamespace(
        artifact=lambda _: SimpleNamespace(
            manifest=lambda: future(SimpleNamespace(info=info))
        )
    )
    saved = {
        "application": APPLICATION,
        "settings": record(settings),
        "checkpoint": reference.to_dict(),
        "plans": reference.to_dict(),
    }
    path = tmp_path / "resume.json"
    path.write_text(json.dumps(saved))
    assert read_resume(path) == (saved, settings)
    with pytest.raises(ValueError, match="settings, corpus or progress"):
        start(service, settings, source, validation, tmp_path, resume=saved)


@pytest.mark.parametrize(
    "values",
    [
        {"ranks": 3},
        {"microbatch_sizes": []},
        {"microbatch_sizes": [3]},
        {"backward_pieces": 9},
        {"schedule_steps": 1},
        {"balance_weight": float("nan")},
        {"seed": -1},
        {"compute_dtype": "float16"},
    ],
)
def test_invalid_settings_fail_on_client_before_connection(values):
    with pytest.raises(ValueError):
        Settings("/tmp/data", "/server/base", **values)


def test_framework_free_cli_and_resume_preserve_explicit_settings(tmp_path):
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for name in ('torch', 'triton', 'mlops', 'shadowspill', 'shadowspill_workloads'):
    sys.modules[name] = None
from examples.train_qwen35_lora import arguments
from examples.training.qwen35_lora.settings import Settings
args = arguments(['--run-dir', '/tmp/unused', '--data', '/tmp/data',
                  '--checkpoint', '/server/only', '--ssh', 'chicago'])
assert args.ssh == 'chicago'
assert Settings(args.data, args.checkpoint).request()['model'].config['lora_seed'] == 0
""",
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    args = arguments(["--run-dir", str(tmp_path), "--resume", "saved.json"])
    assert "steps" not in vars(args) and "data" not in vars(args)
    path = tmp_path / "saved.json"
    settings = Settings(str(tmp_path), "/server/base")
    path.write_text(
        json.dumps({"application": APPLICATION, "settings": record(settings)})
    )
    with pytest.raises(ValueError, match="saved training settings"):
        main(
            ["--run-dir", str(tmp_path / "new"), "--resume", str(path), "--steps", "6"]
        )
    assert (
        replace(settings, microbatch_sizes=list(settings.microbatch_sizes)) == settings
    )
