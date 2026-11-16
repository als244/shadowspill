"""Qwen application setup and composed-objective normalization on a light client."""

import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from examples.train_qwen35_lora import arguments, main
from examples.training.application.loop import update
from examples.training.qwen.settings import Settings, qwen3_settings
from examples.training.qwen.setup import (
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
from shadowspill_workloads.definitions.expert_parallel import ownership
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


@pytest.mark.parametrize("family", ["qwen3moe", "qwen35moe"])
def test_setup_keeps_exact_chunk_choices_global_capacity_and_checkpoint_coefficient(
    tmp_path,
    family,
):
    base = tmp_path / "base"
    base.mkdir()
    checkpoint(base, family)
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
        family=family,
        lora_rank=32 if family == "qwen35moe" else None,
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
    if family == "qwen35moe":
        assert config.model.config["lora_seed"] == 45
    else:
        assert config.model.lora is None and config.model.config["lora_seed"] is None
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


@pytest.mark.parametrize("kind", ["dp", "ep"])
def test_qwen3_full_defaults_preserve_global_update_and_existing_ownership(kind):
    settings = qwen3_settings(data="/client/data", parallelism=kind)
    request = settings.request()
    assert request["batch"].limits["token_capacity"] == 2097152
    assert request["batch"].limits["max_sequences"] == 512
    assert settings.microbatch_sizes == (1, 2, 4, 8, 16, 32)
    assert request["prepare"].external_headroom_bytes == 768 << 20
    assert request["execution"].reject_overbudget is False
    assert request["model"].initialize.kind == "random"
    assert request["model"].initialize.seed == 0
    assert request["model"].lora is None
    parallel = Parallelism.from_dict(request["parallelism"])
    expected = Parallelism.dp(ranks_per_replica=8) if kind == "dp" else ownership(8, 1)
    assert replace(parallel, inputs=None) == expected
    assert parallel.inputs.partitions.groups == tuple((r,) for r in range(8))
    registry = load_registry(["shadowspill", "shadowspill-workloads"])
    setup = resolve_training(registry, **{**request, "parallelism": parallel})
    assert setup.config.model.definition == "shadowspill/qwen3moe@1"
    assert setup.config.model.config["architecture"]["d_model"] == 2048
    assert setup.config.preparation.external_headroom_bytes == 768 << 20
    assert setup.config.execution.reject_overbudget is False


def test_full_cli_is_framework_free_and_uses_global_sequences(tmp_path):
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for name in ('torch', 'triton', 'mlops', 'shadowspill', 'shadowspill_workloads'):
    sys.modules[name] = None
from examples.train_qwen3_distributed import arguments
from examples.training.qwen.settings import qwen3_settings
args = arguments(['--run-dir', '/tmp/unused', '--data', '/client/data',
                  '--preset', 'tiny', '--ranks', '2', '--sequences-per-step', '4',
                  '--external-headroom-mib', '0', '--reject-overbudget'])
s = qwen3_settings(data=args.data, preset=args.preset, ranks=args.ranks,
                  sequences_per_step=args.sequences_per_step,
                  external_headroom_mib=args.external_headroom_mib,
                  reject_overbudget=args.reject_overbudget)
request = s.request()
assert request['batch'].limits['max_sequences'] == 4
assert request['batch'].limits['token_capacity'] == 4 * 4096
assert request['prepare'].external_headroom_bytes == 0
assert request['execution'].reject_overbudget is True
assert request['model'].initialize.kind == 'random'
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    settings = qwen3_settings(data=str(tmp_path), checkpoint="/server/base")
    assert settings.preset is None
    assert settings.request()["model"].initialize.kind == "pretrained"
    path = tmp_path / "full.json"
    path.write_text(
        json.dumps({"application": settings.application, "settings": record(settings)})
    )
    assert read_resume(path)[1] == settings
    with pytest.raises(ValueError, match="another application"):
        main(["--run-dir", str(tmp_path / "run"), "--resume", str(path)])


@pytest.mark.parametrize(
    "options",
    [
        {"ranks": 1},
        {"lora_rank": 2},
        {"compute_dtype": "float32"},
        {"chunk_sizes": (1024,)},
        {"moe_chunk_sizes": (1024,)},
    ],
)
def test_ep_rejects_unsupported_application_choices_before_connecting(options):
    with pytest.raises(ValueError, match="EP requires"):
        qwen3_settings(data="/client/data", parallelism="ep", **options)


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
        "application": settings.application,
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
        "application": settings.application,
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
from examples.training.qwen.settings import Settings
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
        json.dumps({"application": settings.application, "settings": record(settings)})
    )
    with pytest.raises(ValueError, match="saved training settings"):
        main(
            ["--run-dir", str(tmp_path / "new"), "--resume", str(path), "--steps", "6"]
        )
    assert (
        replace(settings, microbatch_sizes=list(settings.microbatch_sizes)) == settings
    )
