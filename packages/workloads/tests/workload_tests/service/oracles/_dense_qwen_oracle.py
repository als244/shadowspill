"""Compiled full-projection references for small pretrained dense decoders."""

from importlib import import_module

import torch

from shadowspill.artifacts.tensors import TensorChunk
from shadowspill_client import Model, Objective
from shadowspill_workloads.initialization.decoder import Initialize
from shadowspill_workloads.lora import configure_lora
from workload_tests.components.test_dense_qwen_components import settings
from workload_tests.initialization.test_pretrained_initialization import checkpoint


def options(directory, family, *, lora, override_decay=False):
    requested = settings(family, lora=lora)
    architecture = dict(requested.config["architecture"])
    architecture.update(d_model=64, d_ff=96, head_dim=16)
    if family == "qwen35":
        architecture.update(lin_k_head_dim=16, lin_v_head_dim=16)
    config = import_module(f"shadowspill_workloads.models.{family}.config").Config(
        **architecture
    )
    reference_type = import_module(
        f"shadowspill_workloads.models.{family}.implementations.pytorch"
    ).Model
    reference = reference_type(config)
    initialize = Initialize(family, config, seed=17)
    with torch.no_grad():
        for name, tensor in (
            *reference.named_parameters(),
            *reference.named_buffers(),
        ):
            initialize(
                TensorChunk(name, tuple(tensor.shape), tensor.dtype, 0, tensor.numel()),
                tensor.reshape(-1),
            )
        if override_decay:
            assert family == "qwen35" and lora
            decay = reference.blocks[0].mixer.A_log
            decay.copy_(torch.linspace(-0.123456, 0.654321, decay.numel()))
    source = directory / "pretrained"
    source.mkdir()
    checkpoint(source, family, reference)
    precision = dict(requested.precision)
    if override_decay:
        assert family == "qwen35" and lora
        # The source checkpoint keeps the original FP32 values. Independently
        # cast the reference's frozen decay after saving, so initialization must
        # honor the override while dt_bias keeps the family's FP32 default.
        precision["parameter_dtypes"] = {"*.A_log": "bfloat16"}
        decay = reference.blocks[0].mixer.A_log
        rounded = decay.detach().to(torch.bfloat16)
        assert not torch.equal(decay, rounded.float())
        with torch.no_grad():
            decay.data = rounded
        assert reference.blocks[0].mixer.dt_bias.dtype == torch.float32
    requested = Model.from_pretrained(
        str(source), lora=requested.lora, precision=precision
    )
    if lora:
        reference = configure_lora(reference, rank=2, alpha=2, factor_dtype="float32")
        initialize = Initialize(family, config, seed=0)
        with torch.no_grad():
            for name, tensor in reference.named_parameters():
                if "lora_" in name:
                    initialize(
                        TensorChunk(
                            name, tuple(tensor.shape), tensor.dtype, 0, tensor.numel()
                        ),
                        tensor.reshape(-1),
                    )
    records = [
        {
            "inputs": {"tokens": [1, 2, 3]},
            "loss_inputs": {
                "target_tokens": [2, 0, 4],
                "loss_mask": [True, False, True],
                "weights": [1.0, 1.0, -0.5],
            },
        },
        {
            "inputs": {"tokens": [4, 5]},
            "loss_inputs": {
                "target_tokens": [5, 6],
                "loss_mask": [True, True],
                "weights": [0.0, 1.2],
            },
        },
    ]
    reference = reference.cuda()
    initial = {
        name: tensor.detach().clone() for name, tensor in reference.state_dict().items()
    }
    # Each independent reference call has a concrete sequence layout. Keep its
    # graphs static, as in the numerical gate, rather than enabling Inductor's
    # automatic symbolic-shape generalization after the second layout.
    compiled = torch.compile(reference, fullgraph=True, dynamic=False)

    def loss(rows):
        lengths = tuple(len(row["inputs"]["tokens"]) for row in rows)
        tokens = [token for row in rows for token in row["inputs"]["tokens"]]
        padding = 8 - len(tokens)
        logits = compiled(
            torch.tensor([tokens + [0] * padding], device="cuda"),
            (*lengths, padding),
        ).float()[0]
        fields = {
            name: torch.tensor(
                [value for row in rows for value in row["loss_inputs"][name]]
                + [0] * padding,
                device="cuda",
            )
            for name in rows[0]["loss_inputs"]
        }
        current = (
            logits.log_softmax(-1)
            .gather(-1, fields["target_tokens"].unsqueeze(-1))
            .squeeze(-1)
        )
        return -(current * fields["weights"] * fields["loss_mask"]).sum()

    def snapshot():
        with torch.no_grad():
            logprobs = [
                compiled(torch.tensor([row["inputs"]["tokens"]], device="cuda"))
                .float()[0]
                .log_softmax(-1)
                .gather(
                    -1,
                    torch.tensor(
                        row["loss_inputs"]["target_tokens"], device="cuda"
                    ).unsqueeze(-1),
                )
                .squeeze(-1)
                .cpu()
                .tolist()
                for row in records
            ]
        return {
            "weights": {
                name: tensor.detach().cpu().tolist()
                for name, tensor in reference.named_parameters()
            },
            "logprobs": logprobs,
        }

    optim = {
        "lr": 0.002,
        "beta1": 0.8,
        "beta2": 0.95,
        "eps": 0.001,
        "weight_decay": 0.01,
    }
    optim = {name: float(torch.tensor(value)) for name, value in optim.items()}
    trained = [
        name for name, tensor in reference.named_parameters() if tensor.requires_grad
    ]
    whole_loss = float(loss(records).detach())
    loss_parts = [float(loss(rows).detach()) for rows in (records[:1], records[1:])]
    oracles = {}
    for split in (False, True):
        reference.load_state_dict(initial)
        reference.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(
            [tensor for tensor in reference.parameters() if tensor.requires_grad],
            lr=optim["lr"],
            betas=(optim["beta1"], optim["beta2"]),
            eps=optim["eps"],
            weight_decay=optim["weight_decay"],
            foreach=False,
        )
        trajectory = [snapshot()]
        for index, scale in enumerate((0.25, 1.0)):
            parts = (records[:1], records[1:]) if split and index == 0 else (records,)
            for part in parts:
                (loss(part) * scale).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            trajectory.append(snapshot())
        oracles["split" if split else "whole"] = trajectory
    return {
        "model": requested.to_dict(),
        "objective": Objective("cross_entropy").to_dict(),
        "optim": optim,
        "controls": {},
        "records": records,
        "mask": "loss_mask",
        "trainable": trained,
        "oracles": oracles,
        "whole_loss": whole_loss,
        "loss_parts": loss_parts,
        "whole_split_equivalence": True,
        "microbatch_sizes": [8],
        "microbatch_counts": [1],
        "device_gib": 6 if family == "qwen35" else 2,
        "weight_rtol": 0.003,
        "weight_atol": 2e-6,
        "logprob_rtol": 1e-5,
        "logprob_atol": 2e-5,
        "loss_rtol": 1e-5,
        "loss_atol": 2e-5,
        "separate_fp32_masters": False,
        "parameter_storage_dtypes": {
            name: str(tensor.dtype).removeprefix("torch.")
            for name, tensor in reference.named_parameters()
        },
        "exact_resume": True,
        "client_script": "_text_service_client.py",
        "model_description": f"pretrained dense {family}, FP32 "
        + ("LoRA" if lora else "full training")
        + (", frozen A_log storage overridden to BF16" if override_decay else ""),
        "oracle_description": (
            "torch.compile(fullgraph=True) independent PyTorch decoder, "
            "full vocabulary projection and FP32 AdamW"
        ),
    }
