"""A pool image: frozen state imported once into a named SSD pool file, bound in
place by every later runtime that opens its pool on the image."""

import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import pytest
import torch
from torch import nn

from qualification.profiling import CORRECTNESS_PROFILING
from shadowspill.memory import device, transfer_route
from shadowspill.pytorch import (
    Runtime,
    import_model_state,
    plan_step,
    read_model_state,
    release_model_state,
)
from shadowspill.pytorch.state.image import PoolImage, write_pool_image
from shadowspill.runtime.configuration import RuntimeConfigurationError
from shadowspill.ssd import ssd

pytestmark = [pytest.mark.cuda, pytest.mark.fresh_process]


class _Frozen(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(64, width), requires_grad=False)
        self.register_buffer("scale", torch.empty(64))

    def forward(self, x):
        return (x @ self.weight.T) * self.scale


class _Adapted(_Frozen):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Parameter(torch.empty(64, 32))

    def forward(self, x):
        return (x @ (self.weight + self.adapter).T) * self.scale


def _fill(model, adopted=frozenset()):
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, tensor in (*model.named_parameters(), *model.named_buffers()):
            value = torch.randn(tuple(tensor.shape), generator=generator)
            if name in adopted:
                continue
            tensor.copy_(
                torch.zeros(tuple(tensor.shape)) if name == "adapter" else value
            )


def _routes():
    return {
        "fetch": transfer_route(source="spill", destination="execution"),
        "evict": transfer_route(source="execution", destination="spill"),
    }


ROOT = Path(__file__).resolve().parents[4]

# Pool sizing takes what the process already holds and 512 MiB of external
# headroom out of the budget. A CUDA context alone holds 520 MiB on an H100,
# which leaves a 1 GiB budget no pool at all; 2 GiB leaves one on any device.
_DEVICE_BUDGET = 2 << 30


def _write_image_here(tmp_path, image_path):
    """Import the frozen model into a new image and describe it; one runtime."""
    with Runtime(
        pools={
            "execution": device(physical_capacity=_DEVICE_BUDGET),
            "spill": ssd(
                capacity=64 << 20,
                directory=tmp_path,
                image=image_path,
                create_image=True,
            ),
        },
        routes=_routes(),
        calibrate=False,
    ) as runtime:
        with torch.device("meta"):
            model = _Frozen()
        model = import_model_state(
            model, runtime=runtime, pool="spill", initialize=_fill
        )
        expected = read_model_state(model, runtime=runtime)
        image = write_pool_image(
            model,
            runtime=runtime,
            pool="spill",
            path=image_path,
            frozen_state_id="frozen-v1",
        )
        release_model_state(model, runtime=runtime)
    image.trim()
    torch.save(expected, tmp_path / "expected.pt")


def _write_image(tmp_path, image_path):
    """Write the image in a process of its own: a process opens one runtime,
    and the test's own runtime is the one that adopts the image."""
    completed = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            __file__,
            str(tmp_path),
            str(image_path),
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (str(ROOT), os.environ.get("PYTHONPATH", ""))
            ),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return PoolImage.read(image_path), torch.load(tmp_path / "expected.pt")


def test_an_image_is_written_once_and_adopted_without_a_copy(tmp_path):
    image_path = tmp_path / "frozen.image"
    image, expected = _write_image(tmp_path, image_path)
    assert image.names() == {"weight", "scale"}
    assert image_path.stat().st_size == image.image_bytes
    assert image.frozen_state_id == "frozen-v1"
    with Runtime(
        pools={
            "execution": device(physical_capacity=_DEVICE_BUDGET),
            "spill": ssd(capacity=512 << 20, directory=tmp_path, image=image_path),
        },
        routes=_routes(),
        calibrate=False,
    ) as runtime:
        assert runtime.pools["spill"].image_bytes == image.image_bytes
        runtime.calibrate_transfer_capabilities(
            large_copy_bytes=1 << 20, warmup_copies=1, measured_copies=2
        )
        with torch.device("meta"):
            model = _Adapted()
        model = import_model_state(
            model, runtime=runtime, pool="spill", initialize=_fill, image=image
        )
        try:
            state = read_model_state(model, runtime=runtime)
            torch.testing.assert_close(
                state["weight"], expected["weight"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                state["scale"], expected["scale"], rtol=0, atol=0
            )
            assert not state["adapter"].any()
            inputs = [[torch.randn(4, 32)]]
            with plan_step(
                model,
                objective=lambda module, x: module(x).square().mean(),
                optimizer=partial(torch.optim.AdamW, lr=0.01, foreach=False),
                example_inputs=inputs,
                runtime=runtime,
                execution="execution",
                spill="spill",
                artifact_store=tmp_path / "artifacts",
                profiling_options=CORRECTNESS_PROFILING,
            ) as step:
                step(inputs)
            after = read_model_state(model, runtime=runtime)
            torch.testing.assert_close(
                after["weight"], expected["weight"], rtol=0, atol=0
            )
            assert after["adapter"].any()
        finally:
            release_model_state(model, runtime=runtime)
    # The image holds what it held: the step wrote only beside it.
    assert image_path.stat().st_size == image.image_bytes
    assert PoolImage.read(image_path) == image


def test_an_image_refuses_state_it_does_not_describe(tmp_path):
    image_path = tmp_path / "frozen.image"
    image, _ = _write_image(tmp_path, image_path)
    with Runtime(
        pools={
            "execution": device(physical_capacity=_DEVICE_BUDGET),
            "spill": ssd(capacity=64 << 20, directory=tmp_path, image=image_path),
        },
        routes=_routes(),
        calibrate=False,
    ) as runtime:
        with torch.device("meta"):
            narrower = _Frozen(width=16)
        with pytest.raises(RuntimeConfigurationError, match="does not describe"):
            import_model_state(
                narrower, runtime=runtime, pool="spill", initialize=_fill, image=image
            )
        with torch.device("meta"):
            model = _Frozen()
        with pytest.raises(RuntimeError, match="adopted"):
            import_model_state(
                model,
                runtime=runtime,
                pool="spill",
                initialize=lambda m: _fill(m),
                image=image,
            )

        def writes_everything(model, adopted=frozenset()):
            _fill(model)

        with pytest.raises(Exception, match=r"write|status|image"):
            import_model_state(
                model,
                runtime=runtime,
                pool="spill",
                initialize=writes_everything,
                image=image,
            )


def test_an_image_pool_needs_an_image_to_adopt(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        ssd(capacity=1 << 20, directory=tmp_path, image=tmp_path / "missing.image")
    (tmp_path / "odd.image").write_bytes(b"\0" * 4000)
    with pytest.raises(ValueError, match="multiple of 4096"):
        ssd(capacity=1 << 20, directory=tmp_path, image=tmp_path / "odd.image")
    with pytest.raises(ValueError, match="needs the image path"):
        ssd(capacity=1 << 20, directory=tmp_path, create_image=True)


if __name__ == "__main__":
    _write_image_here(Path(sys.argv[1]), Path(sys.argv[2]))
