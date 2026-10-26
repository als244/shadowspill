"""Profiling retains only the current pair's inputs; saved Profiles own no tensors."""

import gc
import weakref

import pytest
import torch

from shadowspill import Profiles, Program, build_step, measure_program
from shadowspill.profiling.session import MeasurementSession
from shadowspill.runtime.bootstrap import installed_runtime
from tests.shadowspill.build.test_training import Network, objective


@pytest.mark.cuda
def test_pair_inputs_are_released_between_measurements_and_on_failure(
    tmp_path, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    with torch.device("meta"):
        model = Network()
    program = build_step(
        model,
        optimizer=torch.optim.AdamW(model.parameters(), foreach=False),
        objective=objective,
        example_args=(torch.empty(4, 8),),
        example_kwargs={"labels": torch.empty(4, 8)},
        num_examples=2,
    )
    program.save(tmp_path / "program.ssp")
    program = Program.load(tmp_path / "program.ssp")
    references, completed = [], []
    pair, measure = MeasurementSession.pair, MeasurementSession.measure
    failing_key = None

    def released():
        assert all(reference() is None for reference in references)

    def track_measure(self, key, arguments):
        references.extend(
            weakref.ref(t) for t in arguments if isinstance(t, torch.Tensor)
        )
        if key == failing_key:
            raise RuntimeError("intentional backward profiling failure")
        return measure(self, key, arguments)

    def track_pair(self, job):
        released()
        count = len(references)
        pair(self, job)
        released()
        if len(references) > count:
            completed.append(job["stage"])

    monkeypatch.setattr(MeasurementSession, "measure", track_measure)
    monkeypatch.setattr(MeasurementSession, "pair", track_pair)
    options = dict(warmup_seconds=0.001, measure_seconds=0.002)
    for attempt in range(2):
        profiles = measure_program(program, **options)
        released()
        assert len(set(completed)) > 1
        assert installed_runtime() is None
        assert torch.cuda.memory.get_allocator_backend() == "native"
        profiles.save(tmp_path / f"profiles-{attempt}.ssp")
        assert (
            Profiles.load(tmp_path / f"profiles-{attempt}.ssp").to_dict()
            == profiles.to_dict()
        )

    failing_key = program.bindings["training_templates"][0][0][0]["backward"][
        "profile_key"
    ]

    def fail():
        with pytest.raises(RuntimeError, match="intentional backward"):
            measure_program(program, **options)

    fail()
    # Exception tracebacks may retain frames until the caught exception is released.
    gc.collect()
    released()
