"""A saved generic update keeps matrix-shaped state and independent CPU math."""

import pytest
import torch

from shadowspill import build_step
from tests.shadowspill.execution._ownership_fixture import Network, objective, oracle
from tests.shadowspill.execution.helpers import run_worker, save_plan
from tests.shadowspill.execution.optimizers.matrix import MatrixNormalizedMomentum


@pytest.mark.cuda
def test_matrix_optimizer_saved_worker_matches_combined_replica_inputs(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    fixture = oracle("matrix")
    calls = [call for rank_calls in fixture["examples"] for call in rank_calls]
    initial = fixture["initial_by_rank"][0]
    template = Network()
    template.load_state_dict(initial)
    source = {
        name: (p._version, p.data_ptr()) for name, p in template.named_parameters()
    }
    optimizer = MatrixNormalizedMomentum(template.parameters())
    program = build_step(
        template,
        objective=objective,
        optimizer=optimizer,
        example_args=calls[0][0],
        example_kwargs=calls[0][1],
        num_examples=len(calls),
    )
    assert not optimizer.state
    for name, parameter in template.named_parameters():
        assert (parameter._version, parameter.data_ptr()) == source[name]
        torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
    save_plan(program, tmp_path)
    expected = []
    for index, record in enumerate(fixture["expected"]):
        state = record["states"][0]
        # One process evaluates both replicas' inputs, so its local counter
        # advances four times. The reduced gradient/update is the same.
        state["counter"] = initial["counter"] + (index + 1) * len(calls)
        outputs = [
            (loss, {}) for rank_losses in record["outputs"] for loss in rank_losses
        ]
        expected.append((outputs, state))
    torch.save(
        dict(
            initial=initial,
            examples=calls,
            expected=expected,
            hyperparams=[{}] * 3,
            accumulate=[False] * 3,
            restore=False,
        ),
        tmp_path / "oracle.pt",
    )
    run_worker(tmp_path)
