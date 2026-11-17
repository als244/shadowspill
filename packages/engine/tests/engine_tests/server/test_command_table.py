"""Real HTTP/client futures feed the command display; CPU worker fixture only."""

import io

from engine_tests.server.test_jobs import until
from engine_tests.server.test_training_client import records, runtime
from shadowspill_client.display import CommandTable


def test_http_whole_split_optimizer_and_forward_share_one_command_table(tmp_path):
    stream = io.StringIO()
    table = CommandTable(stream=stream)
    with runtime(tmp_path) as (_, training, _, _):
        validation = training.validate_batch(records()).result(5)
        assert validation.estimated_seconds is not None
        futures = [
            table.track(
                "step",
                training.step(records(), optim={"lr": 0.01}),
                planned_seconds=validation.estimated_seconds,
            ),
            table.track("forward_backward", training.forward_backward(records())),
            table.track("optim_step", training.optim_step(optim={"lr": 0.01})),
            table.track("forward", training.forward(records())),
        ]
        results = [pending.result(5) for pending in futures]
        until(lambda: stream.getvalue().count("done") == 4)
        assert [r.operation for r in results] == [
            "step",
            "forward_backward",
            "optim_step",
            "forward",
        ]
        assert results[2].loss is None
        # This controller-only fixture publishes no execution duration. Display
        # must keep it absent, rather than substituting client wall time.
        assert all(r.execution_seconds is None for r in results)
        output = stream.getvalue()
        assert output.count("pending") == output.count("done") == 4
        assert all(row.seconds is None for row in table._rows)
        training.close().result(5)
