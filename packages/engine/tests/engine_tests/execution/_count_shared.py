"""Execute newly instantiated counts using only the source compiled task forms."""

import torch

from shadowspill.training.observations import to_host


def qualify_count_family(executor, plans, oracle, snapshot, fill):
    executor.initialize(fill, chunk_bytes=32)
    for name, expected in zip(
        ("count_repeat", "count_mixed", "count_backward"),
        oracle["count_updates"],
        strict=True,
    ):
        examples = [oracle["examples"][index] for index in expected["indices"]]
        if name == "count_backward":
            first = executor.run(plan=name, args=examples[:2]).result()
            first = to_host((first.losses, first.metrics))
            last = executor.run(
                plan="backward1", args=examples[2:], accumulate_gradients=True
            ).result()
            last = to_host((last.losses, last.metrics))
            losses, metrics = (*first[0], *last[0]), (*first[1], *last[1])
            executor.run(
                plan="optim",
                hyperparams={
                    plans["optim"].program.bindings["gradient_scale_control"]: expected[
                        "scale"
                    ]
                },
            ).result()
        else:
            result = executor.run(
                plan=name,
                args=examples,
                hyperparams={
                    plans[name].program.bindings["gradient_scale_control"]: expected[
                        "scale"
                    ]
                },
            ).result()
            losses, metrics = to_host((result.losses, result.metrics))
        for loss, prediction, observed_loss, observed in zip(
            expected["losses"], expected["predictions"], losses, metrics, strict=True
        ):
            torch.testing.assert_close(observed_loss, loss)
            torch.testing.assert_close(observed["prediction"], prediction)
        actual = snapshot(executor)
        for key, value in expected["state"].items():
            torch.testing.assert_close(actual[key].reshape(value.shape), value)
