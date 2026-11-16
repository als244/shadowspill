"""Run with only shadowspill-client and NumPy on the client machine."""

import argparse

from shadowspill_client import (
    SSH,
    BatchConfig,
    Datum,
    Memory,
    Model,
    Objective,
    Optimizer,
    ServiceClient,
    Weights,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-host", help="SSH alias of the running server")
    args = parser.parse_args()
    data = [
        Datum(
            {"features": row},
            {"target": target, "weight": 0.25},  # Mean over four real examples.
        )
        for row, target in (
            ([1.0, 0, 0, 0], 1.0),
            ([0, 1.0, 0, 0], 2.0),
            ([0, 0, 1.0, 0], -1.0),
            ([0, 0, 0, 1.0], 0.5),
        )
    ]
    if args.ssh_host:
        service = ServiceClient(ssh=SSH(args.ssh_host, server_port=8000))
    else:
        service = ServiceClient(base_url="http://127.0.0.1:8000")
    with service:
        training = service.create_training_client(
            model=Model(
                "linear_regressor",
                config={"width": 4},
                initialize=Weights.random(seed=0),
            ),
            objective=Objective("weighted_mse"),
            optimizer=Optimizer("sgd"),
            batch=BatchConfig(limits={"max_examples": 4}),
            memory=Memory(device_gib=1, spill_gib=1),
        ).result()
        try:
            result = training.step(data, optim={"lr": 0.05}).result()
            print("step", result.completed_updates, result.loss)
            training.forward_backward(data[:2]).result()
            training.forward_backward(data[2:]).result()
            result = training.optim_step(optim={"lr": 0.05}).result()
            print("split update", result.completed_updates)
            checkpoint = training.save_checkpoint(name="after-two-updates").result()
            plans = training.plans
        finally:
            training.close().result()
        training = service.create_training_client(
            plans=plans, checkpoint=checkpoint
        ).result()
        try:
            result = training.step(data, optim={"lr": 0.025}).result()
            print("resumed", result.completed_updates, result.loss)
        finally:
            training.close().result()


if __name__ == "__main__":
    main()
