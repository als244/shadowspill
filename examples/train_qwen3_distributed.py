"""Qwen3 MoE full training from a client connected to an existing service."""

from examples.training.qwen import cli


def arguments(argv=None):
    return cli.arguments(argv, family="qwen3moe")


def main(argv=None):
    return cli.main(argv, family="qwen3moe")


if __name__ == "__main__":
    raise SystemExit(main())
