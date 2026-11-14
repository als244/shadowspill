"""Pass only explicit CLI choices to the common saved-run configuration."""

import argparse


def arguments(parser, argv):
    directory = parser.get_default("run_dir")
    for action in parser._actions:
        if action.dest != "help":
            action.default = argparse.SUPPRESS
    values = vars(parser.parse_args(argv))
    if directory and "run_dir" not in values:
        values["run_dir"] = directory
    return values.pop("phase", "all"), values
