"""Optional fixed-batch learning report for the recipe finish hook."""

import json

from workloads.recipes.text.diagnostics import fixed_batch_learning_report


def learning_report(root, options, losses, rank):
    threshold = options["minimum_relative_loss_drop"]
    if options["repeat_training_batch"] and len(losses) >= 4:
        report = fixed_batch_learning_report(
            losses, minimum_relative_drop=threshold or 0
        )
        if rank == 0:
            (root / "learning.json").write_text(json.dumps(report, indent=2) + "\n")
            print("fixed-batch learning: " + json.dumps(report), flush=True)
        if threshold is not None and not report["passed"]:
            raise RuntimeError(
                "fixed-batch loss did not meet the requested decrease; "
                "see learning.json"
            )
    elif threshold is not None:
        raise ValueError("loss-drop check needs four newly executed updates")
