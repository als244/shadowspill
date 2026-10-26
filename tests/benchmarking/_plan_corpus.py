"""Fresh CPU-only corpus evaluation; import guards enforce phase isolation."""

import importlib.abc
import pathlib
import sys


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("shadowspill.build", "shadowspill.execution", "shadowspill.pytorch")
        if fullname == "torch" or fullname.startswith(blocked):
            raise AssertionError(fullname)


def main():
    sys.meta_path.insert(0, Block())
    from benchmarking.planning_eval.config import (
        BandwidthScale,
        FrontierConfig,
        FrontierGrid,
        TransferBandwidthBaseline,
    )
    from benchmarking.planning_eval.worker import evaluate_case

    root, case = map(pathlib.Path, sys.argv[1:3])
    capacity = int(sys.argv[3])
    config = FrontierConfig(
        "phase-test",
        1,
        2,
        30,
        1,
        1,
        "refresh",
        TransferBandwidthBaseline(10**9, 10**9, 10**9, 10**9, "test assumption"),
        (
            FrontierGrid(
                "test",
                (capacity, 2 * capacity),
                (4 * capacity,),
                (BandwidthScale(1, 1),),
            ),
        ),
        deterministic=True,
    )
    result = evaluate_case(
        config=config,
        baseline_directory=root / "plans",
        case_directory=case,
        artifact_store=root / "cache",
        plan_store=None,
        verbose_search=False,
        global_point_base=0,
        global_point_count=2,
        revision="0" * 40,
    )
    assert result["passed"], result
    assert sum(result["counts"].values()) == 2, result
    assert len(list((root / "plans").rglob("annotated_program_plan.json"))) == 2


if __name__ == "__main__":
    main()
