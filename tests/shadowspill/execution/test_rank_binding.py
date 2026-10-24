"""Real CPU collectives verify reordered logical participants and startup errors."""

from datetime import timedelta

import pytest

from shadowspill.communication.binding.ranks import RankAssignment, validate_topology


def test_complete_mapping_and_topology_are_validated():
    assignment = RankAssignment.resolve(4, {0: 2, 1: 0, 2: 3, 3: 1})
    assert assignment.workers((3, 0, 2, 1)) == (2, 1, 0, 3)
    assert RankAssignment.resolve(1, None).logical_by_worker == (0,)
    for mapping in (None, {0: 0}, {0: 0, 1: 0}, {False: 0, 1: 1}):
        with pytest.raises(ValueError, match="rank_map"):
            RankAssignment.resolve(2, mapping)
    validate_topology(
        {"first": (0, 2), "second": (1, 3)}, assignment, ("a", "a", "b", "b")
    )
    with pytest.raises(ValueError, match="across physical hosts"):
        validate_topology(
            {"first": (0, 1), "second": (2, 3)}, assignment, ("a", "a", "b", "b")
        )
    with pytest.raises(ValueError, match="same host"):
        validate_topology({"first": (0, 2), "second": (1, 3)}, assignment, ("a",) * 4)


def _group_worker(worker, rendezvous):
    import torch
    import torch.distributed as dist

    from shadowspill import GroupRef
    from shadowspill.communication.binding.groups import GroupBindings
    from shadowspill.parallelism.reductions import gradient_reduction
    from tests.shadowspill.parallelism.test_reductions import reduction_cases

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        world_size=4,
        rank=worker,
        timeout=timedelta(seconds=30),
    )
    try:
        mapping = {0: 2, 1: 0, 2: 3, 3: 1}
        parameters = reduction_cases()
        reductions = [gradient_reduction(parameter) for parameter in parameters]
        groups = (
            (3, 0),
            (1, 2),
            (3, 2, 1, 0),
            *(
                group
                for reduction in reductions
                for phase in reduction.phases
                for group in phase.groups
            ),
        )
        arguments = dict(
            identity="saved-plan-digest",
            groups=groups,
            rank_map=mapping,
            topology={"node0": (0, 1, 2, 3)},
            backend="gloo",
            timeout=timedelta(seconds=30),
        )
        bound = GroupBindings(**arguments)
        assert bound.logical_rank == mapping[worker]
        reference = GroupRef("pairs", ((3, 0), (1, 2)))
        members = reference.members(bound.logical_rank)
        value = torch.tensor([bound.logical_rank], dtype=torch.int64)
        gathered = [torch.empty_like(value) for _ in members]
        dist.all_gather(gathered, value, group=bound.resolve(reference))
        assert [item.item() for item in gathered] == list(members)
        global_group = bound.for_members((3, 2, 1, 0))
        gathered = [torch.empty_like(value) for _ in range(4)]
        dist.all_gather(gathered, value, group=global_group)
        assert [item.item() for item in gathered] == [3, 2, 1, 0]
        # The derived phases must sum all and only copies of one logical
        # parameter, then retain the range this logical rank actually owns.
        for parameter, reduction in zip(parameters, reductions, strict=True):
            gradient = (
                torch.arange(5, dtype=torch.float32) + (bound.logical_rank + 1) * 100
            )
            result = torch.nn.functional.pad(
                gradient, (0, reduction.padded_elements - 5)
            )
            for phase in reduction.phases:
                reference = GroupRef("reduction", phase.groups)
                group = bound.resolve(reference)
                if phase.operation == "reduce_scatter":
                    output = torch.empty(phase.output_elements)
                    dist.reduce_scatter_single(output, result, group=group)
                    result = output
                else:
                    dist.all_reduce(result, group=group)
            copies = parameter.copies(bound.logical_rank)
            expected = sum(
                torch.arange(5) + (rank + 1) * 100 for rank in copies
            ).float()
            expected = torch.nn.functional.pad(
                expected, (0, reduction.padded_elements - 5)
            )
            index = GroupRef("storage", parameter.gradients.groups).index(
                bound.logical_rank
            )
            expected = expected[
                index * reduction.result_elements : (index + 1)
                * reduction.result_elements
            ]
            torch.testing.assert_close(result, expected)
        bound.close()
        bound.close()  # Owned groups are released once; rendezvous is borrowed.
        assert dist.is_initialized()

        # Each fault must reach every worker before backend collectives start.
        for key, bad in (
            ("rank_map", {0: 0}),
            ("rank_map", {i: i for i in range(4)}),
            ("identity", "different-plan"),
            ("groups", ((0, 1, 2, 3),)),
        ):
            local = {**arguments, **({key: bad} if worker == 0 else {})}
            with pytest.raises(ValueError, match=r"startup|different Plans"):
                GroupBindings(**local)
        with pytest.raises(ValueError, match="same host"):
            GroupBindings(**{**arguments, "topology": {"a": (0, 1), "b": (2, 3)}})
    finally:
        dist.destroy_process_group()


def test_real_gloo_preserves_logical_order_after_worker_permutation(tmp_path):
    import torch.multiprocessing as mp

    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(_group_worker, args=(rendezvous,), nprocs=4, join=True)
