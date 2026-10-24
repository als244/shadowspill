"""Exercise loader bindings on an actual built Program, without runtime pools."""

from types import SimpleNamespace

import torch

from shadowspill.artifacts.tensors import tensor_geometry
from shadowspill.execution.tensor_regions import TensorRegion, tensor_regions
from shadowspill.execution.tensors import tensor_entries, visit_tensors


def check_shard_loading(program):
    for rank in range(program.parallelism.world_size):
        state = SimpleNamespace(
            plan=SimpleNamespace(program=program),
            geometry=tensor_geometry(program),
            regions=tensor_regions(program, rank),
            aliases={item.object_id: item.alias_group_id for item in program.objects},
            sizes={
                item.alias_group_id: item.size_bytes for item in program.alias_groups
            },
        )
        values, expected = {}, {}
        entries = tuple(tensor_entries(state))
        for name, value, initializer in entries:
            alias = state.aliases[value]
            if alias in values:
                continue
            region = state.regions.get(value)
            if region is None:
                layout = state.geometry[value]
                dtype = getattr(torch, layout["dtype"].removeprefix("torch."))
                capacity = state.sizes[alias] // dtype.itemsize
                region = TensorRegion(layout, capacity, 0, capacity, capacity)
            dtype = getattr(torch, region.geometry["dtype"].removeprefix("torch."))
            values[alias] = torch.full((region.capacity,), -1, dtype=dtype)
            expected[alias] = torch.zeros_like(values[alias])
            if initializer and initializer["kind"] in {"constant", "value"}:
                expected[alias].fill_(initializer["value"])
            elif not name.startswith("gradient."):
                expected[alias][: region.length] = (
                    torch.arange(region.offset, region.offset + region.length) + 1
                )
        state.initial = set(values)
        state.host = lambda alias, values=values: values[alias].view(torch.uint8)

        def fill(request, tensor):
            assert tensor.numel() * tensor.element_size() <= 12
            assert not request.name.startswith(("master.", "optimizer.", "gradient."))
            tensor.copy_(
                torch.arange(request.offset, request.offset + tensor.numel()) + 1
            )

        visit_tensors(state, fill, 12, initialize=True)
        for alias in values:
            torch.testing.assert_close(values[alias], expected[alias])
