import dataclasses
import json
import runpy
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils._pytree import tree_flatten
import shadowspill.pytorch.profiling.profiler.workspace as module

real = module.output_allocation_views

def save(boundary, output, *, inputs=()):
    try:
        return real(boundary, output, inputs=inputs)
    except Exception:
        rows = []
        for index, leaf in enumerate(tree_flatten(output)[0]):
            if isinstance(leaf, torch.Tensor) and leaf.is_cuda:
                address = leaf.untyped_storage().data_ptr()
                if address:
                    a = boundary.allocation_for_pointer(address)
                    rows.append(dict(leaf=index,shape=tuple(leaf.shape),stride=tuple(leaf.stride()),dtype=str(leaf.dtype),
                                     storage_bytes=leaf.untyped_storage().nbytes(),data_offset=leaf.data_ptr()-int(a.pointer or 0),
                                     storage_offset=leaf.storage_offset(),allocation_id=int(a.allocation_id),
                                     allocation_bytes=int(a.requested_bytes)))
        Path(__file__).resolve().parents[1].joinpath("evidence/olmoe_view_failure.json").write_text(json.dumps(rows,indent=2))
        raise

with patch.object(module, "output_allocation_views", save):
    runpy.run_path(str(Path(__file__).with_name("full_model_lora.py")), run_name="__main__")
