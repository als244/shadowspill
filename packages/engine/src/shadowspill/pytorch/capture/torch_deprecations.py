"""One PyTorch deprecation that fires from inside PyTorch's own code paths."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import torch.fx as fx

#: PyTorch deprecates ``torch.utils._pytree.LeafSpec`` by decorating the class
#: itself, so *constructing* one warns. Export constructs them while building a
#: graph module's input and output specs, and deepcopy reconstructs them when
#: copying that graph module. Both are PyTorch calling PyTorch: the specs are
#: PyTorch's, the construction is PyTorch's, and there is no supported way to
#: export or copy without them. The warning tells a ShadowSpill caller nothing
#: they can act on, so it does not reach them.
#:
#: When the class is finally removed this stops being a warning and starts
#: being an error, which no filter hides.
_LEAF_SPEC_DEPRECATION = r".*isinstance\(treespec, LeafSpec\).*"


@contextmanager
def quiet_leaf_spec_deprecation() -> Iterator[None]:
    """Suppress PyTorch's LeafSpec deprecation for one PyTorch call."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=_LEAF_SPEC_DEPRECATION, category=FutureWarning
        )
        yield


def copy_graph_module[T: fx.GraphModule](graph_module: T) -> T:
    """Return an independent copy of one exported graph module."""

    with quiet_leaf_spec_deprecation():
        return copy.deepcopy(graph_module)


__all__ = ["copy_graph_module", "quiet_leaf_spec_deprecation"]
